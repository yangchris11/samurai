"""In-memory session repository.

Sessions are stored for the lifetime of the process. GPU memory is released
when a session is explicitly closed via ``remove_session``.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sam2.sam2_video_predictor import SAM2VideoPredictor

from api.exceptions import SessionNotFoundError
from api.ml.inference import InferenceState


@dataclass
class SessionRecord:
    """All runtime state associated with one tracking session.

    Attributes:
        session_id: Unique session identifier.
        video_id: ID of the video this session tracks.
        inference_state: Opaque SAM2 state dict (owns GPU tensors).
        num_frames: Total frame count in the video.
        created_at: UTC creation timestamp.
        canceled: Set to ``True`` to abort an in-progress propagation.
    """

    session_id: str
    video_id: str
    inference_state: InferenceState
    num_frames: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    canceled: bool = False


class SessionRepository:
    """Thread-safe in-memory store for active tracking sessions.

    Attributes:
        _sessions: Map from session_id to ``SessionRecord``.
        _lock: Mutex protecting all mutations of ``_sessions``.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: SessionRecord) -> None:
        """Store a new session record.

        Args:
            record: Fully initialised session to store.
        """

        with self._lock:
            self._sessions[record.session_id] = record

    def get(self, session_id: str) -> SessionRecord:
        """Return a session by ID.

        Args:
            session_id: Unique session identifier.

        Returns:
            The matching ``SessionRecord``.

        Raises:
            SessionNotFoundError: If no session with this ID exists.
        """

        with self._lock:
            record = self._sessions.get(session_id)

        if record is None:
            raise SessionNotFoundError(session_id)

        return record

    def remove(
        self,
        session_id: str,
        predictor: SAM2VideoPredictor,
    ) -> bool:
        """Remove a session and release its GPU memory.

        Args:
            session_id: Unique session identifier.
            predictor: Predictor used to reset the inference state.

        Returns:
            ``True`` if the session existed and was removed, ``False`` otherwise.
        """

        with self._lock:
            record = self._sessions.pop(session_id, None)

        if record is None:
            return False

        predictor.reset_state(record.inference_state)
        return True

    def mark_canceled(self, session_id: str) -> None:
        """Signal an active propagation to stop at the next frame.

        Args:
            session_id: Unique session identifier.

        Raises:
            SessionNotFoundError: If no session with this ID exists.
        """

        record = self.get(session_id)
        record.canceled = True

    def clear_canceled(self, session_id: str) -> None:
        """Reset the cancellation flag before starting a new propagation.

        Args:
            session_id: Unique session identifier.

        Raises:
            SessionNotFoundError: If no session with this ID exists.
        """

        record = self.get(session_id)
        record.canceled = False
