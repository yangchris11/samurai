set -x

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=8

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python ./scripts/main_inference_chunk.py \
        --chunk_idx $IDX \
        --num_chunks $CHUNKS &
done

wait