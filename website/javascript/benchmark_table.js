
//Formatter to generate charts
var chartFormatter = function (cell, formatterParams, onRendered) {
    var content = document.createElement("span");
    var values = cell.getValue();

    //invert values if needed
    if (formatterParams.invert) {
        values = values.map(val => val * -1);
    }

    //add values to chart and style
    content.classList.add(formatterParams.type);
    content.inneHrTML = values.join(",");

    //setup chart options
    var options = {
        width: 50,
        // min: 0.0,
        // max: 100.0,
    }

    if (formatterParams.fill) {
        options.fill = formatterParams.fill
    }

    //instantiate piety chart after the cell element has been aded to the DOM
    onRendered(function () {
        peity(content, formatterParams.type, options);
    });

    return content;
};

var colorFormatterGoalInt = function (cell, formatterParams) {
    var value = cell.getValue();

    // Check for the specific string "-"
    if (value === "-") {
        return value;
    }

    // Default values
    var defaults = {
        min: 0.0,
        max: 100.0,
        startColor: { r: 255, g: 255, b: 255 },
        endColor: { r: 238, g: 211, b: 217 }
    };

    // Override defaults with provided formatterParams values
    var min = (formatterParams && formatterParams.min) || defaults.min;
    var max = (formatterParams && formatterParams.max) || defaults.max;
    var startColor = (formatterParams && formatterParams.startColor) || defaults.startColor;
    var endColor = (formatterParams && formatterParams.endColor) || defaults.endColor;

    // Normalize the value between 0 and 1
    var normalizedValue = (value - min) / (max - min);

    // Compute the color gradient 
    var red = Math.floor(startColor.r + (endColor.r - startColor.r) * normalizedValue);
    var green = Math.floor(startColor.g + (endColor.g - startColor.g) * normalizedValue);
    var blue = Math.floor(startColor.b + (endColor.b - startColor.b) * normalizedValue);

    // make sure the value is rounded to 1 decimal place
    value = parseFloat(value).toFixed(1)

    return "<span style='display: block; width: 100%; height: 100%; background-color: rgb(" + red + ", " + green + ", " + blue + ");'>" + value + "</span>";
}

var colorFormatterSubgoal = function (cell, formatterParams) {
    var value = cell.getValue();

    // Check for the specific string "-"
    if (value === "-") {
        return value;
    }

    // Default values
    var defaults = {
        min: 0.0,
        max: 100.0,
        startColor: { r: 255, g: 255, b: 255 },
        endColor: { r: 245, g: 232, b: 221 }
    };

    // Override defaults with provided formatterParams values
    var min = (formatterParams && formatterParams.min) || defaults.min;
    var max = (formatterParams && formatterParams.max) || defaults.max;
    var startColor = (formatterParams && formatterParams.startColor) || defaults.startColor;
    var endColor = (formatterParams && formatterParams.endColor) || defaults.endColor;

    // Normalize the value between 0 and 1
    var normalizedValue = (value - min) / (max - min);

    // Compute the color gradient 
    var red = Math.floor(startColor.r + (endColor.r - startColor.r) * normalizedValue);
    var green = Math.floor(startColor.g + (endColor.g - startColor.g) * normalizedValue);
    var blue = Math.floor(startColor.b + (endColor.b - startColor.b) * normalizedValue);

    // make sure the value is rounded to 1 decimal place
    value = parseFloat(value).toFixed(1)

    return "<span style='display: block; width: 100%; height: 100%; background-color: rgb(" + red + ", " + green + ", " + blue + ");'>" + value + "</span>";
}

var colorFormatterActionSeq = function (cell, formatterParams) {
    var value = cell.getValue();

    // Check for the specific string "-"
    if (value === "-") {
        return value;
    }

    // Default values
    var defaults = {
        min: 0.0,
        max: 100.0,
        startColor: { r: 255, g: 255, b: 255 },
        endColor: { r: 204, g: 211, b: 202 }
    };

    // Override defaults with provided formatterParams values
    var min = (formatterParams && formatterParams.min) || defaults.min;
    var max = (formatterParams && formatterParams.max) || defaults.max;
    var startColor = (formatterParams && formatterParams.startColor) || defaults.startColor;
    var endColor = (formatterParams && formatterParams.endColor) || defaults.endColor;

    // Normalize the value between 0 and 1
    var normalizedValue = (value - min) / (max - min);

    // Compute the color gradient 
    var red = Math.floor(startColor.r + (endColor.r - startColor.r) * normalizedValue);
    var green = Math.floor(startColor.g + (endColor.g - startColor.g) * normalizedValue);
    var blue = Math.floor(startColor.b + (endColor.b - startColor.b) * normalizedValue);

    // make sure the value is rounded to 1 decimal place
    value = parseFloat(value).toFixed(1)

    return "<span style='display: block; width: 100%; height: 100%; background-color: rgb(" + red + ", " + green + ", " + blue + ");'>" + value + "</span>";
}

var colorFormatterTrans = function (cell, formatterParams) {
    var value = cell.getValue();

    // Check for the specific string "-"
    if (value === "-") {
        return value;
    }

    // Default values
    var defaults = {
        min: 0.0,
        max: 100.0,
        startColor: { r: 255, g: 255, b: 255 },
        endColor: { r: 181, g: 192, b: 208 }
    };

    // Override defaults with provided formatterParams values
    var min = (formatterParams && formatterParams.min) || defaults.min;
    var max = (formatterParams && formatterParams.max) || defaults.max;
    var startColor = (formatterParams && formatterParams.startColor) || defaults.startColor;
    var endColor = (formatterParams && formatterParams.endColor) || defaults.endColor;

    // Normalize the value between 0 and 1
    var normalizedValue = (value - min) / (max - min);

    // Compute the color gradient 
    var red = Math.floor(startColor.r + (endColor.r - startColor.r) * normalizedValue);
    var green = Math.floor(startColor.g + (endColor.g - startColor.g) * normalizedValue);
    var blue = Math.floor(startColor.b + (endColor.b - startColor.b) * normalizedValue);

    // make sure the value is rounded to 1 decimal place
    value = parseFloat(value).toFixed(1)

    return "<span style='display: block; width: 100%; height: 100%; background-color: rgb(" + red + ", " + green + ", " + blue + ");'>" + value + "</span>";
}



var barColorFn = function (value, formatterParams) {
    var defaults = {
        range : [-50, 50],
        low: { r: 255, g: 255, b: 255 },
        high: { r: 206, g: 212, b: 218 }
    };

    // Override defaults with provided formatterParams values

    var low_range = (formatterParams && formatterParams.range[0]) || defaults.range[0];
    var high_range = (formatterParams && formatterParams.range[1]) || defaults.range[1];
    var low = (formatterParams && formatterParams.low) || defaults.low;
    var high = (formatterParams && formatterParams.high) || defaults.high;

    // Clamp the value to the range [-100, 100]
    value = Math.max(low_range, Math.min(high_range, value));
    var range = high_range - low_range;

    // Normalize the value to the range [0, 1]
    var normalizedValue = (value + range / 2) / range;
    // Interpolate between the two colors based on the normalized value
    var interpolated = {
        r: Math.floor(low.r + (high.r - low.r) * normalizedValue),
        g: Math.floor(low.g + (high.g - low.g) * normalizedValue),
        b: Math.floor(low.b + (high.b - low.b) * normalizedValue)
    };

    return 'rgba(' + interpolated.r + ',' + interpolated.g + ',' + interpolated.b + ',0.9)';
}

document.addEventListener('DOMContentLoaded', function () {
    Promise.all([
        fetch('website/data/virtualhome_total_benchmark.json').then(response => response.json()),
        fetch('website/data/behavior_total_benchmark.json').then(response => response.json()),
    ])
        .then(([
            virtualhome_total_benchmark_data,
            behavior_total_benchmark_data,
        ]) => {
            var getColumnMinMax = (data, field) => {
                let values = data.map(item => item[field]).filter(val => val !== "-").map(Number);
                return { min: Math.min(...values), max: Math.max(...values) };
            };

            var virtualhome_columns = [
                {
                    title: "Model Family",
                    field: "model",
                    widthGrow: 1.5,
                    minWidth: 180
                },
                {
                    title: "Access",
                    field: "access",
                    widthGrow: 0.9,
                    minWidth: 120
                },
                {
                    title: "Release<br>Date",
                    field: "release",
                    widthGrow: 0.9,
                    minWidth: 120
                },
                {
                    title: "Overall<br>Perf.",
                    field: "overall_performance",
                    formatter: "progress",
                    minWidth: 90,
                    formatterParams: {
                        min: 0, max: 80,
                        legend: true,
                        color: barColorFn,
                    },
                },
                {
                    title: "Goal<br>Interpretation",
                    columns: [{
                        title: "F1",
                        field: "goal_interpretation_f1",
                        hozAlign: "center",
                        formatter: colorFormatterGoalInt,
                        minWidth: 90
                    }]
                },
                {
                    title: "Action Sequencing",
                    columns: [
                        { title: "Task<br>SR", field: "action_sequencing_task_sr", hozAlign: "center", formatter: colorFormatterActionSeq, minWidth: 90, responsive: 2},
                        { title: "Exec.<br>SR", field: "action_sequencing_execution_sr", hozAlign: "center", formatter: colorFormatterActionSeq, minWidth: 90, responsive: 2 },
                    ]
                },
                {
                    title: "Subgoal Decomposition",
                    columns: [
                        { title: "Task<br>SR", field: "subgoal_decomposition_task_sr", hozAlign: "center", formatter: colorFormatterSubgoal, minWidth: 90 },
                        { title: "Exec.<br>SR", field: "subgoal_decomposition_execution_sr", hozAlign: "center", formatter: colorFormatterSubgoal, minWidth: 90 },
                    ]
                },
                {
                    title: "Transition Modeling",
                    columns: [
                        { title: "F1", field: "transition_modeling_f1", hozAlign: "center", formatter: colorFormatterTrans, minWidth: 90 },
                        { title: "Planner<br>SR", field: "transition_modeling_planner_sr", hozAlign: "center", formatter: colorFormatterTrans, minWidth: 90 },
                    ]
                },
            ];

            virtualhome_columns.forEach(column => {
                if (column.columns) {
                    column.columns.forEach(subColumn => {
                        let { min, max } = getColumnMinMax(virtualhome_total_benchmark_data, subColumn.field);
                        subColumn.formatterParams = { min, max };
                    });
                } else if (column.field !== "overall_performance") {
                    let { min, max } = getColumnMinMax(virtualhome_total_benchmark_data, column.field);
                    column.formatterParams = { min, max };
                }
            });

            var virtualhome_table = new Tabulator("#virtualhome-benchmark-main-table", {
                data: virtualhome_total_benchmark_data,
                layout: "fitColumns",
                responsiveLayout: "collapse",
                responsiveLayoutCollapseStartOpen: false,
                movableColumns: false,
                initialSort: [
                    { column: "overall_performance", dir: "desc" },
                ],
                columnDefaults: {
                    tooltip: true,
                },
                columns: virtualhome_columns
            });

            var behavior_columns = [
                {
                    title: "Model Family",
                    field: "model",
                    widthGrow: 1.5,
                    minWidth: 180
                },
                {
                    title: "Access",
                    field: "access",
                    widthGrow: 0.9,
                    minWidth: 120
                },
                {
                    title: "Release<br>Date",
                    field: "release",
                    widthGrow: 0.9,
                    minWidth: 120
                },
                {
                    title: "Overall<br>Perf.",
                    field: "overall_performance",
                    formatter: "progress",
                    minWidth: 90,
                    formatterParams: {
                        min: 0, max: 80,
                        legend: true,
                        color: barColorFn,
                    },
                },
                {
                    title: "Goal<br>Interpretation",
                    columns: [{
                        title: "F1",
                        field: "goal_interpretation_f1",
                        hozAlign: "center",
                        formatter: colorFormatterGoalInt,
                        minWidth: 90
                    }]
                },
                {
                    title: "Action Sequencing",
                    columns: [
                        { title: "Task<br>SR", field: "action_sequencing_task_sr", hozAlign: "center", formatter: colorFormatterActionSeq, minWidth: 90 },
                        { title: "Exec.<br>SR", field: "action_sequencing_execution_sr", hozAlign: "center", formatter: colorFormatterActionSeq, minWidth: 90 },
                    ]
                },
                {
                    title: "Subgoal Decomposition",
                    columns: [
                        { title: "Task<br>SR", field: "subgoal_decomposition_task_sr", hozAlign: "center", formatter: colorFormatterSubgoal, minWidth: 90 },
                        { title: "Exec.<br>SR", field: "subgoal_decomposition_execution_sr", hozAlign: "center", formatter: colorFormatterSubgoal, minWidth: 90 },
                    ]
                },
                {
                    title: "Transition Modeling",
                    columns: [
                        { title: "F1", field: "transition_modeling_f1", hozAlign: "center", formatter: colorFormatterTrans, minWidth: 90 },
                        { title: "Planner<br>SR", field: "transition_modeling_planner_sr", hozAlign: "center", formatter: colorFormatterTrans, minWidth: 90 },
                    ]
                },
            ];

            behavior_columns.forEach(column => {
                if (column.columns) {
                    column.columns.forEach(subColumn => {
                        let { min, max } = getColumnMinMax(behavior_total_benchmark_data, subColumn.field);
                        subColumn.formatterParams = { min, max };
                    });
                } else if (column.field !== "overall_performance") {
                    let { min, max } = getColumnMinMax(behavior_total_benchmark_data, column.field);
                    column.formatterParams = { min, max };
                }
            });

            var behavior_table = new Tabulator("#behavior-benchmark-main-table", {
                data: behavior_total_benchmark_data,
                layout: "fitColumns",
                responsiveLayout: "collapse",
                responsiveLayoutCollapseStartOpen: false,
                movableColumns: false,
                initialSort: [
                    { column: "overall_performance", dir: "desc" },
                ],
                columnDefaults: {
                    tooltip: true,
                },
                columns: behavior_columns
            });
        });
})

