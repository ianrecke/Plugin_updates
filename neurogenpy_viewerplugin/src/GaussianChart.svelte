<script>
    import NormalDistribution from "normal-distribution";
    import { Chart } from "chart.js/dist/chart";
    import { onMount, afterUpdate } from "svelte";
    import quantile from "@stdlib/stats-base-dists-normal-quantile";
    import { toRgba } from "./color";

    let chartRef;
    let chart;
    let ctx;

    export let initialDist;
    export let evidenceDist;

    const style = getComputedStyle(document.body);
    const evidenceColor = style.getPropertyValue("--primary");
    const evidenceColorT = toRgba(evidenceColor, 0.2);
    const initColor = style.getPropertyValue("color");
    const initColorT = toRgba(initColor, 0.2);

    let initialValues;
    let evidenceValues;
    let distLabels;
    let lines = {};

    onMount(() => {
        ctx = chartRef.getContext("2d");

        chart = new Chart(ctx, {
            type: "line",
            data: {
                labels: [-2, 0, 2],
                datasets: [
                    {
                        label: "Initial",
                        data: [0, 1, 0],
                        fill: true,
                        borderColor: initColor,
                        backgroundColor: initColorT,
                    },
                    {
                        label: "Evidence",
                        hidden: !evidenceValues,
                        data: [0, 1, 0],
                        fill: true,
                        borderColor: evidenceColor,
                        backgroundColor: evidenceColorT,
                    },
                ],
            },
            options: {
                responsive: false,
                maintainAspectRatio: true,
                elements: {
                    point: {
                        radius: 0,
                    },
                },
                plugins: {
                    legend: {
                        display: !!evidenceValues,
                    },
                    annotation: {
                        annotations: lines,
                    },
                },
                scales: {
                    x: {
                        ticks: {
                            callback: function (value) {
                                return this.getLabelForValue(value).toFixed(3);
                            },
                        },
                    },
                    y: {
                        ticks: {
                            callback: function (value) {
                                return value.toFixed(2);
                            },
                        },
                    },
                },
            },
        });
    });

    function getDistValues(mean, variance) {
        const dist = new NormalDistribution(mean, variance);
        const stDev = Math.sqrt(variance);
        const minValue = quantile(0.001, mean, stDev);
        const maxValue = quantile(0.999, mean, stDev);

        return { dist: dist, min: minValue, max: maxValue };
    }

    function updateChartValues() {
        evidenceValues = undefined;

        const initialData = getDistValues(
            initialDist["mu"],
            initialDist["sigma"]
        );

        let realMin = initialData["min"];
        let realMax = initialData["max"];

        let evidenceData;
        if (evidenceDist) {
            evidenceData = getDistValues(
                evidenceDist["mu"],
                evidenceDist["sigma"]
            );
            realMin = Math.min(realMin, evidenceData["min"]);
            realMax = Math.max(realMax, evidenceData["max"]);
        }

        let step = (realMax - realMin) / 200;
        distLabels = Array.from({ length: 201 }, (_, i) => realMin + step * i);
        initialValues = distLabels.map((i) => initialData["dist"].pdf(i));

        if (evidenceData) {
            evidenceValues = distLabels.map((i) => evidenceData["dist"].pdf(i));
        }
    }

    afterUpdate(() => {
        updateChartValues();

        chart.data.labels = distLabels;
        chart.data.datasets[0].data = initialValues;
        chart.data.datasets[1].data = evidenceValues;
        chart.options.plugins.legend.display = !!evidenceValues;
        chart.data.datasets[1].hidden = !evidenceValues;

        chart.update();
    });
</script>

<div>
    <canvas id="myChart" bind:this={chartRef} style="width: 100%" />
</div>
<pre class="status" style="width: 50%; float: left;">Mean: {initialDist[
        "mu"
    ].toFixed(3)} - Variance: {initialDist["sigma"].toFixed(3)}</pre>

<pre
    class="status primaryText"
    style="width: 50%"> {#if evidenceDist}Mean: {evidenceDist["mu"].toFixed(
            3
        )} - Variance: {evidenceDist["sigma"].toFixed(3)}
    {/if}</pre>
