<script>
    import NormalDistribution from "normal-distribution";
    import { Chart } from "chart.js/dist/chart";
    import { onMount, afterUpdate } from "svelte";
    import quantile from "@stdlib/stats-base-dists-normal-quantile";

    let chartRef;
    let chart;
    let ctx;

    export let initialDist;
    export let evidenceDist;
    export let evidenceSet;

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
                        borderColor: "#fff",
                        backgroundColor: "rgba(255, 255, 255, 0.2)",
                    },
                    {
                        label: "Evidence",
                        hidden: !evidenceValues,
                        data: [0, 1, 0],
                        fill: true,
                        borderColor: "rgb(255, 62, 0)",
                        backgroundColor: "rgba(255, 62, 0, 0.2)",
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
        if (evidenceSet) {
        }
        // else

        chart.update();
    });
</script>

<div>
    <canvas id="myChart" bind:this={chartRef} style="width: 100%" />
</div>
<pre
    class="status"
    style="color: white; width: {evidenceDist
        ? '50%'
        : '100%'}; float: left;">Mean: {initialDist["mu"].toFixed(
        3
    )} - Variance: {initialDist["sigma"].toFixed(3)}</pre>
{#if evidenceDist}
    <pre
        class="status"
        style="color: rgb(255, 62, 0); width: 50%">Mean: {evidenceDist[
            "mu"
        ].toFixed(3)} - Variance: {evidenceDist["sigma"].toFixed(3)}</pre>{/if}
