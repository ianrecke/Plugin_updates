<script>
    import { Chart } from "chart.js/dist/chart";
    import { onMount, afterUpdate } from "svelte";

    let chartRef;
    let ctx;
    let chart;

    export let initialDist;
    export let evidenceDist;

    let initialValues;
    let evidenceValues;
    let distLabels;

    onMount(() => {
        ctx = chartRef.getContext("2d");

        chart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: distLabels,
                datasets: [
                    {
                        label: "Initial",
                        data: initialValues,
                        fill: true,
                        borderColor: "#fff",
                        backgroundColor: "rgba(255, 255, 255, 0.2)",
                    },
                    {
                        label: "Evidence",
                        hidden: !evidenceValues,
                        data: evidenceValues,
                        fill: true,
                        borderColor: "rgb(255, 62, 0)",
                        backgroundColor: "rgba(255, 62, 0, 0.2)",
                    },
                ],
            },
            options: {
                responsive: false,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: !!evidenceValues,
                    },
                },
            },
        });
    });

    afterUpdate(() => {
        evidenceValues = undefined;
        distLabels = Object.keys(initialDist);
        initialValues = Object.values(initialDist);
        if (evidenceDist)
            evidenceValues = distLabels.map((i) => evidenceDist[i]);

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
