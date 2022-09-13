<script>
    import { Chart } from "chart.js/dist/chart";
    import { onMount, afterUpdate } from "svelte";
    import { toRgba } from "./color";

    let chartRef;
    let ctx;
    let chart;

    export let initialDist;
    export let evidenceDist;
    export let evidence;

    let initialValues;
    let evidenceValues;
    let distLabels;

    const style = getComputedStyle(document.body);
    const evidenceColor = style.getPropertyValue("--primary");
    const evidenceColorT = toRgba(evidenceColor, 0.2);
    const initColor = style.getPropertyValue("color");
    const initColorT = toRgba(initColor, 0.2);

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
                        borderColor: initColor,
                        backgroundColor: initColorT,
                    },
                    {
                        label: "Evidence",
                        hidden: !evidenceValues,
                        data: evidenceValues,
                        fill: true,
                        borderColor: evidenceColor,
                        backgroundColor: evidenceColorT,
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

        if (evidence) evidenceValues = distLabels.map((i) => +(i === evidence));
        else if (evidenceDist)
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
