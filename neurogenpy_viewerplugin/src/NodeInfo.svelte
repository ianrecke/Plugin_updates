<script>
    import Card, { Media, Content } from "@smui/card";
    import Button, { Label } from "@smui/button";
    import NormalDistribution from "normal-distribution";
    import { Chart } from "chart.js/dist/chart";
    import { afterUpdate, createEventDispatcher, onMount } from "svelte";
    import Textfield from "@smui/textfield";
    import quantile from "@stdlib/stats-base-dists-normal-quantile";

    const dispatcher = createEventDispatcher();

    export let initialMarginals;
    export let evidenceMarginals = undefined;
    export let nodeLabel;

    let evidence = {};
    let evidenceSelected = "";
    let currentValue = undefined;
    let previousNode = undefined;

    let distLabels;
    let initialValues;
    let evidenceValues;

    let chart;
    let chartRef;
    let ctx;

    onMount(() => {
        ctx = chartRef.getContext("2d");

        chart = new Chart(ctx, {
            type: "line",
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
                elements: {
                    point: {
                        radius: 0,
                    },
                },
                plugins: {
                    legend: {
                        display: !!evidenceValues,
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

    afterUpdate(() => {
        updateChartValues();

        chart.data.labels = distLabels;
        chart.data.datasets[0].data = initialValues;
        chart.data.datasets[1].data = evidenceValues;
        chart.options.plugins.legend.display = !!evidenceValues;

        chart.update();
    });

    $: nodeLabel, labelChange();

    function labelChange() {
        if (previousNode && currentValue) {
            evidence[previousNode] = currentValue;
        } else if (previousNode && previousNode in evidence) {
            delete evidence[previousNode];
        }
        previousNode = nodeLabel;

        evidenceSelected = "";
        if (nodeLabel in evidence) {
            currentValue = evidence[nodeLabel];
        } else {
            currentValue = undefined;
        }
    }

    function getDistValues(marginals) {
        let mean = marginals[nodeLabel]["mu"];
        let variance = marginals[nodeLabel]["sigma"];

        const dist = new NormalDistribution(mean, variance);
        const stDev = Math.sqrt(variance);
        const minValue = quantile(0.001, mean, stDev);
        const maxValue = quantile(0.999, mean, stDev);

        return { dist: dist, min: minValue, max: maxValue };
    }

    function updateChartValues() {
        const initialData = getDistValues(initialMarginals);

        let realMin = initialData["min"];
        let realMax = initialData["max"];

        let evidenceData;
        if (evidenceMarginals && !(nodeLabel in evidence)) {
            evidenceData = getDistValues(evidenceMarginals);
            realMin = Math.min(realMin, evidenceData["min"]);
            realMax = Math.max(realMax, evidenceData["max"]);
        }

        let step = (realMax - realMin) / 200;
        distLabels = Array.from({ length: 201 }, (_, i) => realMin + step * i);
        initialValues = distLabels.map((i) => initialData["dist"].pdf(i));

        if (evidenceData) {
            evidenceValues = distLabels.map((i) => evidenceData["dist"].pdf(i));
        } else {
            evidenceValues = undefined;
        }
    }

    async function getRelated(method) {
        dispatcher("GetRelated", method);
    }

    function performInference() {
        if (currentValue) {
            evidence[nodeLabel] = currentValue;
        }
        dispatcher("PerformInference", evidence);
    }

    function setEvidence() {
        const parseSelected = parseFloat(evidenceSelected);
        if (!isNaN(parseSelected)) {
            currentValue = parseSelected;
        }
    }
</script>

<Card style="padding: 10px">
    <Media style="padding: 10px 10px 0 10px">
        <div>
            <div>
                <canvas id="myChart" bind:this={chartRef} style="width: 100%" />
            </div>
            <pre
                class="status"
                style="color: white; width: {evidenceMarginals && !(nodeLabel in evidence)? '50%': '100%'}; float: left;">Mean: {initialMarginals[
                    nodeLabel
                ]["mu"].toFixed(3)} - Variance: {initialMarginals[nodeLabel][
                    "sigma"
                ].toFixed(3)}</pre>
            {#if evidenceMarginals && !(nodeLabel in evidence)}
                <pre
                    class="status"
                    style="color: rgb(255, 62, 0); width: 50%">Mean: {evidenceMarginals[
                        nodeLabel
                    ]["mu"].toFixed(3)} - Variance: {evidenceMarginals[
                        nodeLabel
                    ]["sigma"].toFixed(3)}</pre>{/if}
            <h2 class="mdc-typography--headline6" style="margin-left: 5px">
                {nodeLabel}
            </h2>
        </div>
    </Media>

    <Content style="padding: 0 16px">
        <div class="mdc-typography--body2">
            <Textfield
                class="textfieldClass"
                bind:value={evidenceSelected}
                label="Set evidence"
            />
            <Button on:click={setEvidence}>
                <Label>Set</Label>
            </Button>
            {#if currentValue}
                <Button
                    on:click={() => {
                        currentValue = undefined;
                        evidenceSelected = "";
                    }}
                >
                    <Label>Clear</Label>
                </Button>
            {/if}
            {#if currentValue || Object.keys(evidence).length > 0}
                <Button on:click={performInference}>
                    <Label>Run Inference</Label>
                </Button>
            {/if}
            <pre class="status" style="color: white">Value: {currentValue}</pre>
        </div>
        <Button on:click={() => getRelated("mb")}>
            <Label>Markov blanket</Label>
        </Button>

        <Button on:click={() => getRelated("reachable")}>
            <Label>Reachable nodes</Label>
        </Button>
    </Content>
</Card>
