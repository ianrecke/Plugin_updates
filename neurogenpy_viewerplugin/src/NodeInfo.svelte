<script>
    import Card, { Media, Content } from "@smui/card";
    import Button, { Label } from "@smui/button";
    import NormalDistribution from "normal-distribution";
    import { Chart } from "chart.js/dist/chart";
    import { afterUpdate, createEventDispatcher, onMount } from "svelte";
    import Textfield from "@smui/textfield";
    import { NEUROGENPY_ENDPOINT } from "./store";

    const dispatcher = createEventDispatcher();

    export let nodeLabel = undefined;
    export let jsonBN = undefined;

    let evidence = "";
    let currentValue = undefined;

    let distLabels;
    let values;

    let chart;
    let chartRef;
    let ctx;
    let runningFlag = false;

    onMount(() => {
        ctx = chartRef.getContext("2d");

        distLabels = [];
        values = [];

        chart = new Chart(ctx, {
            type: "line",
            data: {
                labels: distLabels,
                datasets: [
                    {
                        label: "Distribution",
                        data: values,
                        fill: true,
                        borderColor: "#fff",
                        backgroundColor: "rgba(255, 255, 255, 0.2)",
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
                        display: false,
                    },
                },
            },
        });
    });

    afterUpdate(() => {
        if (!chart) return;

        // let mean = marginals[nodeLabel]['mu']
        // let std_deviation = marginals[nodeLabel]['sigma']
        let mean = 3;
        let std_deviation = 0.4;

        const normDist = new NormalDistribution(mean, std_deviation);

        let minValue = mean - 3 * std_deviation;
        let maxValue = mean + 3 * std_deviation;
        let step = (maxValue - minValue) / 80;

        distLabels = Array.from({ length: 81 }, (_, i) =>
            (minValue + step * i).toFixed(2)
        );
        values = distLabels.map((i) => normDist.pdf(i));

        chart.data.labels = distLabels;
        chart.data.datasets[0].data = values;

        chart.update();
    });

    async function getMB() {
        if (runningFlag) {
            console.warn(
                `Markov blanket already running, do not start a new one.`
            );
            return;
        }
        runningFlag = true;
        let mb = [];

        try {
            const res = await fetch(`${NEUROGENPY_ENDPOINT}/mb/mb`, {
                method: "POST",
                body: jsonBN,
                headers: {
                    "content-type": "application/json",
                },
            });
            if (res.status >= 400) {
                throw new Error(res.statusText);
            }
            const { poll_url } = await res.json();

            mb = await new Promise((rs, rj) => {
                const intervalRef = setInterval(async () => {
                    const res = await fetch(
                        `${NEUROGENPY_ENDPOINT}/mb/mb/${poll_url}`
                    );
                    if (res.status >= 400) {
                        return rj(res.statusText);
                    }
                    const { status, result } = await res.json();
                    if (status === "SUCCESS") {
                        console.log("SUCCESS", result);
                        clearInterval(intervalRef);
                        rs(result);
                    }
                    if (status === "FAILURE") {
                        console.log("FAILURE");
                        clearInterval(intervalRef);
                        rj("operation failed");
                    }
                }, 1000);
            });
        } finally {
            runningFlag = false;
            dispatcher("HighlightNodes", mb);
        }
    }
</script>

<Card style="padding: 10px; position: relative;">
    <Media>
        <div>
            <div>
                <canvas id="myChart" bind:this={chartRef} style="width: 100%" />
            </div>

            <div class="spacer" />

            <h2 class="mdc-typography--headline6">
                {nodeLabel}
            </h2>
        </div>
    </Media>

    <Content>
        <div class="mdc-typography--body2">
            <Textfield
                class="textfieldClass"
                bind:value={evidence}
                label="Set evidence"
            />
            <Button on:click={() => (currentValue = evidence)}>
                <Label>Set</Label>
            </Button>
            {#if currentValue}
                <Button on:click={() => (currentValue = undefined)}>
                    <Label>Clear</Label>
                </Button>
            {/if}
            <pre class="status" style="color: white">Value: {currentValue}</pre>
        </div>
        <Button on:click={getMB}>
            <Label>Markov blanket</Label>
        </Button>

        <Button>
            <Label>Reachable nodes</Label>
        </Button>
    </Content>
</Card>
