<Card style="padding: 10px; position: relative;">
    <Media>
        <div>
            <div>
                <canvas id="myChart" bind:this={chartRef} style="width: 100%"></canvas>
            </div>

            <div class="spacer"></div>

            <h2 class="mdc-typography--headline6">
                {nodeLabel}
            </h2>

            <div class="spacer"></div>

            <div class="mdc-typography--body2">
                <Textfield class="textfieldClass" bind:value={evidence} label="Set evidence"></Textfield>
                <button class="mdc-typography--button" on:click={() => currentValue = evidence}>Set</button>
                <pre class="status" style="color: white">Value: {currentValue}</pre>
                {#if currentValue}
                    <button class="mdc-typography--button" on:click={() => currentValue = undefined}>Clear</button>
                {/if}
            </div>
            <button class="mdc-typography--button">
                Markov blanket
            </button>
        </div>
    </Media>
</Card>

<div class="spacer"></div>

<Card style="padding: 10px">
    <button on:click={() => dispatcher('CommunitiesSelected', true)}>
        Communities Louvain
    </button>
</Card>


<style>
    div.spacer {
        height: 1rem;
    }
</style>


<script>
    import Card, {Media} from '@smui/card';
    import NormalDistribution from "normal-distribution";
    import {Chart} from 'chart.js/dist/chart';
    import {afterUpdate, createEventDispatcher, onMount} from "svelte";
    import Textfield from "@smui/textfield";

    const dispatcher = createEventDispatcher()

    export let nodeLabel = undefined

    let evidence = ''
    let currentValue = undefined

    export let marginals = undefined

    let distLabels
    let values

    let chart
    let chartRef
    let ctx

    onMount(() => {
        ctx = chartRef.getContext('2d');

        distLabels = []
        values = []

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: distLabels,
                datasets: [{
                    label: 'Distribution',
                    data: values,
                    fill: true,
                    borderColor: "#fff",
                    backgroundColor: 'rgba(255, 255, 255, 0.2)'
                }]
            },
            options: {
                responsive: false,
                maintainAspectRatio: true,
                elements: {
                    point: {
                        radius: 0
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                }
            }
        })
    })

    afterUpdate(() => {
        if (!chart) return;

        let mean = marginals[nodeLabel]['mu']
        let std_deviation = marginals[nodeLabel]['sigma']

        const normDist = new NormalDistribution(mean, std_deviation);

        let minValue = mean - 3 * std_deviation
        let maxValue = mean + 3 * std_deviation
        let step = (maxValue - minValue) / 40

        distLabels = Array.from({length: 41}, (_, i) => (minValue + step * i).toFixed(2))
        values = distLabels.map((i) => normDist.pdf(i))

        chart.data.labels = distLabels
        chart.data.datasets[0].data = values

        chart.update()
    });
</script>