<script>
    import Card, { Media, Content } from "@smui/card";
    import Button, { Label } from "@smui/button";
    import { createEventDispatcher } from "svelte";
    import Textfield from "@smui/textfield";
    import Select, { Option } from "@smui/select";
    import GaussianChart from "./GaussianChart.svelte";
    import DiscreteChart from "./DiscreteChart.svelte";

    const dispatcher = createEventDispatcher();

    export let initialMarginals;
    export let evidenceMarginals = undefined;
    export let nodeLabel;
    export let dataType;

    let evidence = {};
    let evidenceSelected = "";
    let currentValue = undefined;
    let previousNode = undefined;

    let initialDist = initialMarginals[nodeLabel];
    let evidenceDist = undefined;

    $: {
        if (previousNode && currentValue !== undefined)
            evidence[previousNode] = currentValue;
        else if (previousNode && previousNode in evidence)
            delete evidence[previousNode];

        previousNode = nodeLabel;

        initialDist = initialMarginals[nodeLabel];
        if (evidenceMarginals && nodeLabel in evidenceMarginals)
            evidenceDist = evidenceMarginals[nodeLabel];
        else evidenceDist = undefined;

        evidenceSelected = "";
        if (nodeLabel in evidence) currentValue = evidence[nodeLabel];
        else currentValue = undefined;
    }

    async function getRelated(method) {
        dispatcher("GetRelated", method);
    }

    function performInference() {
        if (currentValue !== undefined) evidence[nodeLabel] = currentValue;
        else if (nodeLabel in evidence) delete evidence[nodeLabel];

        if (Object.keys(evidence).length > 0)
            dispatcher("PerformInference", evidence);
        else evidenceMarginals = undefined;

        if (evidenceMarginals && nodeLabel in evidenceMarginals)
            evidenceDist = evidenceMarginals[nodeLabel];
        else evidenceDist = undefined;
    }

    function setEvidence() {
        switch (dataType) {
            case "discrete":
                currentValue = evidenceSelected;
                break;
            case "continuous":
                const parseSelected = parseFloat(evidenceSelected);
                if (!isNaN(parseSelected)) {
                    currentValue = parseSelected;
                }
                break;
        }
    }
</script>

<Card style="padding: 10px">
    <Media style="padding: 10px 10px 0 10px">
        <div>
            {#if dataType === "continuous"}
                <GaussianChart
                    evidenceSet={currentValue}
                    {initialDist}
                    {evidenceDist}
                />
            {:else}
                <DiscreteChart {initialDist} {evidenceDist} />
            {/if}
            <h2 class="mdc-typography--headline6" style="margin-left: 5px">
                {nodeLabel}
            </h2>
        </div>
    </Media>

    <Content style="padding: 0 16px">
        <div class="mdc-typography--body2">
            {#if dataType === "continuous"}
                <Textfield
                    class="textfieldClass"
                    bind:value={evidenceSelected}
                    label="Set evidence"
                />
            {:else}
                <Select bind:value={evidenceSelected} label="Set evidence">
                    {#each Object.keys(initialMarginals[nodeLabel]) as category}
                        <Option value={category}>
                            {category}
                        </Option>
                    {/each}
                </Select>
            {/if}
            <Button on:click={setEvidence}>
                <Label>Set</Label>
            </Button>
            <Button
                disabled={currentValue === undefined}
                on:click={() => {
                    currentValue = undefined;
                    evidenceSelected = "";
                }}
            >
                <Label>Clear</Label>
            </Button>
            <Button
                disabled={currentValue === undefined &&
                    Object.keys(evidence).length === 0}
                on:click={performInference}
            >
                <Label>Infer</Label>
            </Button>
            <Button
                disabled={currentValue === undefined &&
                    Object.keys(evidence).length === 0}
                on:click={() => {
                    currentValue = undefined;
                    evidence = {};
                }}
            >
                <Label>Reset</Label>
            </Button>
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
