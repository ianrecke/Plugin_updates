<script>
    import NodeInfo from "./NodeInfo.svelte";
    import GraphDisplay from "./GraphDisplay.svelte";
    import BnManipulations from "./BNManipulations.svelte";
    import { saveTextFile } from "./saveFile";

    export let result;
    let nodes = result["json_graph"]["nodes"].map((elem) => elem.key);
    const result_json = JSON.stringify(result);

    let nodeLabel = undefined;
    let gd;

    async function downloadFile(fileType) {
        switch (fileType) {
            case "json":
                saveTextFile(result_json, "result.json");
                break;
            case "gexf":
                saveTextFile(result["gexf"], "result.gexf");
                break;
            case "png":
                gd.savePNG();
                break;
        }
    }
</script>

<div class="column left">
    <GraphDisplay
        bind:this={gd}
        {nodes}
        graph_gexf={result["gexf"]}
        on:NodeSelected={(ev) => (nodeLabel = ev.detail)}
    />
</div>
<div class="column right">
    <BnManipulations
        {nodes}
        on:SaveFile={(ev) => downloadFile(ev.detail)}
        on:CommunitiesSelected={(ev) => gd.communitiesLouvain(ev.detail)}
    />

    <div class="spacer" />

    {#if nodeLabel}
        <NodeInfo {nodeLabel} jsonBN={result_json} />
    {/if}
</div>

<style>
    .left {
        width: 65%;
        height: calc(100% - 30px);
        position: absolute;
        overflow: hidden;
        float: left;
        background-color: rgb(33, 33, 37);
        border: 1px solid #907c7c;
    }

    .right {
        margin-left: 67%;
        height: 100%;
        position: relative;
    }

    div.spacer {
        height: 1rem;
    }
</style>
