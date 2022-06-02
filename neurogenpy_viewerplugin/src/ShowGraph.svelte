<div class="column left">
    <GraphDisplay bind:this={gd} result={result} on:NodeSelected={ev => nodeLabel = ev.detail }></GraphDisplay>
</div>
<div class="column right">
    {#if nodeLabel}
        <NodeInfo nodeLabel={nodeLabel} marginals={result['marginals']}
                  on:CommunitiesSelected={ev => handleCommunities(ev.detail)}></NodeInfo>
    {/if}
</div>


<style>
    .left {
        width: 67%;
        height: 100%;
        overflow: hidden;
        background-color: rgb(33, 33, 37);
        border: 1px solid #907c7c;
        margin: 10px 10px 10px 10px;
    }

    .column {
        float: left;
        position: relative;
    }

    .right {
        width: 30%;
        padding: 10px;
    }

</style>


<script>
    import NodeInfo from "./NodeInfo.svelte";
    import GraphDisplay from "./GraphDisplay.svelte";

    export let result
    let nodeLabel = undefined
    let gd

    function handleCommunities(e) {
        gd.communitiesLouvain(e)
    }
</script>
