<script>
    import NodeInfo from "./NodeInfo.svelte";
    import GraphDisplay from "./GraphDisplay.svelte";
    import BnManipulations from "./BNManipulations.svelte";
    import { saveTextFile } from "./saveFile";
    import { getFromNeurogenpy } from "./request";
    import CircularProgress from "@smui/circular-progress";

    export let result;
    const nodes = Object.keys(result["marginals"]);
    const initialMarginals = result["marginals"];
    const gexf_graph = result["gexf"];
    let evidenceMarginals;

    let nodeLabel = undefined;
    let gd;
    let runningFlag = false;

    async function downloadFile(fileType) {
        let positions = {};
        switch (fileType) {
            case "png":
                gd.savePNG();
                break;
            case "gexf":
                positions = gd.getPositions();
            default:
                const json_object = JSON.stringify({
                    file_type: fileType,
                    positions: positions,
                });

                const result = await callNeurogenpy(
                    "download",
                    "/grn/download",
                    json_object
                );

                const downloadableFile = result["result"];
                saveTextFile(downloadableFile, "result." + fileType);
                break;
        }
    }

    async function performInference(evidence) {
        const json_object = JSON.stringify({
            marginals: initialMarginals,
            evidence: evidence,
        });

        const result = await callNeurogenpy(
            "Inference",
            "/grn/inference",
            json_object
        );

        evidenceMarginals = result["marginals"];
    }

    async function getRelated(method) {
        const json_object = JSON.stringify({
            node: nodeLabel,
            method: method,
        });
        const result = await callNeurogenpy(
            "Related nodes",
            "/grn/related",
            json_object
        );

        gd.setHighlightedNodes(result["result"], false);
    }

    async function callNeurogenpy(process, path, json_object) {
        if (runningFlag) {
            console.warn(process + `already running, do not start a new one.`);
            return;
        }
        runningFlag = true;
        let result;
        try {
            result = await getFromNeurogenpy(path, json_object);
        } finally {
            runningFlag = false;
        }
        return result;
    }

    async function neurogenpyLayout(layoutName) {
        const json_object = JSON.stringify({
            layout: layoutName,
        });
        const result = await callNeurogenpy(
            "Layout",
            "/grn/layout",
            json_object
        );

        const layoutResult = result["result"];
        const newLP = {};
        Object.entries(layoutResult).map(
            ([key, value]) =>
                (newLP[key] = {
                    x: value[0],
                    y: value[1],
                })
        );
        gd.setLayout(layoutName, newLP);
    }
</script>

<window on:mouseup on:resize />

<div id="container">
    <div class="column left" disabled={runningFlag}>
        <GraphDisplay
            bind:this={gd}
            {nodes}
            {gexf_graph}
            on:NodeSelected={(ev) => (nodeLabel = ev.detail)}
            on:NeurogenpyLayout={(ev) => neurogenpyLayout(ev.detail)}
        />
    </div>
    <div class="column right" disabled={runningFlag}>
        <div id="manipulations">
            <BnManipulations
                {nodes}
                on:SaveFile={(ev) => downloadFile(ev.detail)}
                on:CommunitiesSelected={(ev) =>
                    gd.communitiesLouvain(ev.detail)}
            />
        </div>

        {#if nodeLabel}
            <div id="nodeInfo">
                <NodeInfo
                    {nodeLabel}
                    {initialMarginals}
                    {evidenceMarginals}
                    on:GetRelated={(ev) => getRelated(ev.detail)}
                    on:PerformInference={(ev) => {
                        performInference(ev.detail);
                    }}
                />
            </div>
        {/if}
    </div>
    {#if runningFlag}
        <div id="progress">
            <CircularProgress style="width:3rem;height:3rem;" indeterminate />
        </div>
    {/if}
</div>

<style>
    #container {
        display: flex;
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
        right: 0;
        align-items: center;
        padding: 15px;
    }

    .left {
        width: 70%;
        height: 100%;
        position: relative;
        overflow: hidden;
        float: left;
        background-color: rgb(33, 33, 37);
        border: 1px solid #907c7c;
        margin-right: 20px;
    }

    .right {
        width: 30%;
        height: 100%;
        position: relative;
    }

    #progress {
        display: flex;
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
        right: 0;
        align-items: center;
        justify-content: center;
    }

    #manipulations {
        position: absolute;
    }

    #nodeInfo {
        position: absolute;
        width: 100%;
        bottom: 0;
    }
</style>
