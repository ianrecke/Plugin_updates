<script>
    import NodeInfo from "./NodeInfo.svelte";
    import GraphDisplay from "./GraphDisplay.svelte";
    import BnManipulations from "./BNManipulations.svelte";
    import { saveTextFile } from "./saveFile";
    import { getFromNeurogenpy } from "./request";
    import CircularProgress from "@smui/circular-progress";
    import Help from "./Help.svelte";
    import Card from "@smui/card";

    export let result;
    export let dataType;
    const nodes = Object.keys(result["marginals"]);
    const initialMarginals = result["marginals"];
    const gexf_graph = result["gexf"];
    let evidenceMarginals;
    let json_bn = result["json_bn"];
    let fileTypes = {
        discrete: ["json", "gexf", "png", "csv", "bif"],
        continuous: ["json", "gexf", "png", "csv"],
    };

    let nodeLabel = undefined;
    let gd;
    let runningFlag = false;

    async function downloadFile(fileType) {

        switch (fileType) {
            case "png":
                gd.savePNG();
                break;
            case "gexf":
                const downloadableGEXF = gd.getGEXF();
                console.log(downloadableGEXF);
                saveTextFile(downloadableGEXF, "result.gexf");
            default:
                const json_object = JSON.stringify({
                    json_bn: json_bn,
                    file_type: fileType,
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
            json_bn: json_bn,
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
            json_bn,
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
            json_bn: json_bn,
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

<div id="container">
    <div class="column borderSurfaceDiv" id="left" disabled={runningFlag}>
        <GraphDisplay
            bind:this={gd}
            {nodes}
            {gexf_graph}
            on:NodeSelected={(ev) => (nodeLabel = ev.detail)}
            on:NeurogenpyLayout={(ev) => neurogenpyLayout(ev.detail)}
        />
    </div>
    <div class="column" id="right" disabled={runningFlag}>
        <div id="manipulations">
            <BnManipulations
                fileTypes={fileTypes[dataType]}
                {nodes}
                {json_bn}
                on:SaveFile={(ev) => downloadFile(ev.detail)}
                on:CommunitiesSelected={(ev) =>
                    gd.communitiesLouvain(ev.detail)}
            />
        </div>

        <div id="nodeInfo">
            <Card style="padding: 10px; height:100%">
                {#if nodeLabel}
                    <NodeInfo
                        {nodeLabel}
                        {initialMarginals}
                        {evidenceMarginals}
                        {dataType}
                        on:GetRelated={(ev) => getRelated(ev.detail)}
                        on:PerformInference={(ev) => {
                            performInference(ev.detail);
                        }}
                    />
                {:else}
                    <h2 style="margin: auto">No gene selected.</h2>
                {/if}
            </Card>
        </div>
    </div>
    {#if runningFlag}
        <div id="progress">
            <CircularProgress style="width:3rem;height:3rem;" indeterminate />
        </div>
    {/if}
    <div id="help"><Help /></div>
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

    #left {
        width: 70%;
        height: 100%;
        position: relative;
        overflow: hidden;
        float: left;
        margin-right: 20px;
    }

    #right {
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
        position: relative;
        height: 45%;
        top: 0;
    }

    #nodeInfo {
        position: relative;
        height: 55%;
        width: 100%;
        bottom: 0;
    }

    #help {
        position: absolute;
        top: 20px;
        right: 10px;
    }
</style>
