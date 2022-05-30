<div class="row">
    <div class="column left">
        <div bind:this={container} class="sigmaElement">
        </div>
        <div id="zoom">
            <button on:click={() => camera.animatedZoom({duration: 600})}>
                <i class="material-icons">add</i>
            </button>
            <button on:click={() => camera.animatedUnzoom({duration: 600})}>
                <i class="material-icons">remove</i>
            </button>
            <button on:click={() => camera.animatedReset({duration: 600})}>
                <i class="material-icons">replay</i>
            </button>
        </div>
        <div id="controls">
            <Select bind:value={selectedLayout} label="Select layout" on:SMUISelect:change={changeLayout}>
                {#each layouts as layout}
                    <Option value={layout}>
                        {layout}
                    </Option>
                {/each}
            </Select>
            <FormField>
                <Checkbox bind:checked={showLabels} on:click={changeLabels}/>
                <span slot="label" style="color: white">Labels</span>
            </FormField>
        </div>
    </div>
    <div class="column right">
        {#if nodeLabel}
            <NodeInfo {nodeLabel}></NodeInfo>
        {/if}
    </div>
</div>


<style>
    #zoom {
        position: absolute;
        left: 0.5em;
        bottom: 0;
    }

    #controls {
        position: absolute;
        right: 0.5em;
        top: 0.5em;
        text-align: right;
    }

    .sigmaElement {
        position: absolute;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }

    button {
        background-color: transparent;
        background-repeat: no-repeat;
        border: none;
        outline: none;
        border-radius: 50%;
    }

    button:hover {
        color: white;
    }

    button:active {
        background-color: transparent;
    }

    .left {
        position: relative;
        border: 1px solid #907c7c;
        background-color: rgba(33, 33, 37, 100);
        width: 80%;
        height: 100%;
    }

    .column {
        float: left;
        padding: 10px;
    }

    .right {
        width: 20%;
    }

    .row:after {
        content: "";
        display: table;
        clear: both;
    }
</style>


<script>
    import sigma from "sigma"
    import Graph from "graphology"
    import Select, {Option} from "@smui/select"
    import Checkbox from '@smui/checkbox';
    import FormField from '@smui/form-field';
    import {onMount} from "svelte";
    import {circular} from "graphology-layout";
    import {animateNodes} from "sigma/utils/animate";
    import NodeInfo from "./NodeInfo.svelte";


    export let result = undefined
    const graph = new Graph();

    let container = undefined
    let camera = undefined
    let showLabels = true
    let renderer = undefined

    let nodeLabel = undefined

    let layouts = [`ForceAtlas2`, `Circular`];
    let selectedLayout = undefined

    onMount(() => {

        graph.addNode("John", {x: 0, y: 10, size: 50, label: "John", color: "blue"});
        graph.addNode("Brad", {x: 4, y: 8, size: 20, label: "Brad", color: "yellow"});
        graph.addNode("Tom", {x: 5, y: 5, size: 40, label: "Tom", color: "green"});
        graph.addNode("Mary", {x: 10, y: 0, size: 30, label: "Mary", color: "red"});

        graph.addEdge("John", "Mary");
        graph.addEdge("Mary", "Brad");
        graph.addEdge("John", "Tom");
        // const graph = parse(Graph, result["result"]);

        renderer = new sigma(graph, container, {
            enableEdgeClickEvents: true,
            minCameraRatio: 0.1,
            maxCameraRatio: 10,
        });

        renderer.setSetting('labelColor', {'color': '#fff'})
        renderer.on("clickNode", (e) => nodeClicked(e))
        camera = renderer.getCamera();
    })

    function nodeClicked(e) {
        nodeLabel = graph.getNodeAttribute(e.node, "label")
    }

    function changeLabels() {
        renderer.setSetting('renderLabels', !showLabels)
        renderer.refresh()
    }

    function changeLayout() {
        switch (selectedLayout) {
            case "Circular":
                circularLayout()
        }
    }

    function circularLayout() {
        // if (fa2Layout.isRunning()) stopFA2();
        // if (cancelCurrentAnimation) cancelCurrentAnimation();

        //since we want to use animations we need to process positions before applying them through animateNodes
        const circularPositions = circular(graph, {scale: 100});
        //In other context, it's possible to apply the position directly we : circular.assign(graph, {scale:100})
        animateNodes(graph, circularPositions, {duration: 2000, easing: "linear"});
    }


</script>
