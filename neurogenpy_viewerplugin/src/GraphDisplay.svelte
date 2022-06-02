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

<style>
    #zoom {
        position: absolute;
        left: 5px;
        bottom: 0;
    }

    #controls {
        position: absolute;
        right: 15px;
        top: 0;
    }

    button {
        background-color: transparent;
        background-repeat: no-repeat;
        border: none;
        outline: none;
        margin: 0 0 0 0;
    }

    .sigmaElement {
        position: absolute;
        top: 5px;
        bottom: 5px;
        left: 5px;
        right: 5px;
        overflow: hidden;
    }

    button:hover {
        color: white;
    }

    button:active {
        background-color: transparent;
    }
</style>

<script>
    import {afterUpdate, createEventDispatcher, onMount} from "svelte";
    import sigma from "sigma";
    import {drawHover} from "./utils";
    import forceAtlas2 from "graphology-layout-forceatlas2";
    import FA2Layout from "graphology-layout-forceatlas2/worker";
    import {circular} from "graphology-layout";
    import {animateNodes} from "sigma/utils/animate";
    import Graph from "graphology";
    import Select, {Option} from "@smui/select";
    import FormField from "@smui/form-field";
    import Checkbox from "@smui/checkbox";
    import louvain from 'graphology-communities-louvain';

    export let result = undefined
    const graph = new Graph();
    const dispatcher = createEventDispatcher()

    let container = undefined
    let camera = undefined
    let showLabels = true
    let renderer = undefined

    let layouts = [`ForceAtlas2`, `Circular`];
    let selectedLayout = undefined
    let fa2Layout = undefined

    let hoveredNode = undefined
    let hoveredNeighbors = undefined
    let cancelCurrentAnimation = null

    onMount(() => {
        graph.addNode("John", {x: 0, y: 10, size: 50, label: "John", color: "blue"});
        graph.addNode("Brad", {x: 4, y: 8, size: 20, label: "Brad", color: "yellow"});
        graph.addNode("Tom", {x: 5, y: 5, size: 40, label: "Tom", color: "green"});
        graph.addNode("Mary", {x: 10, y: 0, size: 30, label: "Mary", color: "red"});

        graph.addDirectedEdge("John", "Mary", {size: 4});
        // graph.addDirectedEdge("Mary", "Brad", {size: 4});
        graph.addDirectedEdge("John", "Tom", {size: 4});
        // const graph_json = JSON.parse(result['result'])
        // graph.import(graph_json)

        renderer = new sigma(graph, container, {
            minCameraRatio: 0.1,
            maxCameraRatio: 10,
            hoverRenderer: drawHover,
            defaultEdgeType: "arrow",
        });

        renderer.setSetting('labelColor', {'color': '#fff'})
        renderer.setSetting("nodeReducer", (node, data) => {
            const res = {...data};

            if (hoveredNeighbors && !hoveredNeighbors.has(node) && hoveredNode !== node) {
                res.label = "";
                res.color = 'rgb(33, 33, 37)';
            }

            return res;
        });

        renderer.setSetting("edgeReducer", (edge, data) => {
            const res = {...data}

            if (hoveredNode && !graph.hasExtremity(edge, hoveredNode)) {
                res.hidden = true;
            }

            return res;
        });
        renderer.on("clickNode", (e) => nodeClicked(e))
        renderer.on("enterNode", ({node}) => {
            setHoveredNode(node);
            // const mouseLayer = document.querySelector(".sigma-mouse");
            // if (mouseLayer) mouseLayer.classList.add("mouse-pointer");
        })
        renderer.on("leaveNode", () => {
            setHoveredNode(undefined);
        });
        camera = renderer.getCamera();

        const sensibleSettings = forceAtlas2.inferSettings(graph);
        fa2Layout = new FA2Layout(graph, {
            settings: sensibleSettings,
        });
    })

    afterUpdate(() => communitiesLouvain())

    function nodeClicked(e) {
        dispatcher('NodeSelected', graph.getNodeAttribute(e.node, "label"))
    }

    function changeLabels() {
        renderer.setSetting('renderLabels', !showLabels)
        renderer.refresh()
    }

    function setHoveredNode(node) {
        if (node) {
            hoveredNode = node;
            hoveredNeighbors = new Set(graph.neighbors(node));
        } else {
            hoveredNode = undefined;
            hoveredNeighbors = undefined;
        }

        renderer.refresh();
    }

    function changeLayout() {
        switch (selectedLayout) {
            case "Circular":
                circularLayout()
                break
            case "ForceAtlas2":
                toggleFA2Layout()
                break
        }
    }

    function stopFA2() {
        fa2Layout.stop();
    }

    function startFA2() {
        if (cancelCurrentAnimation) cancelCurrentAnimation();
        fa2Layout.start();
    }

    function toggleFA2Layout() {
        if (fa2Layout.isRunning()) {
            stopFA2();
        } else {
            startFA2();
        }
    }

    function circularLayout() {
        if (fa2Layout.isRunning()) stopFA2();
        if (cancelCurrentAnimation) cancelCurrentAnimation();

        const circularPositions = circular(graph, {scale: 100});
        cancelCurrentAnimation = animateNodes(graph, circularPositions, {duration: 2000, easing: "linear"});
    }

    export function communitiesLouvain(printComs) {
        if (printComs) {
            const communities = louvain(graph);

            const distinctComs = [...new Set(Object.keys(communities).map((key, index) => communities[key]))];
            const colors = Object.assign({}, ...distinctComs.map((x) => ({[x]: generateRandomColor()})));

            graph.forEachNode((node, attributes) => graph.setNodeAttribute(node, "color", colors[node]))
            renderer.refresh()
        }
    }

    function generateRandomColor() {
        const letters = '0123456789ABCDEF';
        let color = '#';
        for (let i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }

</script>