<script>
    import { createEventDispatcher, onMount } from "svelte";
    import sigma from "./sigma";
    import { drawHover } from "./hover";
    import forceAtlas2 from "graphology-layout-forceatlas2";
    import FA2Layout from "graphology-layout-forceatlas2/worker";
    import { circular } from "graphology-layout";
    import { animateNodes } from "./sigma/utils/animate";
    import Graph from "graphology";
    import { parse } from "graphology-gexf/browser";
    import Select, { Option } from "@smui/select";
    import FormField from "@smui/form-field";
    import Switch from "@smui/switch";
    import louvain from "graphology-communities-louvain";
    import { watchResize } from "svelte-watch-resize";
    import NodeSelection from "./NodeSelection.svelte";
    import { saveAsPNG } from "./saveFile";

    export let graph_gexf = undefined;
    const graph = parse(Graph, graph_gexf);

    const dispatcher = createEventDispatcher();

    let container = undefined;
    let camera = undefined;

    let showLabels = true;
    let renderer = undefined;

    const layouts = [`ForceAtlas2`, `Circular`];
    let selectedLayout = undefined;
    let fa2Layout = undefined;

    let hoveredNode = undefined;
    let hoveredNeighbors = undefined;
    let cancelCurrentAnimation = null;

    let fullScreen = false;
    let display;

    export let nodes = [];
    onMount(() => {
        renderer = new sigma(graph, container, {
            minCameraRatio: 0.1,
            maxCameraRatio: 10,
            hoverRenderer: drawHover,
            defaultEdgeType: "arrow",
            allowInvalidContainer: true,
        });

        renderer.setSetting("labelColor", { color: "#fff" });
        renderer.setSetting("nodeReducer", (node, data) => {
            const res = { ...data };

            if (
                hoveredNeighbors &&
                !hoveredNeighbors.has(node) &&
                hoveredNode !== node
            ) {
                res.label = "";
                res.color = "rgb(33, 33, 37)";
            }

            res.highlighted = node === hoveredNode;

            return res;
        });

        renderer.setSetting("edgeReducer", (edge, data) => {
            const res = { ...data };

            if (hoveredNode && !graph.hasExtremity(edge, hoveredNode)) {
                res.hidden = true;
            }

            return res;
        });

        renderer.on("clickNode", (e) => {
            nodeClicked(e);
        });
        renderer.on("enterNode", ({ node }) => {
            setHoveredNode(node);
        });
        renderer.on("leaveNode", () => {
            setHoveredNode(undefined);
        });
        camera = renderer.getCamera();

        const sensibleSettings = forceAtlas2.inferSettings(graph);
        fa2Layout = new FA2Layout(graph, {
            settings: sensibleSettings,
        });
    });

    function nodeClicked(e) {
        dispatcher("NodeSelected", graph.getNodeAttribute(e.node, "label"));
    }

    function changeLabels() {
        renderer.setSetting("renderLabels", showLabels);
        renderer.refresh();
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

    function changeFullScreen() {
        let doc = display.ownerDocument;
        if (fullScreen) {
            if (doc.exitFullscreen) {
                doc.exitFullscreen();
            } else if (doc.webkitExitFullscreen) {
                doc.webkitExitFullscreen();
            } else if (doc.mozCancelFullScreen) {
                doc.mozCancelFullScreen();
            } else if (doc.msExitFullscreen) {
                doc.msExitFullscreen();
            }
        } else {
            if (display.requestFullscreen) {
                display.requestFullscreen();
            } else if (display.msRequestFullscreen) {
                display.msRequestFullscreen();
            } else if (display.mozRequestFullScreen) {
                display.mozRequestFullScreen();
            } else if (display.webkitRequestFullscreen) {
                display.webkitRequestFullscreen();
            }
        }

        fullScreen = !fullScreen;
    }

    function changeLayout() {
        switch (selectedLayout) {
            case "Circular":
                circularLayout();
                break;
            case "ForceAtlas2":
                toggleFA2Layout();
                break;
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

        const circularPositions = circular(graph, { scale: 100 });
        cancelCurrentAnimation = animateNodes(graph, circularPositions, {
            duration: 2000,
            easing: "linear",
        });
    }

    export function communitiesLouvain(printComs) {
        if (printComs) {
            const communities = louvain(graph);

            const distinctComs = [
                ...new Set(
                    Object.keys(communities).map(
                        (key, index) => communities[key]
                    )
                ),
            ];
            const colors = Object.assign(
                {},
                ...distinctComs.map((x) => ({ [x]: generateRandomColor() }))
            );

            graph.forEachNode((node, attributes) =>
                graph.setNodeAttribute(node, "color", colors[communities[node]])
            );
            renderer.refresh();
        }
    }

    function generateRandomColor() {
        const letters = "0123456789ABCDEF";
        let color = "#";
        for (let i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }

    export function savePNG() {
        saveAsPNG(renderer);
    }
</script>

<div bind:this={display}>
    <div
        bind:this={container}
        class="sigmaElement"
        use:watchResize={() => renderer.refresh()}
    />
    <div id="zoom">
        <button on:click={() => camera.animatedZoom({ duration: 600 })}>
            <i class="material-icons">add</i>
        </button>
        <button on:click={() => camera.animatedUnzoom({ duration: 600 })}>
            <i class="material-icons">remove</i>
        </button>
        <button on:click={() => camera.animatedReset({ duration: 600 })}>
            <i class="material-icons">replay</i>
        </button>
        <button on:click={changeFullScreen}>
            <i class="material-icons"
                >{fullScreen ? "fullscreen_exit" : "fullscreen"}</i
            >
        </button>
    </div>
    <div id="controls">
        <NodeSelection
            on:NodeSelected={(ev) => {
                setHoveredNode(ev.detail);
                dispatcher(
                    "NodeSelected",
                    graph.getNodeAttribute(ev.detail, "label")
                );
            }}
            {nodes}
            searchOne={true}
            label="Search a node"
        />
        <Select
            bind:value={selectedLayout}
            label="Select layout"
            on:SMUISelect:change={changeLayout}
        >
            {#each layouts as layout}
                <Option value={layout}>
                    {layout}
                </Option>
            {/each}
        </Select>
        <FormField style="margin-right: 15px">
            <Switch
                bind:checked={showLabels}
                on:SMUISwitch:change={changeLabels}
            />
            <span slot="label" style="color: white">Labels</span>
        </FormField>
    </div>
</div>

<!-- TODO: Drag and drop node, filter edges, change pointer -->
<style>
    #zoom {
        position: absolute;
        left: 0;
        bottom: 0;
        background-color: rgba(255, 62, 0, 0.2);
        padding: 2px 5px 0 0;
        border-radius: 0 10px 0 0;
    }

    #controls {
        position: absolute;
        right: 0;
        top: 0;
        background-color: rgba(255, 62, 0, 0.2);
        padding: 0 0 10px 10px;
        border-radius: 0 0 0 10px;
    }

    button {
        background-color: transparent;
        background-repeat: no-repeat;
        border: none;
        outline: none;
        margin: 0 0 0 0;
        color: rgba(255, 255, 255, 0.6);
    }

    .sigmaElement {
        position: absolute;
        top: 5px;
        bottom: 5px;
        left: 5px;
        right: 5px;
        overflow: hidden;
        background-color: rgb(33, 33, 37);
    }

    button:hover {
        color: white;
    }

    button:active {
        background-color: transparent;
    }
</style>
