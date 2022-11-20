<script>
    import { createEventDispatcher, onMount } from "svelte";
    import sigma from "sigma";
    import { drawHover } from "./hover";
    import forceAtlas2 from "graphology-layout-forceatlas2";
    import FA2Layout from "graphology-layout-forceatlas2/worker";
    import { circular } from "graphology-layout";
    import { animateNodes } from "sigma/utils/animate";
    import Graph from "graphology";
    import { parse, write } from "graphology-gexf/browser";
    import Select, { Option } from "@smui/select";
    import FormField from "@smui/form-field";
    import Switch from "@smui/switch";
    import louvain from "graphology-communities-louvain";
    import { watchResize } from "svelte-watch-resize";
    import NodeSelection from "./NodeSelection.svelte";
    import { saveAsPNG } from "./saveFile";
    import { generateRandomColor } from "./color.ts";

    export let gexf_graph = undefined;
    let graph = parse(Graph, gexf_graph);

    const dispatcher = createEventDispatcher();

    let container = undefined;
    let camera = undefined;

    let showLabels = true;
    let renderer = undefined;

    const layouts = [
        { text: "ForceAtlas2", id: "ForceAtlas2" },
        { text: "Circular", id: "Circular" },
        { text: "Dot", id: "Dot" },
        { text: "Grid", id: "Grid" },
        { text: "FruchtermanReingold", id: "fruchterman_reingold" },
        { text: "Sugiyama", id: "Sugiyama" },
    ];

    let selectedLayout = undefined;
    let fa2Layout = undefined;
    let layoutPositions = Object.assign(
        {},
        ...layouts.map((l) => ({ [l["id"]]: undefined }))
    );

    let highlightedNeighbors = new Set();
    let hoveredNode = undefined;
    let highlightedNodes = undefined;
    let cancelCurrentAnimation = null;

    let fullScreen = false;
    let display;

    let draggedNode = undefined;
    let isDragging = false;

    const style = getComputedStyle(document.body);
    const containerColor = style.getPropertyValue("--surface");
    const onContainerColor = style.getPropertyValue("color");

    export let nodes = [];
    onMount(() => createGraph());

    function createGraph() {
        renderer = new sigma(graph, container, {
            minCameraRatio: 0.1,
            maxCameraRatio: 10,
            hoverRenderer: drawHover,
            defaultEdgeType: "arrow",
            allowInvalidContainer: true,
            enableEdgeWheelEvents: true,
            defaultNodeColor: onContainerColor,
            defaultEdgeColor: onContainerColor,
            labelColor: { color: onContainerColor },
            nodeReducer: nodeReducerFunction,
            edgeReducer: edgeReducerFunction,
        });

        renderer.on("clickNode", (e) => {
            if (!isDragging) nodeClicked(e);
        });
        renderer.on("enterNode", ({ node }) => {
            setHighlightedNodes([node], true);
        });
        renderer.on("leaveNode", () => {
            setHighlightedNodes(undefined, true);
        });

        renderer.on("downNode", (e) => {
            isDragging = true;
            draggedNode = e.node;
            graph.setNodeAttribute(draggedNode, "highlighted", true);
        });

        renderer.getMouseCaptor().on("mousemovebody", (e) => {
            if (!isDragging || !draggedNode) return;

            if (fa2Layout.isRunning()) stopFA2();

            const pos = renderer.viewportToGraph(e);

            graph.setNodeAttribute(draggedNode, "x", pos.x);
            graph.setNodeAttribute(draggedNode, "y", pos.y);

            e.preventSigmaDefault();
            e.original.preventDefault();
            e.original.stopPropagation;
        });

        renderer.getMouseCaptor().on("mouseup", () => {
            if (draggedNode)
                graph.removeNodeAttribute(draggedNode, "highlighted");

            isDragging = false;
            draggedNode = undefined;
        });

        camera = renderer.getCamera();

        const sensibleSettings = forceAtlas2.inferSettings(graph);
        fa2Layout = new FA2Layout(graph, {
            settings: sensibleSettings,
        });

        // Sigma assigns some events to the parent window, so we have to reassign them.
        // Events forwarding should be the best solution. Svelte Material UI also presents
        // this issue.
        let mouseCaptor = renderer.getMouseCaptor();
        document.removeEventListener("mousemove", mouseCaptor.handleMove);
        document.removeEventListener("mouseup", mouseCaptor.handleUp);

        display.ownerDocument.addEventListener(
            "mousemove",
            mouseCaptor.handleMove,
            false
        );
        display.ownerDocument.addEventListener(
            "mouseup",
            mouseCaptor.handleUp,
            false
        );
    }

    function nodeClicked(e) {
        dispatcher("NodeSelected", graph.getNodeAttribute(e.node, "label"));
    }

    function changeLabels() {
        renderer.setSetting("renderLabels", showLabels);
        renderer.refresh();
    }

    export function setHighlightedNodes(nodes, neigh) {
        if (nodes) {
            highlightedNodes = new Set(nodes);
            if (neigh) {
                hoveredNode = nodes[0];
                highlightedNeighbors = new Set(graph.neighbors(hoveredNode));
            } else {
                highlightedNeighbors = new Set();
            }
        } else {
            highlightedNodes = undefined;
            hoveredNode = undefined;
            highlightedNeighbors = new Set();
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
        if (layoutPositions[selectedLayout.id]) {
            animateLayout(layoutPositions[selectedLayout.id]);
        } else {
            switch (selectedLayout.text) {
                case "Circular":
                    layoutPositions["Circular"] = circular(graph, {
                        scale: 100,
                    });
                    animateLayout(layoutPositions["Circular"]);
                    break;
                case "ForceAtlas2":
                    toggleFA2Layout();
                    break;
                default:
                    dispatcher("NeurogenpyLayout", selectedLayout.id);
                    break;
            }
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

    function animateLayout(positions) {
        if (fa2Layout.isRunning()) stopFA2();
        if (cancelCurrentAnimation) cancelCurrentAnimation();

        cancelCurrentAnimation = animateNodes(graph, positions, {
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

            graph.forEachNode((node) =>
                graph.setNodeAttribute(node, "color", colors[communities[node]])
            );

            graph.forEachEdge((edge, attributes, source, target) => {
                const sourceCom = communities[source];
                if (sourceCom === communities[target]) {
                    graph.setEdgeAttribute(edge, "color", colors[sourceCom]);
                } else {
                    graph.setEdgeAttribute(edge, "color", onContainerColor);
                }
            });
            renderer.refresh();
        }
    }

    export function savePNG() {
        saveAsPNG(renderer);
    }

    export function getGEXF() {
        return write(graph);
    }

    export function setLayout(layoutName, lp) {
        layoutPositions[layoutName] = lp;
        animateLayout(layoutPositions[layoutName]);
    }

    function nodeReducerFunction(node, data) {
        const res = { ...data };

        if (
            !isDragging &&
            highlightedNodes &&
            !highlightedNodes.has(node) &&
            !highlightedNeighbors.has(node)
        ) {
            res.label = "";
            res.color = containerColor;
        }
        res.highlighted = highlightedNodes && highlightedNodes.has(node);

        return res;
    }

    function edgeReducerFunction(edge, data) {
        const res = { ...data };

        if (!isDragging && highlightedNodes) {
            if (hoveredNode) {
                if (!graph.hasExtremity(edge, hoveredNode)) {
                    res.hidden = true;
                }
            } else {
                if (
                    !highlightedNodes.has(graph.source(edge)) ||
                    !highlightedNodes.has(graph.target(edge))
                ) {
                    res.hidden = true;
                }
            }
        }

        return res;
    }
</script>

<div bind:this={display}>
    <div
        bind:this={container}
        class="surfaceDiv"
        id="sigmaElement"
        use:watchResize={() => renderer.refresh()}
    />

    <div id="zoom" class="primaryDiv">
        <button
            class="lightButton"
            on:click={() => camera.animatedZoom({ duration: 600 })}
        >
            <i class="material-icons">add</i>
        </button>
        <button
            class="lightButton"
            on:click={() => camera.animatedUnzoom({ duration: 600 })}
        >
            <i class="material-icons">remove</i>
        </button>
        <button
            class="lightButton"
            on:click={() => camera.animatedReset({ duration: 600 })}
        >
            <i class="material-icons">replay</i>
        </button>
        <button class="lightButton" on:click={changeFullScreen}>
            <i class="material-icons"
                >{fullScreen ? "fullscreen_exit" : "fullscreen"}</i
            >
        </button>
    </div>
    <div id="controls1" class="controls primaryDiv">
        <NodeSelection
            on:NodeSelected={(ev) => {
                setHighlightedNodes([ev.detail], true);
                dispatcher(
                    "NodeSelected",
                    graph.getNodeAttribute(ev.detail, "label")
                );
            }}
            {nodes}
            searchOne={true}
            label="Search a gene"
        />
    </div>
    <div id="controls2" class="controls primaryDiv">
        <Select
            bind:value={selectedLayout}
            label="Select layout"
            on:SMUISelect:change={changeLayout}
        >
            {#each layouts as layout}
                <Option value={layout}>
                    {layout.text}
                </Option>
            {/each}
        </Select>
        <FormField style="margin-right: 15px">
            <Switch
                bind:checked={showLabels}
                on:SMUISwitch:change={changeLabels}
            />
            <span slot="label">Labels</span>
        </FormField>
    </div>
</div>

<!-- TODO: filter edges, change pointer -->
<style>
    #zoom {
        position: absolute;
        left: 0;
        bottom: 0;
        padding: 2px 5px 0 0;
        border-radius: 0 10px 0 0;
    }

    .controls {
        position: absolute;
        top: 0;
    }

    #controls1 {
        left: 0;
        padding: 0 10px 10px 10px;
        border-radius: 0 0 10px 0;
    }

    #controls2 {
        right: 0;
        padding: 0 0 10px 10px;
        border-radius: 0 0 0 10px;
    }

    #sigmaElement {
        position: absolute;
        top: 5px;
        bottom: 5px;
        left: 5px;
        right: 5px;
        overflow: hidden;
    }

    button {
        background-repeat: no-repeat;
        border: none;
        outline: none;
        margin: 0 0 0 0;
    }
</style>
