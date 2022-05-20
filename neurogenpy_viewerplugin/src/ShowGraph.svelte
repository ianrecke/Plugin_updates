<FormField>
    <Switch
            bind:checked
            on:SMUISwitch:change={handleSwitchEvent}/>
    <span slot="label">Show graph</span>
</FormField>

<div id="sigma-container"></div>
<div id="controls">
    <div class="input"><label for="zoom-in">Zoom in</label>
        <button id="zoom-in">+</button>
    </div>
    <div class="input"><label for="zoom-out">Zoom out</label>
        <button id="zoom-out">-</button>
    </div>
    <div class="input"><label for="zoom-reset">Reset zoom</label>
        <button id="zoom-reset">⊙</button>
    </div>
    <div class="input">
        <label for="labels-threshold">Labels threshold</label>
        <input id="labels-threshold" type="range" min="0" max="15" step="0.5"/>
    </div>
</div>

<style>
    body {
        font-family: sans-serif;
    }

    html,
    body,
    #sigma-container {
        width: 100%;
        height: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
    }

    #controls {
        position: absolute;
        right: 1em;
        top: 1em;
        text-align: right;
    }

    .input {
        position: relative;
        display: inline-block;
        vertical-align: middle;
    }

    .input:not(:hover) label {
        display: none;
    }

    .input label {
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: black;
        color: white;
        padding: 0.2em;
        border-radius: 2px;
        margin-top: 0.3em;
        font-size: 0.8em;
        white-space: nowrap;
    }

    .input button {
        width: 2.5em;
        height: 2.5em;
        display: inline-block;
        text-align: center;
        background: white;
        outline: none;
        border: 1px solid dimgrey;
        border-radius: 2px;
        cursor: pointer;
    }
</style>


<script>
    import Sigma from "sigma"
    import Graph from "graphology";
    import {parse} from "graphology-gexf/browser";
    import Switch from "@smui/switch"
    import FormField from '@smui/form-field'

    let checked = false
    export let result = "bn.gexf";

    // Load external GEXF file:
    fetch(result)
        .then((res) => res.text())
        .then((gexf) => {
            // Parse GEXF string:
            const graph = parse(Graph, gexf);

            // Retrieve some useful DOM elements:
            const container = document.getElementById("sigma-container");
            const zoomInBtn = document.getElementById("zoom-in");
            const zoomOutBtn = document.getElementById("zoom-out");
            const zoomResetBtn = document.getElementById("zoom-reset");
            const labelsThresholdRange = document.getElementById("labels-threshold");

            const renderer = new Sigma(graph, container, {
                minCameraRatio: 0.1,
                maxCameraRatio: 10,
            });
            const camera = renderer.getCamera();

            // Bind zoom manipulation buttons
            zoomInBtn.addEventListener("click", () => {
                camera.animatedZoom({duration: 600});
            });
            zoomOutBtn.addEventListener("click", () => {
                camera.animatedUnzoom({duration: 600});
            });
            zoomResetBtn.addEventListener("click", () => {
                camera.animatedReset({duration: 600});
            });

            // Bind labels threshold to range input
            labelsThresholdRange.addEventListener("input", () => {
                renderer.setSetting("labelRenderedSizeThreshold", +labelsThresholdRange.value);
            });

            // Set proper range initial value:
            labelsThresholdRange.value = renderer.getSetting("labelRenderedSizeThreshold") + "";
        });

    function handleSwitchEvent(event) {}

</script>