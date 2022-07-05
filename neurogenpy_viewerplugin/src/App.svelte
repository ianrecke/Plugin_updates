<script>
    import RoiSelection from "./RoiSelection.svelte";
    import StructureLearning from "./StructureLearning.svelte";
    import GeneSelection from "./GeneSelection.svelte";
    import { getGeneNames, hasDataSrc, parcellationId } from "./store.js";
    import Card, { Content } from "@smui/card";
    import Button, { Label } from "@smui/button";
    import CircularProgress from "@smui/circular-progress";
    import Description from "./Description.svelte";
    import { onDestroy, tick } from "svelte";
    import ParametersEstimation from "./ParametersEstimation.svelte";
    import ShowGraph from "./ShowGraph.svelte";
    import { getFromNeurogenpy } from "./request";
    import Select, { Option } from "@smui/select";

    const getUuid = () =>
        crypto.getRandomValues(new Uint32Array(1))[0].toString(16);

    let destroyFlag = false;
    let destroyCbObj = [];

    let src = undefined;
    let runningFlag = false;
    let srcOrigin = undefined;
    let errorText = undefined;
    let genes = [];
    let result = {};
    let dataType = "continuous";

    let param = {
        parcellation_id: parcellationId,
    };
    let sg = undefined;

    function handleUpdateParam(newParam) {
        param = {
            ...param,
            ...newParam,
        };
    }

    getGeneNames().then(({ genes: _genes }) => {
        genes = _genes;
    });

    const requestMap = new Map();

    const styles = Array.from(
        window.document.querySelectorAll('link[rel="stylesheet"], style')
    );

    function handleMessage(msg) {
        const { source, data, origin } = msg;
        const { id, method, result, error } = data;
        if (!!result || !!error) {
            if (!id) {
                throw new Error(`expecting result/error to have id`);
            }
            if (!requestMap.has(id)) {
                throw new Error(
                    `expecting requestMap to have id ${id}, but it does not.`
                );
            }

            const { rs, rj } = requestMap.get(id);
            requestMap.delete(id);

            if (result) return rs(result);
            if (error) return rj(error);
        }
        if (!/^sxplr/.test(method)) return;
        switch (method) {
            case "sxplr.init":
                {
                    hasDataSrc.update(() => true);
                    source.postMessage(
                        {
                            id,
                            jsonrpc: "2.0",
                            result: {
                                name: "NeurogenPy - Learn GRNs",
                            },
                        },
                        origin
                    );
                    src = source;
                    srcOrigin = origin;
                    break;
                }
                break;
        }
    }

    onDestroy(async () => {
        destroyFlag = true;
        await tick();
        src.postMessage(
            {
                method: `sxplr.exit`,
                params: {
                    requests: destroyCbObj,
                },
                jsonrpc: "2.0",
                id: getUuid(),
            },
            srcOrigin
        );
    });

    async function postMessage(_msg) {
        if (destroyFlag) {
            destroyCbObj = [...destroyCbObj, { ..._msg, id: getUuid() }];
            return;
        }
        const id = getUuid();
        const { abortSignal, ...msg } = _msg;
        if (abortSignal) {
            abortSignal.onAbort(() => {
                src.postMessage(
                    {
                        method: `sxplr.cancelRequest`,
                        jsonrpc: "2.0",
                        params: { id },
                    },
                    srcOrigin
                );
            });
        }
        src.postMessage(
            {
                ...msg,
                id,
                jsonrpc: "2.0",
            },
            srcOrigin
        );
        return new Promise((rs, rj) => {
            requestMap.set(id, { rs, rj });
        });
    }

    async function showGraph() {
        let win = window.open(
            "",
            "NeurogenPy - Learn GRNs",
            "toolbar=no,location=no,directories=no,status=no,menubar=no," +
                "scrollbars=yes,resizable=yes,width=" +
                (9 * screen.width) / 10 +
                ",height=" +
                (9 * screen.height) / 10
        );

        // FIXME: Multiple ShowGraphs retrieved if the button is clicked multiple times
        // if (!sg) {
        styles.map((st) => {
            let styleElement;
            styleElement = win.document.createElement("link");
            styleElement.rel = "stylesheet";
            styleElement.href = st.href;
            win.document.head.appendChild(styleElement);
        });

        const sgPromise = new Promise((resolve, reject) => {
            setTimeout(() => {
                let promise = new ShowGraph({
                    target: win.document.body,
                    props: { result: result, dataType: dataType },
                });
                resolve(promise);
            }, 500);
        });

        sg = await sgPromise;
        // }

        win.focus();
    }

    async function learnGRN() {
        handleUpdateParam({ data_type: dataType });
        if (runningFlag) {
            console.warn(
                `GRN learning already running, do not start a new one.`
            );
            return;
        }
        runningFlag = true;
        result = null;
        errorText = null;
        try {
            result = await getFromNeurogenpy("/grn/grn", JSON.stringify(param));
        } catch (e) {
            errorText = e.toString();
        } finally {
            runningFlag = false;
        }
    }
</script>

<svelte:window height="100%" width="100%" on:message={handleMessage}/>

{#if !destroyFlag}
    <Description />
    <Card>
        <Content>
            <h4>Select options</h4>
            <RoiSelection
                on:RoiSelected={(ev) => handleUpdateParam({ roi: ev.detail })}
                label="Region of interest"
                {postMessage}
            />
            <GeneSelection
                on:GeneSelected={(ev) =>
                    handleUpdateParam({ genes: ev.detail })}
                {genes}
            />
            <Select
                bind:value={dataType}
                label="Data type"
            >
                <Option value={"continuous"}>
                    {"Continuous"}
                </Option>
                <Option value={"discrete"}>
                    {"Discrete"}
                </Option>
            </Select>
            <StructureLearning
                {dataType}
                on:AlgorithmSelected={(ev) =>
                    handleUpdateParam({ algorithm: ev.detail })}
            />
            <ParametersEstimation
                {dataType}
                on:EstimationSelected={(ev) =>
                    handleUpdateParam({ estimation: ev.detail })}
            />
        </Content>
    </Card>

    <div class="spacer" />

    <Card>
        <Content>
            <Button
                on:click={async function () {
                    await learnGRN();
                    if (result) {
                        showGraph();
                    }
                }}
                disabled={runningFlag}
            >
                <Label>Learn GRN</Label>
            </Button>
            {#if runningFlag}
                <CircularProgress
                    style="width:1rem;height:1rem;"
                    indeterminate
                />
            {/if}

            {#if errorText}
                {errorText}
            {/if}
        </Content>
    </Card>
{/if}

<style>
    div.spacer {
        height: 1rem;
    }
</style>
