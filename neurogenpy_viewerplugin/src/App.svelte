<svelte:window on:message={handleMessage}/>

{#if !destroyFlag}
    <Card>
        <Content>
            <h3>ROI Selection</h3>
            <RoiSelection
                    on:RoiSelected={ev => handleUpdateParam({ roi: ev.detail })}
                    label="Select Region of interest"
                    postMessage={postMessage}/>
        </Content>
    </Card>

    <div class="spacer"></div>

    <Card>
        <Content>
            <h3>Genes selection</h3>
            <GeneSelection on:GeneSelected={ev => handleUpdateParam({ genes: ev.detail })} genes={genes}/>
            <StructureLearning
                    on:AlgorithmSelected={ev => handleUpdateParam({ algorithm: ev.detail })}
                    label="Select structure learning algorithm"
                    postMessage={postMessage}/>
            <ParametersEstimation
                    on:EstimationSelected={ev => handleUpdateParam({ estimation: ev.detail })}
                    label="Select parameter estimation method"
                    postMessage={postMessage}/>
        </Content>
    </Card>

    <div class="spacer"></div>

    <Card>
        <Content>
            <Button on:click={learnGRN} disabled={runningFlag}>
                <Label>
                    Learn Gene regulatory network
                </Label>
            </Button>
            {#if runningFlag}
                <CircularProgress style="width:1rem;height:1rem;" indeterminate/>
            {/if}

            {#if downloadUrl}
                <Button href={downloadUrl} download="result.json">
                    <Icon class="material-icons">file_download</Icon>
                    <Label>
                        Save
                    </Label>
                </Button>
            {/if}

            <!--{#if result && hasDataSrcFlag}-->
            <!--    <ShowGraph {result}/>-->
            <!--{/if}-->

            {#if errorText}
                {errorText}
            {/if}
        </Content>
    </Card>
{/if}

<script>
    import RoiSelection from "./RoiSelection.svelte"
    import StructureLearning from "./StructureLearning.svelte"
    import GeneSelection from "./GeneSelection.svelte"
    import {hasDataSrc, getGeneNames, parcellationId, NEUROGENPY_ENDPOINT } from "./store.js"
    import Card, {Content} from "@smui/card"
    import Button, {Label, Icon} from "@smui/button"
    import CircularProgress from "@smui/circular-progress"
    import ShowGraph from "./ShowGraph.svelte"
    import {onDestroy, tick} from "svelte"
    import ParametersEstimation from "./ParametersEstimation.svelte";


    const getUuid = () => crypto.getRandomValues(new Uint32Array(1))[0].toString(16)

    let destroyFlag = false
    let destroyCbObj = []

    let hasDataSrcFlag = false
    let src = undefined
    let runningFlag = false
    let srcOrigin = undefined
    let errorText = undefined
    let downloadUrl = undefined
    let genes = []
    let result = undefined

    let param = {
        parcellation_id: parcellationId
    }

    hasDataSrc.subscribe(flag => hasDataSrcFlag = flag)

    function handleUpdateParam(newParam) {
        param = {
            ...param,
            ...newParam,
        }
    }

    getGeneNames().then(({genes: _genes}) => {
        genes = _genes
    })

    const requestMap = new Map()

    function handleMessage(msg) {
        const {source, data, origin} = msg
        const {id, method, params, result, error} = data
        if (!!result || !!error) {
            if (!id) {
                throw new Error(`expecting result/error to have id`)
            }
            if (!requestMap.has(id)) {
                throw new Error(`expecting requestMap to have id ${id}, but it does not.`)
            }

            const {rs, rj} = requestMap.get(id)
            requestMap.delete(id)

            if (result) return rs(result)
            if (error) return rj(error)
        }
        if (!/^sxplr/.test(method)) return
        switch (method) {
            case 'sxplr.init': {
                hasDataSrc.update(() => true)
                source.postMessage({
                    id,
                    jsonrpc: '2.0',
                    result: {
                        name: 'neurogenpy'
                    }
                }, origin)
                src = source
                srcOrigin = origin
                break
            }
        }
    }

    onDestroy(async () => {
        destroyFlag = true
        await tick()
        src.postMessage({
            method: `sxplr.exit`,
            params: {
                requests: destroyCbObj
            },
            jsonrpc: '2.0',
            id: getUuid()
        }, srcOrigin)
    })

    async function postMessage(_msg) {
        if (destroyFlag) {
            destroyCbObj = [...destroyCbObj, {..._msg, id: getUuid()}]
            return
        }
        const id = getUuid()
        const {abortSignal, ...msg} = _msg
        if (abortSignal) {
            abortSignal.onAbort(() => {
                src.postMessage({
                    method: `sxplr.cancelRequest`,
                    jsonrpc: '2.0',
                    params: {id},
                }, srcOrigin)
            })
        }
        src.postMessage({
            ...msg,
            id,
            jsonrpc: '2.0',
        }, srcOrigin)
        return new Promise((rs, rj) => {
            requestMap.set(id, {rs, rj})
        })
    }

    async function learnGRN() {
        if (runningFlag) {
            console.warn(`GRN learning already running, do not start a new one.`)
            return
        }
        runningFlag = true
        result = null
        errorText = null
        if (downloadUrl) {
            URL.revokeObjectURL(downloadUrl)
            downloadUrl = null
        }

        try {
            const res = await fetch(`${NEUROGENPY_ENDPOINT}/grn/grn`, {
                method: 'POST',
                body: JSON.stringify(param),
                headers: {
                    'content-type': 'application/json'
                }
            })
            if (res.status >= 400) {
                throw new Error(res.statusText)
            }
            const {poll_url} = await res.json()

            result = await new Promise((rs, rj) => {
                const intervalRef = setInterval(async () => {
                    const res = await fetch(`${NEUROGENPY_ENDPOINT}/grn/grn/${poll_url}`)
                    if (res.status >= 400) {
                        return rj(res.statusText)
                    }
                    const {status, result} = await res.json()
                    if (status === "SUCCESS") {
                        console.log('SUCCESS', result)
                        clearInterval(intervalRef)
                        rs(result)
                    }
                    if (status === "FAILURE") {
                        console.log('FAILURE')
                        clearInterval(intervalRef)
                        rj("operation failed")
                    }
                }, 1000)
            })

            const blob = new Blob([JSON.stringify(result, null, 2)], {type: 'text/plain'})
            downloadUrl = URL.createObjectURL(blob)
        } catch (e) {
            errorText = e.toString()
        } finally {
            runningFlag = false
        }
    }

</script>

<style>
    div.spacer {
        height: 1rem;
    }
</style>