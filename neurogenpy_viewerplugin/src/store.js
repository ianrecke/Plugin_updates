import { writable } from "svelte/store"

export const hasDataSrc = writable(false)

export const SIIBRA_API_ENDPOINT = `https://siibra-api-latest.apps-dev.hbp.eu`
export const NEUROGENPY_ENDPOINT = `http://localhost:6001`

export const getGeneNames = async () => {
    const res = await fetch(`${SIIBRA_API_ENDPOINT}/v2_0/genes`).then()
    return await res.json()
}

const atlasId = "juelich/iav/atlas/v1.0.0/1"
export const parcellationId = "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290"
export const searchRegion = async input => {
    const url = new URL(`${SIIBRA_API_ENDPOINT}/v2_0/atlases/${atlasId}/parcellations/${parcellationId}/regions`)
    url.searchParams.set('find', input)
    const res = await fetch(url)
    return await res.json()
}
