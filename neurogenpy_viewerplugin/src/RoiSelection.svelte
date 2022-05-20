<!-- <svelte:options accessors={true} /> -->

<div class="roi-selection-row">
  <Autocomplete
    showMenuWithNoInput={false}
    search={searchfn}
    bind:value={selectedRegionName}
    bind:text={autocmplText}
    on:keydown={handleKeydown}
    on:SMUIAutocomplete:selected={ev => regionSelected(ev.detail)}>
    <Textfield label={label} bind:value={autocmplText} />
  </Autocomplete>


  {#if hasDataSrcFlag}
  <Button color={scanning ? "primary" : "secondary"}
    on:click={ev => handleSensor(ev)}>
    <Label>
      <span class="material-icons">sensors</span>
    </Label>
  </Button>
  {/if}
</div>

{#if !!selectedRegionName}
<ChipsSet chips={[selectedRegionName]} let:chip>
  <Chip chip={selectedRegionName}>
    <Text>{selectedRegionName}</Text>
  </Chip>
</ChipsSet>
{/if}

<style>
.roi-selection-row
{
  display: flex;
  align-items: baseline;
}
</style>

<script>
  export let label = "Select ROI"
  export let postMessage = async (...arg) => { throw new Error(`expecting parent component overwrite postMessage function`) }
  export let selectedRegionName = undefined

  import Autocomplete from '@smui-extra/autocomplete';
  import Button, { Label } from "@smui/button"
  import Textfield from '@smui/textfield';
  import { hasDataSrc, searchRegion } from "./store.js"
  import { createEventDispatcher } from "svelte"
  import Chip, { Set as ChipsSet, Text, TrailingAction } from "@smui/chips"

  let hasDataSrcFlag = false
  let searchId = 1
  let currentAutocompleteList = []
  let scanning = false
  let abortSignal = undefined
  let autocmplText = ''
  const dispatcher = createEventDispatcher()

  hasDataSrc.subscribe(flag => hasDataSrcFlag = flag)

  class AbortSignal {
    constructor(){
      this.cb = []
    }
    abort(){
      while (this.cb.length > 0) this.cb.pop()()
    }
    onAbort(cb){
      this.cb.push(cb)
    }
  }

  function regionSelected(regionName) {
    selectedRegionName = regionName
    dispatcher("RoiSelected", selectedRegionName)
    currentAutocompleteList = []
  }
  function handleKeydown(ev){
    if (ev.code === "Enter") {
      if (currentAutocompleteList.length > 0) {
        regionSelected(currentAutocompleteList[0])
      }
    }
  }


  async function searchfn(input){
    if (input === '' || !input) {
      return []
    }
    searchId += 1
    const thisSearchId = searchId
    const returnArr = await searchRegion(input)
    if (thisSearchId !== searchId) return []
    currentAutocompleteList = returnArr.map(r => r.name)
    return currentAutocompleteList
  }

  const handleSensor = async () => {
    if (scanning) {
      if (abortSignal) abortSignal.abort()
      return
    }

    scanning = true
    try {
      abortSignal = new AbortSignal()
      const result = await postMessage({
        method: `sxplr.getUserToSelectARoi`,
        params: {
          type: 'region',
          message: `siibra-jugex: Please select a region for ${label}`
        },
        abortSignal
      })
      if (scanning) {
        regionSelected(result && result.name)
      }
    } catch (e) {

    } finally {
      abortSignal = null
      scanning = false
    }
  }


</script>