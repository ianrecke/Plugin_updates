<script>
  import Autocomplete from "@smui-extra/autocomplete";
  import Textfield from "@smui/textfield";
  import Chip, { Set as ChipsSet, Text, TrailingAction } from "@smui/chips";
  import { createEventDispatcher } from "svelte";

  export let genes = [];
  let selectedGenes = [];

  const dispatcher = createEventDispatcher();

  let autocmplText = "";
  let currentAutocompleteList = [];

  $: {
    if (autocmplText === "") {
      currentAutocompleteList = [];
    } else {
      let regex;
      try {
        regex = new RegExp(autocmplText, "i");
      } catch (e) {
        // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions#escaping
        // CC0 or MIT
        regex = new RegExp(autocmplText.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      }
      currentAutocompleteList = genes
        .filter((gene) => regex.test(gene))
        .slice(0, 5);
    }
  }

  function geneSelected(gene) {
    selectedGenes = Array.from(new Set([...selectedGenes, gene]));
    dispatcher("GeneSelected", selectedGenes);
    autocmplText = "";
  }

  function handleKeydown(ev) {
    if (ev.code === "Enter") {
      if (currentAutocompleteList.length > 0) {
        geneSelected(currentAutocompleteList[0]);
      }
    }
  }

  async function searchfn(_input) {
    return currentAutocompleteList;
  }
</script>

<div>
  <Autocomplete
    showMenuWithNoInput={false}
    bind:value={autocmplText}
    search={searchfn}
    on:SMUIAutocomplete:selected={(ev) => geneSelected(ev.detail)}
    on:keydown={handleKeydown}
  >
    <Textfield label="Genes" bind:value={autocmplText} />
  </Autocomplete>
</div>
<div>
  <ChipsSet chips={selectedGenes} let:chip selectedGenes>
    <Chip {chip}>
      <Text>{chip}</Text>
      <TrailingAction icon$class="material-icons">cancel</TrailingAction>
    </Chip>
  </ChipsSet>
</div>
