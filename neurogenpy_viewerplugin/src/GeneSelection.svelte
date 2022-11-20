<script>
  import Autocomplete from "@smui-extra/autocomplete";
  import Textfield from "@smui/textfield";
  import Chip, { Set as ChipsSet, Text, TrailingAction } from "@smui/chips";
  import { createEventDispatcher } from "svelte";
  import Button, { Icon } from "@smui/button";

  export let genes = [];
  let selectedGenes = [];

  const dispatcher = createEventDispatcher();

  let autocmplText = "";
  let currentAutocompleteList = [];
  let browseInput;

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

  async function readFile(ev) {
    const [file] = ev.target.files;
    if (!file) return;
    const data = await file.text();

    const genesJSON = JSON.parse(data);
    selectedGenes = genesJSON["genes"];
    dispatcher("GeneSelected", selectedGenes);
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
  <input
    bind:this={browseInput}
    type="file"
    id="file"
    class="hidden"
    style="display: none;"
    on:change={(ev) => readFile(ev)}
  />
  <Button on:click={() => browseInput.click()} class="button">
    <Icon class="material-icons">file_upload</Icon>
  </Button>
</div>
<div>
  <ChipsSet chips={selectedGenes} let:chip selectedGenes>
    <Chip {chip}>
      <Text>{chip}</Text>
      <TrailingAction icon$class="material-icons">cancel</TrailingAction>
    </Chip>
  </ChipsSet>
</div>
