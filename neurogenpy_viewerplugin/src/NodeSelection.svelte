<script>
    import Autocomplete from "@smui-extra/autocomplete";
    import Textfield from "@smui/textfield";
    import Chip, { Set as ChipsSet, Text, TrailingAction } from "@smui/chips";
    import { createEventDispatcher } from "svelte";
    import Icon from "@smui/textfield/icon";

    export let selectedNodes = [];
    let autocmplText = "";
    let currentAutocompleteList = [];
    export let searchOne = false;

    export let nodes = [];
    export let label = "";

    const dispatcher = createEventDispatcher();

    $: {
        if (autocmplText === "") {
            currentAutocompleteList = [];
        } else {
            let regex;
            try {
                regex = new RegExp(autocmplText, "i");
            } catch (e) {
                regex = new RegExp(
                    autocmplText.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
                );
            }
            currentAutocompleteList = nodes
                .filter((node) => regex.test(node))
                .slice(0, 5);
        }
    }

    async function searchfn(_input) {
        return currentAutocompleteList;
    }

    function handleKeydown(ev) {
        if (ev.code === "Enter") {
            if (currentAutocompleteList.length > 0) {
                nodeSearched(currentAutocompleteList[0]);
            }
        }
    }

    function nodeSearched(node) {
        if (searchOne) {
            selectedNodes = node;
        } else {
            selectedNodes = Array.from(new Set([...selectedNodes, node]));
        }
        dispatcher("NodeSelected", selectedNodes);
        autocmplText = "";
    }
</script>

<Autocomplete
    class="nodeSelection"
    showMenuWithNoInput={false}
    search={searchfn}
    bind:value={autocmplText}
    on:keydown={handleKeydown}
    on:SMUIAutocomplete:selected={(ev) => nodeSearched(ev.detail)}
>
    <Textfield {label} bind:value={autocmplText}>
        <Icon class="material-icons" slot="leadingIcon">search</Icon>
    </Textfield>
</Autocomplete>
{#if !searchOne}
    <div>
        <ChipsSet chips={selectedNodes} let:chip selectedNodes>
            <Chip {chip}>
                <Text>{chip}</Text>
                <TrailingAction icon$class="material-icons"
                    >cancel</TrailingAction
                >
            </Chip>
        </ChipsSet>
    </div>
{/if}
