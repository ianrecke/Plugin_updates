<script>
    import Autocomplete from "@smui-extra/autocomplete";
    import { createEventDispatcher } from "svelte";
    import { algorithms } from "./algorithms.json";
    import Textfield from "@smui/textfield";

    export let dataType;

    let algsNames = Object.keys(algorithms);

    let autocmplText = "";
    let counter = 0;

    const dispatcher = createEventDispatcher();

    async function search(input) {
        const myCounter = ++counter;

        // Pretend to be loading something...
        await new Promise((resolve) => setTimeout(resolve, 1000));

        if (myCounter !== counter) return false;

        return algsNames.filter(
            (item) =>
                (algorithms[item].available === dataType ||
                    algorithms[item].available === "both") &&
                item.toLowerCase().includes(input.toLowerCase())
        );
    }

    function algorithmSelected(alg) {
        dispatcher("AlgorithmSelected", alg);
    }
</script>

<div>
    <Autocomplete
        combobox
        {search}
        bind:text={autocmplText}
        on:SMUIAutocomplete:selected={(ev) =>
            algorithmSelected(algorithms[ev.detail].id)}
        style="overflow: visible"
    >
        <Textfield
            label="Structure learning"
            inputStyle={{ fontSize: 5 }}
            bind:value={autocmplText}
        />
    </Autocomplete>
</div>
