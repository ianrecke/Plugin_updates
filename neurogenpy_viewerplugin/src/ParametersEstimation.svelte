<script>
    import Autocomplete from "@smui-extra/autocomplete";
    import { createEventDispatcher } from "svelte";
    import { estimation } from "./estimation.json";
    import Textfield from "@smui/textfield";

    export let dataType;
    let autocmplText = "";
    const methods = Object.keys(estimation);
    let counter = 0;

    const dispatcher = createEventDispatcher();

    function estimationSelected(est) {
        dispatcher("EstimationSelected", est);
    }

    async function search(input) {
        const myCounter = ++counter;

        await new Promise((resolve) => setTimeout(resolve, 100));

        if (myCounter !== counter) return false;

        return methods.filter(
            (item) =>
                (estimation[item].available === dataType ||
                    estimation[item].available === "both") &&
                item.toLowerCase().includes(input.toLowerCase())
        );
    }
</script>

<div>
    <Autocomplete
        combobox
        {search}
        bind:text={autocmplText}
        on:SMUIAutocomplete:selected={(ev) =>
            estimationSelected(estimation[ev.detail].id)}
    >
        <Textfield
            label="Parameter estimation"
            inputStyle={{ fontSize: 9 }}
            bind:value={autocmplText}
        />
    </Autocomplete>
</div>
