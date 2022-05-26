<div>
    <Autocomplete
            combobox options={Object.keys(methods)}
            bind:value={selected}
            bind:text={autocmplText}
            on:SMUIAutocomplete:selected={ev => estimationSelected(methods[ev.detail])}
    >
        <Textfield label="Select parameter estimation method" inputStyle={{fontSize: 9}}
                   bind:value={autocmplText}/>
    </Autocomplete>
</div>


<script>
    import Autocomplete from "@smui-extra/autocomplete";
    import {createEventDispatcher} from "svelte"
    import {estimation} from "./estimation.json"
    import Textfield from "@smui/textfield";


    let autocmplText = ""
    let selected = undefined
    const methods = Object.assign({}, ...estimation.map((x) => ({[x.name]: x.id})));
    const dispatcher = createEventDispatcher()

    function estimationSelected(est) {
        dispatcher('EstimationSelected', est)
    }
</script>
