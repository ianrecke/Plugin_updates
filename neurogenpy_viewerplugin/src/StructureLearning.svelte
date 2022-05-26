<div>
    <Autocomplete
            combobox options={Object.keys(algs)}
            bind:value={selected}
            bind:text={autocmplText}
            on:SMUIAutocomplete:selected={ev => algorithmSelected(algs[ev.detail])}
    >
        <Textfield label="Select structure learning algorithm" inputStyle={{fontSize: 5}}
                   bind:value={autocmplText}/>
    </Autocomplete>
</div>


<script>
    import Autocomplete from "@smui-extra/autocomplete";
    import {createEventDispatcher} from "svelte"
    import {algorithms} from "./algorithms.json"
    import Textfield from "@smui/textfield";

    const algs = Object.assign({}, ...algorithms.map((x) => ({[x.name]: x.id})));
    let autocmplText = ""
    let selected = undefined
    const dispatcher = createEventDispatcher()


    function algorithmSelected(alg) {
        dispatcher('AlgorithmSelected', alg)
    }
</script>
