<script>
    import Card, { Content } from "@smui/card";
    import DSeparation from "./DSeparation.svelte";
    import Button, { Icon, Label } from "@smui/button";
    import Select, { Option } from "@smui/select";
    import { createEventDispatcher } from "svelte";

    export let nodes = [];
    const dispatcher = createEventDispatcher();

    export let fileTypes;
    export let json_bn;
    let fileType = fileTypes[0];
</script>

<Card style="padding: 10px; height: 90%">
    <Content style="height:100%">
        <DSeparation {nodes} {json_bn} />

        <div style="height: 10%">
            <Button on:click={() => dispatcher("CommunitiesSelected", true)}>
                <Label>Communities Louvain</Label>
            </Button>
        </div>

        <div style="height: 20%">
            <Select bind:value={fileType} label="Select file format" style="">
                {#each fileTypes as ft}
                    <Option value={ft}>
                        {ft}
                    </Option>
                {/each}
            </Select>
            <Button on:click={() => dispatcher("SaveFile", fileType)}>
                <Icon class="material-icons">file_download</Icon>
                <Label>Download</Label>
            </Button>
        </div>
    </Content>
</Card>
