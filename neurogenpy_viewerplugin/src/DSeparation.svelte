<script>
    import Drawer, {
        AppContent,
        Content,
        Header,
        Title,
        Subtitle,
    } from "@smui/drawer";
    import Button, { Label } from "@smui/button";
    import List, { Item, Text } from "@smui/list";
    import NodeSelection from "./NodeSelection.svelte";

    export let nodes = [];
    let active = "X nodes";
    let selectedNodes = { "X nodes": [], "Y nodes": [], "Z nodes": [] };
    let showedNodes = selectedNodes[active];

    function setActive(value) {
        selectedNodes[active] = showedNodes;
        active = value;
        showedNodes = selectedNodes[active];
    }
</script>

<div class="drawer-container">
    <Drawer>
        <Header>
            <Title>D-Separation</Title>
            <Subtitle
                >Check if sets X and Y are d-separated by another set Z.</Subtitle
            >
        </Header>
        <Content>
            <List>
                <Item
                    on:click={() => setActive("X nodes")}
                    activated={active === "X nodes"}
                >
                    <Text>X nodes</Text>
                </Item>
                <Item
                    on:click={() => setActive("Y nodes")}
                    activated={active === "Y nodes"}
                >
                    <Text>Y nodes</Text>
                </Item>
                <Item
                    on:click={() => setActive("Z nodes")}
                    activated={active === "Z nodes"}
                >
                    <Text>Z nodes</Text>
                </Item>
            </List>
        </Content>
    </Drawer>

    <AppContent class="app-content">
        <main class="main-content">
            <br />

            <NodeSelection
                label="Select nodes"
                bind:selectedNodes={showedNodes}
                {nodes}
            />

            <Button>
                <Label>Check d-separation</Label>
            </Button>
        </main>
    </AppContent>
</div>

<style>
    .drawer-container {
        position: relative;
        display: flex;
        border: 1px solid
            var(--mdc-theme-text-hint-on-background, rgba(0, 0, 0, 0.1));
        overflow: hidden;
        z-index: 0;
    }
    * :global(.app-content) {
        flex: auto;
        overflow: auto;
        position: relative;
        flex-grow: 1;
    }
    .main-content {
        overflow: auto;
        padding: 16px;
        height: 100%;
        box-sizing: border-box;
    }
</style>
