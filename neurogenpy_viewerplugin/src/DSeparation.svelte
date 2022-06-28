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
    import Dialog, {
        Title as DialogTitle,
        Content as DialogContent,
        Actions,
    } from "@smui/dialog";
    import { getFromNeurogenpy } from "./request";

    export let nodes = [];
    let active = "X";
    let selectedNodes = { X: [], Y: [], Z: [] };
    let showedNodes = selectedNodes[active];
    let openDialog = false;
    let dseparated = undefined;

    function setActive(value) {
        selectedNodes[active] = showedNodes;
        active = value;
        showedNodes = selectedNodes[active];
    }

    async function checkDSeparation() {
        try {
            let result = await getFromNeurogenpy(
                "/grn/dseparated",
                JSON.stringify({
                    X: selectedNodes["X"],
                    Y: selectedNodes["Y"],
                    Z: selectedNodes["Z"],
                })
            );

            dseparated = result["result"];
            openDialog = true;
        } finally {
        }
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
                    on:click={() => setActive("X")}
                    activated={active === "X"}
                >
                    <Text>X nodes</Text>
                </Item>
                <Item
                    on:click={() => setActive("Y")}
                    activated={active === "Y"}
                >
                    <Text>Y nodes</Text>
                </Item>
                <Item
                    on:click={() => setActive("Z")}
                    activated={active === "Z"}
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

            <Button on:click={checkDSeparation}>
                <Label>Check d-separation</Label>
            </Button>
        </main>
    </AppContent>
    <Dialog
        bind:open={openDialog}
        aria-labelledby="simple-title"
        aria-describedby="simple-content"
    >
        <DialogTitle id="simple-title">D-Separation</DialogTitle>

        <DialogContent id="simple-content">
            X and Y are {dseparated ? "" : "not "}D-separated by Z.
        </DialogContent>

        <Actions>
            <Button>
                <Label>OK</Label>
            </Button>
        </Actions>
    </Dialog>
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

    * :global(.material-icons) {
        position: "absolute";
        left: "95%";
        top: "5%";
    }
</style>
