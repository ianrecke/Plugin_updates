<script>
    import Dialog, { Header, Title, Content } from "@smui/dialog";
    import IconButton from "@smui/icon-button";
    import Button, { Label } from "@smui/button";

    let open = false;
</script>

<Button on:click={() => (open = true)}>
    <Label>
        <span class="material-icons">help</span>
    </Label>
</Button>

<Dialog
    bind:open
    fullscreen
    aria-labelledby="fullscreen-title"
    aria-describedby="fullscreen-content"
>
    <Header>
        <Title id="fullscreen-title">How to use</Title>
        <IconButton action="close" class="material-icons">close</IconButton>
    </Header>
    <Content id="fullscreen-content">
        <p>
            The learned network shown can be easily manipulated and some results
            can be obtained from it. Some of them are specific for Bayesian
            Network models. Their meanings can be easily found in any basic
            reference about Bayesian networks. The functionality is provided by
            <a href="https://neurogenpy.readthedocs.io/en/latest/">NeurogenPy</a
            >.
        </p>

        <br />
        <p>
            - <b>Graph manipulations</b>: the graph can be easily manipulated
            (move the nodes, zoom in and out, show and hide labels, etc.)
        </p>
        <p>
            - <b>Select a node</b>: a node can be selected by searching it or by
            clicking on it. Once it is selected, the visualization is restricted
            to the node and its neighbors, the probability distribution for the
            node is shown and some node particular operations are displayed.
        </p>
        <p>
            - <b>Layouts</b>: different layouts can be selected for visualizing
            the graph (ForceAtlas2, Circular, Sugiyama, etc.)
        </p>
        <p>
            - <b>Download</b>: different available formats (JSON, GEXF, PNG,
            etc.) can be downloaded. Some of them download the information of
            the full network and others only download the structure of the
            network.
        </p>
        <p>
            - <b>Communities Louvain</b>: Color your graph according to the
            different communities it presents using the Louvain method.
        </p>
        <p>
            - <b>D-Separation</b>: Check if two sets of nodes X and Y are
            d-separated by another set of nodes Z. D-Separation is a graph
            criterion that implies probabilistic conditional independence.
        </p>
        <p>
            - <b>Markov Blanket</b>: Find the Markov blanket of the selected
            node.
        </p>
        <p>
            - <b>Inference</b>: Set the evidence for some nodes and select
            "Infer". The new distribution according to the current evidence for
            the selected node will be shown in the same chart as the initial
            one.
        </p>
    </Content>
</Dialog>
