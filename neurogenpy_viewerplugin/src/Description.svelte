<script>
    let more = false;

    let par1 =
        "Learn gene regulatory networks modeled as Bayesian networks for a desired set of genes.";
</script>

<div class="description" style="font-size:12px;">
    {#if more}
        <p />
        {par1}
        <p>
            This tool downloads expression values of user specified sets of
            genes from Allen Brain API¹ and learns a Gene Regulatory network
            from it.
        </p>
        <p>
            It models the network as a Bayesian Network using the structure
            learning algorithm and parameter estimation method selected by the
            user. The data can be considered to have a Gaussian distribution or
            be discretized before the learning process in three levels:
            inhibition, activation and no-change.
        </p>
        <p>
            ¹For retrieving microarray data, brainscapes connects to the web API
            of the Allen Brain Atlas (© 2015 Allen Institute for Brain Science),
            available from
            <a href="https://brain-map.org/api/index.html"
                >https://brain-map.org/api/index.html</a
            >. Any use of the microarray data needs to be in accordance with
            their terms of use, as specified at
            <a href="https://alleninstitute.org/legal/terms-use/"
                >https://alleninstitute.org/legal/terms-use/</a
            >.
        </p>
        <a on:click|preventDefault={() => (more = false)} href="“#/”"
            >Read less</a
        >
    {:else}
        <p>
            {par1.substring(0, 33) + "..."}
            <a on:click|preventDefault={() => (more = true)} href="“#/”"
                >Read more</a
            >
        </p>
    {/if}
</div>
