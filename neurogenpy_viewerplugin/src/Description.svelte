<div class="description" style="font-size:12px;">
    {#if more}
        <p>{par1}</p>
        <p>{par2}</p>
        <p>{par3}</p>
        <p>{par4}</p>
    {:else}
        <p>{par1.substring(0, 33) + "..."}</p>
    {/if}
</div>

<Button
        on:click={() => more = !more}
        id="loadmore"
        type="button"
        class="btn btn-secondary">
    {more ? "Read less" : "Read more"}
</Button>

<script>
    import Button from "@smui/button";

    let more = false
    const par1 = "Learn gene regulatory networks modeled as Bayesian networks for a desired set of genes."
    const par2 = "This tool downloads expression values of user specified sets of genes from Allen Brain API\u00B9."
    const par3 = "\n\nAfter that, it learns the network using the selected structure learning algorithm and parameter" +
        " estimation method. "
    const par4 = "\u00B9For retrieving microarray data, brainscapes connects to the web API of the Allen Brain Atlas " +
        "(© 2015 Allen Institute for Brain Science), available from https://brain-map.org/api/index.html. Any use of " +
        "the microarray data needs to be in accordance with their terms of use, as specified at " +
        "https://alleninstitute.org/legal/terms-use/."
</script>

