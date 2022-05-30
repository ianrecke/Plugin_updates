<div class="description" style="font-size:12px;">
    {#if more}
        <p>{par1}</p>
        <p>{par2}</p>
        <p>{par3}</p>
        <p>{par4_1} <a href="https://brain-map.org/api/index.html">https://brain-map.org/api/index.html</a>. {par4_2} <a
                href="https://alleninstitute.org/legal/terms-use/">https://alleninstitute.org/legal/terms-use/</a>.</p>
        <a on:click|preventDefault={() => more = false} href=“#/”>Read less</a>
    {:else}
        <p>{par1.substring(0, 33) + "..."} <a on:click|preventDefault={() => more = true} href=“#/”>Read more</a>
        </p>
    {/if}
</div>

<script>

    let more = false
    const par1 = "Learn gene regulatory networks modeled as Bayesian networks for a desired set of genes."
    const par2 = "This tool downloads expression values of user specified sets of genes from Allen Brain API\u00B9."
    const par3 = "\n\nAfter that, it learns the network using the selected structure learning algorithm and parameter" +
        " estimation method. "
    const par4_1 = "\u00B9For retrieving microarray data, brainscapes connects to the web API of the Allen Brain " +
        "Atlas (© 2015 Allen Institute for Brain Science), available from"
    const par4_2 = "Any use of the microarray data needs to be in accordance with their terms of use, as specified at"
</script>

