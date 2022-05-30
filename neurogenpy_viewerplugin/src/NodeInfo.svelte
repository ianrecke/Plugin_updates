<Card style="min-width: 300px;">
    <Media class="card-media-square" aspectRatio="square">
        <div style="color: #fff; position: absolute; bottom: 16px; left: 16px;">
            <h2 class="mdc-typography--headline6" style="margin: 0;">
                {nodeLabel}
            </h2>
            <button>
                Markov blanket
            </button>
        </div>
    </Media>
</Card>


<script>
    import Card, {Media} from '@smui/card';

    export let nodeLabel = undefined

</script>