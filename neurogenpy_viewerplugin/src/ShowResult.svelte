<FormField>
  <Switch
    bind:checked
    on:SMUISwitch:change={handleSwitchEvent}/>
  <span slot="label">Annotate</span>
</FormField>

<script>
import Switch from "@smui/switch"
import FormField from '@smui/form-field'
import { onDestroy } from "svelte";

const MNI152ID = "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2"

export let postMessage = async (...arg) => { throw new Error(`postMessage need to be overriden by parent!`) }
export let result;
let checked = false

let addedAnnotations = []

function handleSwitchEvent(event) {
  if (!result) throw new Error(`result is not defined`)
  const flag = event.detail.selected
  if (flag) {
    addedAnnotations = result.mnicoords.map(({ mnicoord, roi }) => {
      return mnicoord.map((coord, idx) => {
        return {
          '@id': `neurogenpy-${roi}-${idx}`,
          name: `${roi}: ${idx}`,
          description: `${roi}: ${idx}: ${JSON.stringify(coord)}`,
          color: '#ffffff',
          openminds: {
            coordinateSpace: {
              "@id": MNI152ID
            },
            "@type": "https://openminds.ebrains.eu/sands/CoordinatePoint",
            "@id": `neurogenpy-${roi}-${idx}`,
            coordinates: coord.map(c => {
              return {
                value: c
              }
            })
          }
        }
      })
    }).flatMap(v => v)
    postMessage({
      method: `sxplr.addAnnotations`,
      params: {
        annotations: addedAnnotations
      }
    })
  } else {
    postMessage({
      method: `sxplr.rmAnnotations`,
      params: {
        annotations: addedAnnotations
      }
    })
    addedAnnotations = []
  }
}

onDestroy(() => {
  postMessage({
    method: `sxplr.rmAnnotations`,
    params: {
      annotations: addedAnnotations
    }
  })
})


</script>

