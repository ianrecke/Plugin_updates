(()=>{
  /* your code here */

  if(interactiveViewer.pluginControl['upm.cig.neurogenpy'].initState){
    /* init plugin with initState */
  }

  const submitButton = document.getElementById('upm.cig.neurogenpy.submit')
  submitButton.addEventListener('click',(ev)=>{
    console.log('submit button was clicked')
  })
})()