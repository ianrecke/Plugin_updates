import App from './App.svelte';

const app = setTimeout(() => {
    new App({
        target: document.querySelector("ng-component[plugincontainer='true']"), props: {}
    });
}, 200);

window.addEventListener('pagehide', () => {
    app.$destroy()
})

export default app;