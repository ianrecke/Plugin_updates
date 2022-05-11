import App from './App.svelte';

const app = new App({
	target: document.body,
	props: {
		
	}
});

window.addEventListener('pagehide', () => {
	app.$destroy()
})

export default app;