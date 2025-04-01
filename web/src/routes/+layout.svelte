<script lang="ts">
	import Header from '$components/core/Header.svelte';
	import Footer from '$components/core/Footer.svelte';
	import '$lib/global.css';
</script>

<div class="layout">
	<Header />
	<slot />
	<Footer />
</div>

<style>
	.layout {
		display: flex;
		flex-direction: column;
		min-height: 100vh;
	}
</style>
