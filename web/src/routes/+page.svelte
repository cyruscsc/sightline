<script lang="ts">
	import InputCard from '$components/cards/InputCard.svelte';
	import OutputCard from '$components/cards/OutputCard.svelte';
</script>

<svelte:head>
	<title>SightLine</title>
	<meta name="description" content="Get insights from arXiv papers" />
</svelte:head>

<div class="container">
	<InputCard />
	<OutputCard />
</div>

<style>
	.container {
		max-width: 48rem;
		width: 100%;
		margin: 0 auto;
		display: flex;
		flex-direction: column;
		gap: 2rem;
		flex: 1;
		padding: 1rem 0;
	}
</style>
