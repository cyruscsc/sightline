<script lang="ts">
	import InputCard from '$components/cards/InputCard.svelte';
	import OutputCard from '$components/cards/OutputCard.svelte';
</script>

<svelte:head>
	<title>SightLine</title>
	<meta name="description" content="Get insights from arXiv papers" />
</svelte:head>

<main>
	<div class="container">
		<h1>SightLine</h1>
		<p>Get insights from arXiv papers</p>

		<InputCard />
		<OutputCard />
	</div>
</main>

<style>
	main {
		min-height: 100vh;
		padding: 2rem 1rem;
		background-color: #f9f9f9;
	}

	.container {
		max-width: 800px;
		margin: 0 auto;
	}

	h1 {
		font-size: 2.5rem;
		margin-bottom: 0.5rem;
		color: #333;
	}
</style>
