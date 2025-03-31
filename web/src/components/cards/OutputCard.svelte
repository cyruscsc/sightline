<script lang="ts">
	import { sightline } from '$lib/state.svelte.ts';
	import Spinner from '$components/Spinner.svelte';
</script>

<div class="output-container">
	<h2>Results</h2>
	{#if sightline.isLoading}
		<div class="loading">
			<Spinner size="3rem" />
			<div>Loading...</div>
		</div>
	{:else if sightline.result}
		<div class="result">
			<div>{@html sightline.result}</div>
		</div>
	{:else}
		<div class="empty-result">
			<div>Results will appear here</div>
		</div>
	{/if}
</div>

<style>
	.output-container {
		background-color: white;
		border-radius: 16px;
		padding: 1.5rem;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
		margin-bottom: 2rem;
	}

	h2 {
		margin-top: 0;
		margin-bottom: 1rem;
	}

	.loading,
	.empty-result {
		text-align: center;
		padding: 2rem 0;
		color: #6b7280;
	}

	.result {
		line-height: 1.5;
	}
</style>
