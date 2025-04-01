<script lang="ts">
	import { sightline } from '$lib/state.svelte';
	import Spinner from '$components/core/Spinner.svelte';
</script>

<div class="output-container">
	{#if sightline.isLoading}
		<div class="loading">
			<Spinner />
		</div>
	{:else if sightline.result}
		<div class="result">
			{@html sightline.result}
		</div>
	{:else}
		<div class="empty-result">
			<div>Results will appear here</div>
		</div>
	{/if}
</div>

<style>
	.output-container {
		background-color: var(--card-bg);
		border-radius: 16px;
		padding: 1.5rem;
		box-shadow: var(--shadow);
		margin-bottom: 2rem;
	}

	.loading,
	.empty-result {
		text-align: center;
		padding: 2rem 0;
		color: var(--small-text);
	}

	.result {
		line-height: 1.5;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}
</style>
