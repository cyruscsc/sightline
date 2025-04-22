<script lang="ts">
	import { sightline } from '$lib/state.svelte';

	let strategies = $state([
		{ name: 'Simple', value: 'simple' },
		{ name: 'Multi-Query', value: 'multi-query' },
		{ name: 'RAG-Fusion', value: 'rag-fusion' },
		{ name: 'HyDE', value: 'hyde' }
	]);
</script>

<div class="form-group">
	<label for="arxiv-url">Strategy</label>
	<select bind:value={sightline.strategy} class="select">
		{#each strategies as strategy}
			<option value={strategy.value}>{strategy.name}</option>
		{/each}
	</select>
</div>

<style>
	.form-group {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.select {
		padding: 0.5rem;
		border: 1px solid var(--border);
		border-radius: 8px;
		font-size: 1rem;
		background-color: var(--input-bg);
		color: var(--text);
		outline: none;
		transition: border-color 0.2s;
	}

	.select:focus {
		border-color: var(--border-focus);
	}
</style>
