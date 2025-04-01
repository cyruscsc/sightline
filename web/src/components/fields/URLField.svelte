<script lang="ts">
	import { sightline } from '$lib/state.svelte';
</script>

<div class="form-group">
	<label for="arxiv-url">arXiv Paper URL</label>
	<input
		type="text"
		id="arxiv-url"
		class="input"
		bind:value={sightline.arxivUrl}
		placeholder="e.g., https://arxiv.org/abs/2303.08774"
	/>
</div>

<style>
	.form-group {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.input {
		padding: 0.5rem;
		border: 1px solid var(--border);
		border-radius: 8px;
		font-size: 1rem;
		background-color: var(--input-bg);
		color: var(--text);
		outline: none;
		transition: border-color 0.2s;
	}

	.input:focus {
		border-color: var(--border-focus);
	}
</style>
