<script lang="ts">
	import { sightline } from '$lib/state.svelte';
	import { summarizePaper } from '$lib/api';

	async function handleSummarize() {
		if (!sightline.arxivUrl) {
			sightline.errorMessage = 'Please enter an arXiv paper URL';
			return;
		}

		sightline.errorMessage = '';
		sightline.isLoading = true;
		sightline.result = '';

		try {
			sightline.result = await summarizePaper(sightline.arxivUrl);
		} catch (error: any) {
			sightline.errorMessage = `Failed to get summary: ${error.message || 'Unknown error'}`;
		} finally {
			sightline.isLoading = false;
		}
	}
</script>

<button
	on:click={handleSummarize}
	class="button"
	disabled={sightline.isLoading || !sightline.arxivUrl}>Summarize</button
>

<style>
	.button {
		padding: 0.5rem 1rem;
		background-color: var(--btn-bg);
		color: var(--btn-text);
		border: none;
		border-radius: 8px;
		font-size: 1rem;
		cursor: pointer;
		transition: background-color 0.2s;
	}

	.button:hover {
		background-color: var(--btn-bg-hover);
	}

	.button:disabled {
		background-color: var(--btn-bg-disabled);
		cursor: not-allowed;
	}
</style>
