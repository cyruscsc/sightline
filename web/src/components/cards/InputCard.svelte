<script lang="ts">
	import { sightline } from '$lib/state.svelte.ts';
	import URLField from '$components/fields/URLField.svelte';
	import QuestionField from '$components/fields/QuestionField.svelte';
	import SummarizeButton from '$components/buttons/SummarizeButton.svelte';
	import AskButton from '$components/buttons/AskButton.svelte';
</script>

<div class="input-container">
	<URLField />
	<QuestionField />

	<div class="button-group">
		<SummarizeButton />
		<AskButton />
	</div>

	{#if sightline.errorMessage}
		<div class="error-message">{sightline.errorMessage}</div>
	{/if}
</div>

<style>
	.input-container {
		background-color: white;
		border-radius: 16px;
		padding: 1.5rem;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
		margin-bottom: 2rem;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.button-group {
		margin-top: 0.5rem;
		display: flex;
		gap: 1rem;
	}

	.error-message {
		padding: 0.75rem;
		background-color: #fee2e2;
		color: #b91c1c;
		border-radius: 4px;
	}

	@media (max-width: 640px) {
		.button-group {
			flex-direction: column;
		}

		:global(.button-group button) {
			width: 100%;
		}
	}
</style>
