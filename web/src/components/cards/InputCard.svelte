<script lang="ts">
	import { sightline } from '$lib/state.svelte';
	import URLField from '$components/fields/URLField.svelte';
	import QuestionField from '$components/fields/QuestionField.svelte';
	import StrategyField from '$components/fields/StrategyField.svelte';
	import SummarizeButton from '$components/buttons/SummarizeButton.svelte';
	import AskButton from '$components/buttons/AskButton.svelte';
</script>

<div class="input-container">
	<URLField />
	<div class="question-group">
		<QuestionField />
		<StrategyField />
	</div>

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
		background-color: var(--card-bg);
		border-radius: 16px;
		padding: 1.5rem;
		box-shadow: var(--shadow);
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.question-group {
		display: flex;
		gap: 1rem;
	}

	.button-group {
		margin-top: 0.5rem;
		display: flex;
		gap: 1rem;
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
