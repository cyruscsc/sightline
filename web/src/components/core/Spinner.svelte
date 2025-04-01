<div class="spinner"></div>

<style>
	.spinner {
		display: inline-block;
		width: 2rem;
		height: 2rem;
		border-color: var(--spinner-color) transparent var(--spinner-color) transparent;
		border-width: 0.25rem;
		border-style: solid;
		border-radius: 50%;
		animation: spin 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
	}

	@keyframes spin {
		0% {
			transform: rotate(0deg);
		}
		100% {
			transform: rotate(360deg);
		}
	}
</style>
