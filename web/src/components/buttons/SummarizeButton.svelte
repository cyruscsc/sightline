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

<button on:click={handleSummarize} disabled={sightline.isLoading || !sightline.arxivUrl}>Summarize</button>

<style>
  button {
    padding: 0.5rem 1rem;
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.2s;
  }
  
  button:hover {
    background-color: #2563eb;
  }
  
  button:disabled {
    background-color: #9ca3af;
    cursor: not-allowed;
  }
</style> 