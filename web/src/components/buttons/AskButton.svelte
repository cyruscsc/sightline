<script lang="ts">
  import { sightline } from '$lib/state.svelte.ts';
  import { askQuestion } from '$lib/api';
  
  async function handleAsk() {
    if (!sightline.arxivUrl) {
      sightline.errorMessage = 'Please enter an arXiv paper URL';
      return;
    }
    
    if (!sightline.question) {
      sightline.errorMessage = 'Please enter a question';
      return;
    }
    
    sightline.errorMessage = '';
    sightline.isLoading = true;
    sightline.result = '';
    
    try {
      sightline.result = await askQuestion(sightline.arxivUrl, sightline.question);
    } catch (error: any) {
      sightline.errorMessage = `Failed to get answer: ${error.message || 'Unknown error'}`;
    } finally {
      sightline.isLoading = false;
    }
  }
</script>

<button on:click={handleAsk} disabled={sightline.isLoading || !sightline.question}>Ask</button>

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