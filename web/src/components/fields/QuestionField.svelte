<script lang="ts">
  import { sightline } from '$lib/state.svelte';
</script>

<div class="form-group">
  <label for="question">Optional Question</label>
  <input 
    type="text"
    id="question" 
    bind:value={sightline.question} 
    placeholder="Ask a question about the paper..."
  />
</div>

<style>
  .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  input {
    padding: 0.5rem;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 1rem;
  }
</style> 