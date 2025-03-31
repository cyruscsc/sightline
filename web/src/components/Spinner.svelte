<script lang="ts">
  export let size = '2rem';
  export let color = '#3b82f6';
</script>

<div class="spinner" style="width: {size}; height: {size}; border-color: {color} transparent {color} transparent;">
</div>

<style>
  .spinner {
    display: inline-block;
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