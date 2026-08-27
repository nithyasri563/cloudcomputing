// Micro Climate Crop Advisor — Global JS utilities

// Flash message auto-dismiss
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 400); }, 4000);
    el.style.transition = 'opacity .4s';
  });
});
