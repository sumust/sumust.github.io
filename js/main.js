// Hamburger menu toggle
const hamburger = document.getElementById('hamburger');
const siteNav   = document.getElementById('site-nav');

if (hamburger && siteNav) {
  hamburger.addEventListener('click', () => {
    siteNav.classList.toggle('open');
    hamburger.setAttribute('aria-expanded',
      siteNav.classList.contains('open').toString());
  });
  // Close nav when a link is clicked on mobile
  siteNav.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => siteNav.classList.remove('open'));
  });
}

// Mark the active nav link based on the current page filename
(function markActiveNav() {
  const path = window.location.pathname;
  const filename = path.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-link').forEach(link => {
    const href = link.getAttribute('href').split('/').pop();
    if (href === filename || (filename === '' && href === 'index.html')) {
      link.classList.add('active');
    }
  });
})();
