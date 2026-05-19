(function () {
  document.getElementById('year').textContent = new Date().getFullYear();

  var params = new URLSearchParams(window.location.search);
  var slug = params.get('post');

  if (!slug) {
    var cleanPath = window.location.pathname.replace(/\/(?:index\.html)?$/, '');
    var lastSegment = cleanPath.split('/').pop();
    if (lastSegment && !['post.html', 'post', 'blog'].includes(lastSegment)) {
      slug = lastSegment;
    }
  }

  var headerEl = document.getElementById('post-header');
  var bodyEl = document.getElementById('post-body');

  if (!slug || !/^[a-zA-Z0-9_-]+$/.test(slug)) {
    headerEl.innerHTML = '<h1>Post not found</h1>';
    bodyEl.innerHTML = '<p>Sorry, this post does not exist. <a href="../blog.html">Back to blog</a>.</p>';
    return;
  }

  function formatDate(dateStr) {
    var d = new Date(dateStr + 'T00:00:00');
    return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
  }

  fetch('posts.json')
    .then(function (r) { return r.json(); })
    .then(function (posts) {
      var meta = null;
      for (var i = 0; i < posts.length; i++) {
        if (posts[i].slug === slug) { meta = posts[i]; break; }
      }

      return fetch('posts/' + slug + '.md')
        .then(function (r) {
          if (!r.ok) throw new Error('Not found');
          return r.text();
        })
        .then(function (md) {
          if (meta) {
            document.title = meta.title + ' - Lain Mustafaoglu';
            headerEl.innerHTML =
              '<div class="post-meta">' +
                '<span>' + formatDate(meta.date) + '</span>' +
                '<span class="blog-tag">' + meta.tag + '</span>' +
              '</div>' +
              '<h1>' + meta.title + '</h1>' +
              '<p class="post-author">By ' + (meta.author || 'Lain Mustafaoglu') + '</p>';
          } else {
            headerEl.innerHTML = '';
          }

          var cleanMd = md.replace(/^#\s+.+\n*/, '');
          marked.use({ gfm: true, breaks: false });
          bodyEl.innerHTML = marked.parse(cleanMd);
          if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([bodyEl]);
          }
        });
    })
    .catch(function () {
      headerEl.innerHTML = '<h1>Post not found</h1>';
      bodyEl.innerHTML = '<p>Sorry, we couldn\'t load this post. <a href="../blog.html">Back to blog</a>.</p>';
    });
})();
