# Personal Academic Website

A clean, aesthetic personal website template designed for CS PhD students and
researchers. Built with plain HTML & CSS — no frameworks, no build step — so
it works out-of-the-box with **GitHub Pages**.

## Live Demo

Once deployed, the site will be available at:

```
https://sumust.github.io/
```

> **Note:** For the site to be served at the root URL above, the repository
> must be named `sumust.github.io`. If it has a different name, rename it
> under **Settings → General → Repository name**.

## Quick Start

1. **Fork / clone** this repository.
2. Open `index.html` and replace every placeholder (`Your Name`,
   `your@email`, university details, publications, etc.) with your own
   information.
3. Drop your profile photo into `images/profile.jpg` (the template gracefully
   hides the image area if the file is missing).
4. Drop your CV as `cv.pdf` in the project root (the nav and contact section
   link to it).
5. Commit & push to the `main` branch.

## Enabling GitHub Pages

1. Go to **Settings → Pages** in your repository.
2. Under **Source**, choose **GitHub Actions**.
3. The included workflow (`.github/workflows/deploy.yml`) will automatically
   build and deploy the site on every push to `main`.

## Customization Checklist

| What to change | Where |
|---|---|
| Name, title, affiliation | `index.html` — hero section |
| Favourite quote | `index.html` — `.quote-section` |
| Bio & research interests | `index.html` — `#about` section |
| Research projects | `index.html` — `#research` section |
| Selected publications (teaser) | `index.html` — `#publications` section |
| Full publications list | `publications.html` |
| Blog posts | `blog.html` + files in `blog/` |
| Teaching experience | `index.html` — `#teaching` section |
| CV download | Place `cv.pdf` in the project root |
| Contact details | `index.html` — `#contact` section |
| Social links | `index.html` — hero `.social-links` + footer |
| Colours & fonts | `css/style.css` — `:root` variables |
| Profile photo | `images/profile.jpg` |

## Adding a Blog Post

1. Write a Markdown file in `blog/posts/` (e.g. `blog/posts/my-new-post.md`).
2. Add an entry to `blog/posts.json`:
   ```json
   {
     "slug": "my-new-post",
     "title": "My New Post Title",
     "date": "2025-04-01",
     "tag": "Research",
     "readTime": "5 min read",
     "summary": "A short description shown on the blog listing page."
   }
   ```
3. Commit and push — that's it! The blog listing and post pages render
   automatically from the Markdown files.

## Project Structure

```
.
├── index.html                          # Main page (home)
├── publications.html                   # Full publications list
├── blog.html                           # Blog listing (auto-rendered)
├── blog/
│   ├── post.html                       # Blog post template (renders Markdown)
│   ├── posts.json                      # Blog manifest (add entries here)
│   └── posts/                          # Markdown blog posts
│       ├── getting-started-with-research.md
│       ├── gentle-intro-to-transformers.md
│       ├── research-workflow.md
│       └── neurips-2024.md
├── css/
│   └── style.css                       # All styles
├── images/
│   └── profile.jpg                     # Your profile photo
├── cv.pdf                              # Your CV (add your own)
├── .github/
│   └── workflows/
│       └── deploy.yml                  # GitHub Pages deployment
└── README.md
```

## License

This project is available under the [MIT License](https://opensource.org/licenses/MIT).
Feel free to use, modify, and share it.