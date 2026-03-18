# Personal Academic Website

A clean, responsive personal website template designed for CS PhD students and
researchers.  Built with plain HTML & CSS — no frameworks, no build step — so
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
4. Commit & push to the `main` branch.

## Enabling GitHub Pages

1. Go to **Settings → Pages** in your repository.
2. Under **Source**, choose **GitHub Actions**.
3. The included workflow (`.github/workflows/deploy.yml`) will automatically
   build and deploy the site on every push to `main`.

## Customization Checklist

| What to change | Where |
|---|---|
| Name, title, affiliation | `index.html` — hero section |
| Bio & research interests | `index.html` — `#about` section |
| Research projects | `index.html` — `#research` section |
| Publications | `index.html` — `#publications` section |
| Teaching experience | `index.html` — `#teaching` section |
| CV download link | `index.html` — `#cv` section |
| Contact details | `index.html` — `#contact` section |
| Social links | `index.html` — hero `.social-links` |
| Colours & fonts | `css/style.css` — `:root` variables |
| Profile photo | `images/profile.jpg` |

## Project Structure

```
.
├── index.html                  # Main page
├── css/
│   └── style.css               # All styles
├── images/
│   └── profile.jpg             # Your profile photo
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Pages deployment
└── README.md
```

## License

This project is available under the [MIT License](https://opensource.org/licenses/MIT).
Feel free to use, modify, and share it.