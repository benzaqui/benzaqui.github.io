# CLAUDE.md

This file provides guidance for AI assistants working in this repository.

## Project Overview

Personal portfolio website for Arthur Benzaquin (Microengineering engineer, EPFL graduate), hosted on GitHub Pages at `benzaqui.github.io`. The site is a static HTML/CSS page with no backend, no build pipeline, and no JavaScript frameworks.

## Repository Structure

```
benzaqui.github.io/
├── index.html              # Main homepage (single-page portfolio)
├── 404.html                # Custom 404 error page
├── CNAME                   # GitHub Pages custom domain config (currently empty)
├── assets/
│   ├── css/
│   │   ├── main.css        # Source stylesheet (edit this one)
│   │   └── main.min.css    # Minified CSS (used in production — regenerate after edits)
│   ├── img/
│   │   └── favicon/        # Favicon set for all platforms
│   │       ├── favicon.ico
│   │       ├── favicon-16x16.png
│   │       ├── favicon-32x32.png
│   │       ├── apple-touch-icon.png
│   │       ├── android-chrome-192x192.png
│   │       ├── android-chrome-512x512.png
│   │       ├── manifest.json       # PWA web app manifest
│   │       └── browserconfig.xml   # Windows tile config
│   ├── web_resume.pdf      # Resume document (linked from homepage)
│   └── FinalFootYProductDesign2019.pdf  # EPFL project archive (linked as Archives)
```

## Technology Stack

- **HTML5** — semantic markup
- **CSS3** — vanilla CSS with animations and media queries; no preprocessors
- **Google Fonts** — Open Sans (weights 300, 400, 700), loaded via CDN
- **No JavaScript** — the site has zero JS
- **No build tools** — no npm, webpack, Gulp, or Grunt
- **GitHub Pages** — automatic deployment from the `master` branch

## CSS Conventions

- Edit `assets/css/main.css` (the human-readable source).
- After editing, regenerate `main.min.css` by stripping whitespace/comments (the HTML references the minified file).
- The CSS uses a normalize/reset block attributed to `http://alessandroscarpellini.it/`.
- Responsive breakpoints: `500px`, `700px`, `1100px`.
- Color scheme: dark blue gradient background (`#141E30` → `#243b55`), white text.
- Animations: CSS keyframe `slideInUp`, staggered with delays from `650ms` to `1050ms`.
- Font sizes scale from `44px` (mobile) to `64px` (desktop).

## HTML Conventions

- Both `index.html` and `404.html` link `main.min.css` (not `main.css`).
- The 404 page contains some leftover template metadata (references to "Vincent Ballet") — update if refreshing that page.
- Internal links to assets use relative paths (e.g., `assets/web_resume.pdf`).

## Deployment

- **Automatic**: Pushing to `master` deploys to GitHub Pages immediately.
- **Custom domain**: Configure by writing the domain into `CNAME` (currently empty → deployed at `benzaqui.github.io`).
- No CI/CD pipeline exists — deployment is handled entirely by GitHub Pages.

## Development Workflow

1. Make changes directly to HTML/CSS files — no build step required for previewing.
2. If CSS was edited, minify `main.css` → `main.min.css` before committing.
3. Commit with a clear message describing what changed.
4. Push to `master` to deploy.

## Key Constraints

- Keep the site dependency-free — no npm packages, no JS frameworks, no CDN dependencies beyond Google Fonts.
- Do not add JavaScript unless explicitly requested.
- Do not introduce a build system unless explicitly requested.
- PDF assets are large (6 MB for the project archive); avoid adding more large binaries.
- The site is intentionally minimal — prefer small, focused changes over refactors.
