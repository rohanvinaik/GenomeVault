# Visitor Tracking Setup for GenomeVault Repository

Industry-standard, privacy-compliant methods to track who's viewing your repository.

---

## Option 1: GitHub Profile Views Counter (Simplest)

Add this badge to your README.md:

```markdown
![Profile Views](https://komarev.com/ghpvc/?username=rohanvinaik&label=Profile%20Views&color=brightgreen)
```

**Pros:**
- Free, no signup
- Shows total profile views
- Privacy-compliant (no personal data)

**Cons:**
- Only tracks profile views, not repo-specific

---

## Option 2: Shields.io + Analytics (Repository-Specific)

### A. Add Repository-Specific Badges

```markdown
![GitHub stars](https://img.shields.io/github/stars/rohanvinaik/GenomeVault?style=social)
![GitHub forks](https://img.shields.io/github/forks/rohanvinaik/GenomeVault?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/rohanvinaik/GenomeVault?style=social)
![GitHub repo size](https://img.shields.io/github/repo-size/rohanvinaik/GenomeVault)
![GitHub last commit](https://img.shields.io/github/last-commit/rohanvinaik/GenomeVault)
```

### B. Add Visit Counter (via visitors-counter.com)

```markdown
![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=rohanvinaik.GenomeVault)
```

**Pros:**
- Free, repo-specific
- Shows incremental visitor count
- No signup required

**Cons:**
- Counts badge loads, not unique humans
- No detailed analytics

---

## Option 3: Google Analytics for GitHub Pages (Most Detailed)

If you enable GitHub Pages for documentation:

### Setup:
1. Enable GitHub Pages in repo settings
2. Create `docs/index.html` or use existing README
3. Add Google Analytics tracking code

**Example tracking snippet:**
```html
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

**Pros:**
- Detailed analytics (location, device, browser, referrers)
- Real-time visitor tracking
- Bounce rate, session duration, etc.

**Cons:**
- Requires GitHub Pages enabled
- Only tracks page views, not repo clones
- Requires Google Analytics account (free)

---

## Option 4: Plausible Analytics (Privacy-Focused Alternative)

### Self-hosted or Cloud
```html
<script defer data-domain="yourusername.github.io" src="https://plausible.io/js/script.js"></script>
```

**Pros:**
- GDPR-compliant, no cookies
- Privacy-focused (doesn't track individuals)
- Clean, simple dashboard
- Shows visitor sources, top pages, devices

**Cons:**
- Paid service ($9/month) or requires self-hosting
- Requires GitHub Pages

---

## Option 5: Simple Analytics (Cookie-Free)

```html
<script async defer src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
<noscript><img src="https://queue.simpleanalyticscdn.com/noscript.gif" alt="" referrerpolicy="no-referrer-when-downgrade" /></noscript>
```

**Pros:**
- No cookies, GDPR-compliant
- Shows visitor trends, referrers, pages
- $19/month with 100k pageviews

**Cons:**
- Paid service
- Requires GitHub Pages

---

## Option 6: Cloudflare Web Analytics (Free, Privacy-First)

Add to your GitHub Pages:
```html
<script defer src='https://static.cloudflare.com/beacon.min.js' data-cf-beacon='{"token": "YOUR_TOKEN_HERE"}'></script>
```

**Pros:**
- Completely free
- Privacy-first (no personal data)
- Shows pageviews, visitors, referrers, top pages
- Works with GitHub Pages

**Cons:**
- Requires Cloudflare account
- Requires GitHub Pages

---

## Option 7: GitHub Stars Notifications (Track Engagement)

Set up email notifications for stars/forks/watchers:

1. Go to https://github.com/rohanvinaik/GenomeVault/subscription
2. Enable "Watching" → "All Activity"
3. You'll get emails when people star/fork/watch

**Pros:**
- Built-in GitHub feature
- Real-time notifications
- No setup required

**Cons:**
- Only tracks engagement, not views
- Can be noisy

---

## Option 8: Custom Tracking Pixel (Advanced)

Add a 1x1 transparent image to README that logs requests:

### Setup:
1. Create a free account on Logflare, Supabase, or your own server
2. Add tracking pixel to README:

```markdown
![Tracker](https://your-tracking-endpoint.com/pixel.gif?repo=genomevault&page=readme)
```

**Pros:**
- Fully customizable
- Can track referrers, user agents, timestamps
- Own your data

**Cons:**
- Requires backend setup
- May violate GitHub ToS if too invasive

---

## Recommended Approach for GenomeVault

**Immediate (No Setup):**
```markdown
# Add to README.md
![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=rohanvinaik.GenomeVault)
![GitHub stars](https://img.shields.io/github/stars/rohanvinaik/GenomeVault?style=social)
![GitHub forks](https://img.shields.io/github/forks/rohanvinaik/GenomeVault?style=social)
```

**For Detailed Analytics (Recommended):**
1. Enable GitHub Pages (Settings → Pages → Deploy from branch `main` / `docs`)
2. Add Cloudflare Web Analytics (free) or Google Analytics
3. Set up custom domain (optional): `genomevault.dev`

**For Academic/Research Context:**
- Add Zenodo DOI badge (also tracks downloads)
- Submit to Papers with Code (tracks citations + implementations)

---

## Example README.md Addition

```markdown
## Repository Stats

![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=rohanvinaik.GenomeVault)
![GitHub stars](https://img.shields.io/github/stars/rohanvinaik/GenomeVault?style=social)
![GitHub forks](https://img.shields.io/github/forks/rohanvinaik/GenomeVault?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/rohanvinaik/GenomeVault)
![License](https://img.shields.io/github/license/rohanvinaik/GenomeVault)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

---

## Privacy Considerations

All recommended methods are:
- ✅ GDPR-compliant
- ✅ No personal data collection
- ✅ Cookie-free options available
- ✅ Industry-standard tools
- ✅ Transparent to users

**Avoid:**
- ❌ Fingerprinting techniques
- ❌ Third-party spyware
- ❌ Hidden tracking without disclosure
- ❌ Collecting IP addresses without notice

---

## Next Steps

1. Choose a tracking method (I recommend visitor badge + GitHub Pages + Cloudflare Analytics)
2. Update README.md with badges
3. Enable GitHub Pages for documentation
4. Add analytics tracking code to docs
5. Monitor weekly using GitHub's native traffic insights + your chosen analytics

Would you like me to implement any of these options for you?
