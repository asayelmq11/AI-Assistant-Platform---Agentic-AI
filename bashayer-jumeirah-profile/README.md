# Bashayer Qaid — Personal Hospitality Profile

Source for `Bashayer_Qaid_Jumeirah_Hospitality_Profile.pdf`, a six-page A4 profile
prepared for Jumeirah Jabal Omar Makkah. It is an independent candidate document,
not a Jumeirah publication.

## Files

| File | Purpose |
|---|---|
| `index.html` | All copy. Each `<section class="page">` is one PDF page. |
| `styles.css` | Layout, typography, colours (tokens at the top under `:root`). |
| `fonts/` | Cormorant Garamond + Jost (SIL Open Font License), bundled so the PDF is identical on any machine. |
| `build.js` | Renders the PDF with headless Chromium and checks the layout. |
| `SOURCES.md` | Every factual claim in the profile and where it comes from. |

## Before sending: complete the contact details

Open `index.html`, find the `CONTACT DETAILS` comment on page 6, and replace
the two `<span class="to-complete">…</span>` placeholders with Bashayer's real
email and phone, e.g.

```html
<div class="v"><a href="mailto:name@example.com">name@example.com</a></div>
```

The build lists any remaining `to-complete` fields and prints
**NOT SEND-READY** until none are left.

## Rebuild the PDF

```bash
npm install          # first time only (installs Playwright)
npx playwright install chromium   # only if no Chromium is available
npm run build        # writes Bashayer_Qaid_Jumeirah_Hospitality_Profile.pdf
npm run preview      # also writes PNG previews to ./preview (needs pdftoppm/poppler)
```

The build fails (exit code 1) if any text runs past a page's footer or the
page count is not 6. If you add copy, shorten something else rather than
shrinking type below ~9pt.

No Node? Open `index.html` in Chrome → Print → Destination "Save as PDF",
Paper A4, Margins "None", "Background graphics" on.

## Editing rules

- Keep every statement consistent with `SOURCES.md`. Do not add dialogue,
  dates, names, metrics or outcomes that are not in the verified facts.
- The Hajj story (page 3) and the guest-care example (page 5, rows R and T)
  are Bashayer's verified accounts. Keep her actions separate from what the
  emergency responders did.
- Do not add Jumeirah's logo or anything that makes the document look official.
