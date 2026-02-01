# Report Assistant UI – Styling Specification

## Overall Philosophy
- Minimalist, calm, professional
- Finance / research oriented (credible, not playful)
- Strong hierarchy: headline → subtext → primary action
- Light mode only
- Avoid visual noise; rely on spacing and typography rather than borders

---

## Color Palette
Use neutral, warm-leaning light tones with high contrast text.

- Background: `#FAFAF8` or `#F7F7F5`
- Primary text: `#1F1F1F`
- Secondary text: `#5F5F5F`
- Accent / CTA: `#1F2937` (dark gray, not pure black)
- Accent hover: `#111827`
- Divider / subtle borders: `#E5E7EB`

Avoid bright blues or saturated colors.

---

## Typography
Use system fonts only (Streamlit-safe).

Recommended stack:
```
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
Inter, Helvetica, Arial, sans-serif;
```

### Font Scale
- Hero headline: 48–56px, serif-like feel (simulate via font-weight + spacing)
- Section headline: 24–28px
- Body text: 15–16px
- Metadata / labels: 12–13px

Font weights:
- Headline: 500–600
- Body: 400
- UI labels: 500

---

## Layout & Spacing
- Max content width: 1100–1200px
- Centered layout
- Generous vertical whitespace
- Use padding instead of boxes wherever possible

Example spacing rhythm:
- Hero → subtext: 16px
- Subtext → CTA: 32px
- Section blocks: 64–96px apart

---

## Buttons
Primary CTA only (avoid multiple competing buttons).

- Rounded pill shape
- Padding: 14px 28px
- No icons unless necessary
- Text: concise and action-oriented

Hover:
- Slight darkening
- No animation beyond color change

---

## Cards / Containers
Used sparingly (e.g. file selection UI).

- Light background (`#FFFFFF`)
- Soft shadow (very subtle)
- Rounded corners (8–12px)
- No heavy borders

---

## Tables & Lists
For file selection / report lists:

- Zebra striping very subtle or none
- Row hover highlight
- Clear typography over visual chrome
- Align numbers right, text left

---

## Icons
- Optional
- If used, monochrome only
- Never decorative; functional only

---

## CSS Constraints (Streamlit)
- No external CSS frameworks
- Prefer class selectors over element overrides
- Avoid breaking default Streamlit components
- Because streamlit does not allow importing css using 
```html
<link rel="stylesheet" href="styles.css">
```
instead we use the streamlit way:
```python
css = Path("ui/assets/styles.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
```

---

## Overall Reference Mood
- “Premium SaaS landing page”
- “Quiet confidence”
- “Analyst-grade, not consumer-grade”
