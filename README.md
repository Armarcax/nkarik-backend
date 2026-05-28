# 🎨 Nkarik AI (Նկարիկ)

Nkarik AI is a multilingual Armenian children's drawing and AI art generation web application. It features a modular, age-grouped architecture designed to grow with the user.

## 🏗️ Architecture

The application is split into 4 standalone modules plus a central landing page:

1.  **🧸 Kid Mode (`nkarik_kid.html`) - Ages 1–4**
    *   Simple shapes, numbers 0–5, and letters Ա–Զ.
    *   Large touch targets and a 12-color palette.
    *   "Magic!" button for rainbow gradient fills and confetti celebrations.
    *   100% local rendering (No AI/Network required).

2.  **🎨 Junior Mode (`nkarik_junior.html`) - Ages 5–7**
    *   Alphabet tracing for Armenian (Ա–Ֆ), English (A–Z), and Russian (А–Я).
    *   Pixel-accurate tracing scoring engine.
    *   Local result generation with 3-star ratings.
    *   Multilingual UI support.

3.  **🤖 School Mode (`nkarik_school.html`) - Ages 8–17**
    *   AI-powered art generation via Pollinations AI.
    *   Modes: **Trace** (draw over guide), **Free** (doodle), and **Prompt** (text input).
    *   12 integrated art styles and automated Armenian/Russian dictionary translation.

4.  **🎓 Pro Mode (`nkarik_pro.html`) - Ages 18+**
    *   Advanced AI studio with granular controls.
    *   Seed control, CFG Scale (Guidance), and Resolution sliders (256px to 1024px).
    *   18+ curated art styles.
    *   Variation generation and high-detail toggles.

## 📂 Project Structure

```text
nkarik/
├── index.html          # Age selector landing page
├── nkarik_kid.html     # Toddler module
├── nkarik_junior.html  # Young module
├── nkarik_school.html  # School module
├── nkarik_pro.html     # Professional module
└── shared/
    ├── templates.js    # Shared SVG data, UI strings, and dictionary
    └── styles.css      # Shared CSS tokens and base styles
```

## 🚀 Local Testing

Since the application uses standard ES Modules and local file imports, it is best served via a local web server to avoid CORS or file-protocol restrictions in some browsers.

### Using Python
```bash
python3 -m http.server 8000
```
Then navigate to `http://localhost:8000`.

### Using Node.js (npx)
```bash
npx serve .
```
Then navigate to the provided local URL.

## 🛠️ Technical Fixes

*   **Browser Compatibility:** Replaced all global `event` references with explicit element passing to fix crashes in Safari and Firefox.
*   **Canvas Persistence:** Implemented `ImageData` caching to prevent drawing loss during window resizing.
*   **Download Integrity:** All PNG downloads are composited against a solid white background to ensure visibility.
*   **Performance:** All initialization logic moved to `requestAnimationFrame` to prevent race conditions during DOM load.

## 🇦🇲 Multilingual Support

The app supports Armenian (`hy`), English (`en`), and Russian (`ru`). Text translations are managed centrally in `shared/templates.js`.
