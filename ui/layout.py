"""
DoodleBook — single-tab Storybook UI.

Construction-paper scrapbook aesthetic: Gaegu/Caveat fonts, paper textures,
hand-drawn SVG frames, floating crayon sparkles. Fully custom — zero Gradio defaults.
"""

import gradio as gr
from config import VOICE_CHOICES, DEFAULT_VOICE

THEMES = [
    "brave adventure",
    "making a new friend",
    "overcoming a fear",
    "helping someone",
    "lost and found",
    "learning something new",
    "kindness to animals",
    "the magic of imagination",
    "celebrating who you are",
    "a rainy day adventure",
]

HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Gaegu:wght@400;700&family=Caveat:wght@500;700&family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
"""

SVG_DEFS = """
<svg width="0" height="0" aria-hidden="true" style="position:absolute">
  <filter id="wobble">
    <feTurbulence type="fractalNoise" baseFrequency="0.012 0.018"
                  numOctaves="2" seed="7" result="noise"/>
    <feDisplacementMap in="SourceGraphic" in2="noise" scale="7"
                       xChannelSelector="R" yChannelSelector="G"/>
  </filter>
  <filter id="wobble-strong">
    <feTurbulence type="fractalNoise" baseFrequency="0.02"
                  numOctaves="3" seed="3" result="n"/>
    <feDisplacementMap in="SourceGraphic" in2="n" scale="4"/>
  </filter>
</svg>
"""

CSS = r"""
/* ============================================================================
   DOODLEBOOK — CONSTRUCTION-PAPER SCRAPBOOK
   ============================================================================ */

:root {
  --paper:      #f6ecd4;
  --paper-2:    #efe0c2;
  --ink:        #2e2a26;
  --ink-soft:   #6b5d4f;
  --crayon-orange: #ef6a3a;
  --crayon-teal:   #2ba39a;
  --crayon-sun:    #f4c64a;
  --crayon-berry:  #d6517a;
  --crayon-sky:    #4a9fd6;
  --crayon-leaf:   #74b85a;
  --tape:       rgba(244, 198, 74, 0.55);
}

.gradio-container,
.gradio-container *:not(svg):not(path) {
  --block-background-fill: transparent;
  --block-border-width: 0px;
  --block-shadow: none;
  --panel-background-fill: transparent;
  --input-background-fill: #fffdf6;
  --body-text-color: var(--ink);
}

.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  background: transparent !important;
  font-family: 'Nunito', sans-serif !important;
  color: var(--ink);
}

body, gradio-app {
  background-color: var(--paper) !important;
  background-image:
    radial-gradient(circle at 12% 18%, rgba(239,106,58,0.10), transparent 38%),
    radial-gradient(circle at 88% 12%, rgba(43,163,154,0.10), transparent 40%),
    radial-gradient(circle at 70% 88%, rgba(214,81,122,0.08), transparent 42%),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E") !important;
  background-attachment: fixed !important;
}

/* ============================== KID ANIMATIONS ============================== */
.db-spark {
  position: absolute;
  border-radius: 50%;
  opacity: 0;
  animation: db-sparkle-float linear infinite;
}
@keyframes db-sparkle-float {
  0%   { transform: translateY(0) scale(0.6) rotate(0deg);   opacity: 0; }
  15%  { opacity: 0.55; }
  85%  { opacity: 0.3; }
  100% { transform: translateY(-110px) scale(1.1) rotate(200deg); opacity: 0; }
}

/* ============================== HEADER ============================== */
.app-header {
  text-align: center;
  padding: 34px 16px 18px;
  position: relative;
}
.app-title {
  font-family: 'Gaegu', cursive !important;
  font-weight: 700 !important;
  font-size: clamp(44px, 7vw, 78px) !important;
  line-height: 0.95 !important;
  color: var(--ink) !important;
  margin: 0 !important;
  text-shadow: 2px 2px 0 var(--crayon-sun), 4px 4px 0 rgba(239,106,58,.35);
}
.app-title .doodle-o { color: var(--crayon-orange); display: inline-block; transform: rotate(-6deg); }
.app-subtitle {
  font-family: 'Caveat', cursive !important;
  font-size: clamp(20px, 3vw, 28px) !important;
  color: var(--ink-soft) !important;
  margin-top: 6px !important;
}
.title-squiggle { display:block; margin: 6px auto 0; width: min(360px, 70%); height: 16px; }
.title-squiggle path {
  stroke: var(--crayon-teal); stroke-width: 5; fill: none;
  stroke-linecap: round; filter: url(#wobble-strong);
}

/* Feature strip */
.feature-strip {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px 16px;
  margin: 18px auto 4px;
  max-width: 700px;
}
.feature-strip span {
  font-family: 'Gaegu', cursive;
  font-size: clamp(15px, 2vw, 19px);
  font-weight: 700;
  color: var(--ink);
  background: #fffdf6;
  border: 2.5px solid var(--ink);
  border-radius: 12px;
  padding: 5px 14px;
  box-shadow: 2px 3px 0 rgba(46,42,38,.12);
  display: inline-block;
}
.feature-strip span:nth-child(1) { transform: rotate(-1.5deg); border-color: var(--crayon-orange); }
.feature-strip span:nth-child(2) { transform: rotate(1deg);   border-color: var(--crayon-teal);   }
.feature-strip span:nth-child(3) { transform: rotate(-0.8deg); border-color: var(--crayon-berry); }
.feature-strip span:nth-child(4) { transform: rotate(1.2deg);  border-color: var(--crayon-sky);   }
.feature-strip span:nth-child(5) { transform: rotate(-1deg);   border-color: var(--crayon-leaf);  }

/* ============================== STORYBOOK CARDS ============================== */
.input-card, .output-card {
  position: relative;
  background: #fffdf6 !important;
  border-radius: 18px;
  padding: 30px 26px 26px;
  margin: 8px;
  box-shadow: 0 10px 26px rgba(46,42,38,.12), 0 2px 0 rgba(0,0,0,.04);
}
.input-card  { transform: rotate(-0.7deg); }
.output-card { transform: rotate(0.5deg); }
.input-card:hover, .output-card:hover { transform: rotate(0deg); transition: transform .35s ease; }

.input-card .form, .input-card .block, .input-card .panel,
.input-card .wrap, .input-card .gap, .input-card .styler,
.output-card .form, .output-card .block, .output-card .panel,
.output-card .gap, .output-card .styler {
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}
.input-card .dataset, .input-card [data-testid="dataset"],
.input-card table, .input-card thead, .input-card tbody,
.input-card tr, .input-card th, .input-card td {
  background: #fffdf6 !important;
  color: var(--ink) !important;
  border-color: var(--ink-soft) !important;
}
.input-card .dataset * { color: var(--ink) !important; }

.card-eyebrow {
  font-family: 'Gaegu', cursive; font-weight: 700;
  font-size: 22px; color: var(--crayon-orange);
  margin: 0 0 10px; transform: rotate(-1.5deg); display: block;
}

/* ============================== STORYBOOK FIELDS ============================== */
.field label span, .doodle-input label span, .tiny-toggle label span {
  font-family: 'Gaegu', cursive !important; font-weight: 700 !important;
  font-size: 19px !important; color: var(--ink) !important;
}
.field textarea, .field input[type="text"], .field input:not([type]),
.field .wrap .secondary-wrap input, .field [data-testid="textbox"],
.field input[role="listbox"], .field .secondary-wrap, .field [class*="dropdown"] input {
  font-family: 'Nunito', sans-serif !important; font-size: 17px !important;
  color: var(--ink) !important; background: #fffdf6 !important;
  border: 2.5px solid var(--ink) !important; border-radius: 12px !important;
  padding: 11px 14px !important; box-shadow: 2px 3px 0 rgba(46,42,38,.10) !important;
}
.field textarea:focus, .field input:focus {
  border-color: var(--crayon-orange) !important;
  box-shadow: 2px 3px 0 rgba(239,106,58,.30) !important; outline: none !important;
}

.doodle-input { position: relative; }
.doodle-input .image-container, .doodle-input [data-testid="image"] {
  background: #ffffff !important; border: none !important; border-radius: 4px !important;
  padding: 12px 12px 34px !important;
  box-shadow: 0 8px 18px rgba(46,42,38,.18) !important; transform: rotate(-2deg);
}
.doodle-input img { border-radius: 2px !important; }
.doodle-input .upload-container, .doodle-input [data-testid="image"] .wrap {
  border: 3px dashed var(--crayon-sky) !important; border-radius: 8px !important;
  background: #f3f9ff !important; color: var(--ink-soft) !important;
}

/* Custom voice audio widget */
.custom-voice-field label span {
  font-family: 'Gaegu', cursive !important; font-weight: 700 !important;
  font-size: 19px !important; color: var(--crayon-teal) !important;
}
.custom-voice-field [data-testid="audio"], .custom-voice-field .audio {
  background: #f0faf8 !important;
  border: 2.5px solid var(--crayon-teal) !important;
  border-radius: 12px !important;
}

/* ============================== BUTTONS ============================== */
.btn-make, .btn-make button {
  font-family: 'Gaegu', cursive !important; font-weight: 700 !important;
  font-size: 26px !important; letter-spacing: .5px !important;
  color: #fff !important; background: var(--crayon-orange) !important;
  border: 2.5px solid var(--ink) !important; border-radius: 16px !important;
  padding: 14px 26px !important; width: 100% !important;
  transform: rotate(-1deg); box-shadow: 4px 5px 0 var(--ink) !important;
  transition: transform .12s ease, box-shadow .12s ease !important;
}
.btn-make:hover, .btn-make button:hover {
  transform: rotate(-1deg) translate(-2px,-2px);
  box-shadow: 6px 7px 0 var(--ink) !important; background: #f5764a !important;
}

.btn-pdf, .btn-pdf button {
  font-family: 'Gaegu', cursive !important; font-weight: 700 !important;
  font-size: 19px !important; color: var(--ink) !important;
  background: var(--crayon-sun) !important; border: 2.5px solid var(--ink) !important;
  border-radius: 14px !important; box-shadow: 3px 4px 0 var(--ink) !important;
  transform: rotate(0.8deg);
}
.btn-pdf:hover, .btn-pdf button:hover { background: #f8d066 !important; }

.download-row { margin-top: 8px !important; gap: 10px !important; }
.download-row > div { flex: 1 1 0 !important; }
.tiny-toggle { transform: rotate(-0.6deg); }

/* ============================== THEME/VOICE CHIPS ============================== */
.theme-pick .wrap, .theme-pick [role="radiogroup"] {
  display: flex !important; flex-wrap: wrap !important; gap: 8px !important;
}
.theme-pick label {
  background: #fffdf6 !important; border: 2.5px solid var(--ink) !important;
  border-radius: 14px !important; padding: 8px 14px !important; margin: 0 !important;
  cursor: pointer !important; font-family: 'Gaegu', cursive !important;
  font-size: 16px !important; color: var(--ink) !important;
  box-shadow: 2px 3px 0 rgba(46,42,38,.12) !important;
  transition: transform .1s ease, background .1s ease !important;
}
.theme-pick label:hover { transform: translateY(-1px); background: #fff3e0 !important; }
.theme-pick label:has(input:checked) {
  background: var(--crayon-orange) !important; color: #fff !important;
  box-shadow: 2px 3px 0 var(--ink) !important;
}
.theme-pick input[type="radio"] { accent-color: var(--crayon-orange); margin-right: 6px; }

/* Page-count slider */
.page-slider label span {
  font-family: 'Gaegu', cursive !important; font-weight: 700 !important;
  font-size: 19px !important; color: var(--ink) !important;
}
.page-slider input[type="range"] { accent-color: var(--crayon-teal) !important; }

/* ============================== STATUS ============================== */
.status-display textarea {
  font-family: 'Caveat', cursive !important; font-size: 20px !important;
  color: var(--crayon-teal) !important; background: #f0faf8 !important;
  border: 2.5px dashed var(--crayon-teal) !important; border-radius: 12px !important;
  text-align: center !important;
}
.status-display, .status-display .block, .status-display .form,
.status-display .container, .status-display label span {
  background: transparent !important; color: var(--crayon-teal) !important;
}

/* ============================== AUDIO ============================== */
.audio-player {
  background: #f0faf8 !important; border: 2.5px solid var(--ink) !important;
  border-radius: 14px !important; padding: 10px 12px !important;
  box-shadow: 3px 4px 0 rgba(46,42,38,.12) !important;
}
.audio-player label span { font-family: 'Gaegu', cursive !important; font-size: 18px !important; }

/* ============================== THE BOOK ============================== */
.book-stage { min-height: 220px; }
.book-container { max-width: 100%; padding: 4px; }
.book-title {
  font-family: 'Gaegu', cursive !important; font-weight: 700 !important;
  font-size: clamp(30px, 4.5vw, 46px) !important; text-align: center !important;
  color: var(--ink) !important; margin: 6px 0 22px !important;
  text-shadow: 2px 2px 0 var(--crayon-sun);
}
.book-cover {
  text-align: center; padding: 34px 22px 30px; margin: 10px auto 26px;
  background: radial-gradient(circle at 30% 20%, rgba(244,198,74,.25), transparent 55%), #fff8e6;
  border-radius: 18px; box-shadow: 0 12px 28px rgba(46,42,38,.16); transform: rotate(-1deg);
}
.book-cover .cover-kicker { font-family: 'Caveat', cursive; font-size: 22px; color: var(--crayon-berry); }
.book-cover .cover-title {
  font-family: 'Gaegu', cursive; font-weight: 700;
  font-size: clamp(34px, 5vw, 52px); color: var(--ink); margin: 4px 0 2px;
  text-shadow: 2px 2px 0 var(--crayon-sun);
}
.book-page {
  position: relative; background: #fffdf6; border-radius: 14px;
  padding: 22px 22px 26px; margin: 30px auto; max-width: 640px;
  box-shadow: 0 8px 20px rgba(46,42,38,.13);
}
.book-page:nth-child(even) { transform: rotate(0.9deg); }
.book-page:nth-child(odd)  { transform: rotate(-0.9deg); }
.book-page:hover { transform: rotate(0deg) translateY(-3px); transition: transform .3s ease; }
.book-page::before {
  content: ""; position: absolute; top: -11px; left: 50%;
  width: 96px; height: 24px; transform: translateX(-50%) rotate(-2.5deg);
  background: repeating-linear-gradient(45deg, rgba(255,255,255,.4) 0 5px, transparent 5px 10px), var(--tape);
  box-shadow: 0 2px 5px rgba(0,0,0,.12);
}
.book-page .page-art { position: relative; }
.book-page .page-art::after {
  content: ""; position: absolute; inset: 4px;
  border: 3px solid var(--ink); border-radius: 12px;
  filter: url(#wobble); pointer-events: none;
}
.book-page img { display: block; width: 100%; border-radius: 10px; }
.page-text {
  font-family: 'Caveat', cursive !important; font-size: clamp(22px, 3vw, 30px) !important;
  line-height: 1.45 !important; color: var(--ink) !important;
  text-align: center !important; margin: 18px 6px 4px !important;
}
.page-num {
  display: block; text-align: center; margin-top: 8px;
  font-family: 'Gaegu', cursive; font-weight: 700; font-size: 16px; color: var(--ink-soft);
}
.page-loading {
  text-align: center; padding: 40px 20px;
  font-family: 'Gaegu', cursive; font-weight: 700; font-size: 22px; color: var(--ink-soft);
  animation: bob 1.6s ease-in-out infinite;
}
.book-empty { text-align: center; padding: 54px 24px; font-family: 'Gaegu', cursive; color: var(--ink-soft); }
.book-empty .big { font-size: 30px; color: var(--ink); }
.book-empty .arrow { font-size: 42px; display:block; margin-bottom: 8px; animation: bob 1.6s ease-in-out infinite; }
@keyframes bob { 0%,100% { transform: translateY(0) } 50% { transform: translateY(-7px) } }
.engine-badge {
  display:inline-block; margin: 0 auto 4px; padding: 3px 12px;
  font-family:'Gaegu',cursive; font-weight:700; font-size:14px;
  border:2px solid var(--ink); border-radius: 20px; transform: rotate(-1.5deg);
}
.engine-badge.flux   { background: var(--crayon-leaf); color:#fff; }
.engine-badge.sketch { background: var(--crayon-sun); color: var(--ink); }

/* ============================== ACCORDION ============================== */
.behind-magic {
  position: relative; margin: 16px 8px 0 !important;
  background: #fff8ef !important; border: 2.5px dashed var(--crayon-berry) !important;
  border-radius: 16px !important; padding: 6px 14px !important; transform: rotate(-0.4deg);
}
.behind-magic span, .behind-magic button {
  font-family: 'Gaegu', cursive !important; font-weight: 700 !important;
  color: var(--crayon-berry) !important; font-size: 20px !important;
}

/* ============================== FOOTER ============================== */
.app-footer {
  text-align: center; padding: 26px 16px 36px; margin-top: 30px;
  font-family: 'Caveat', cursive; color: var(--ink-soft); font-size: 19px;
}
.app-footer .badges { font-family:'Gaegu',cursive; font-weight:700; color: var(--ink); }

/* ============================== LIGHT-MODE LOCK ============================== */
.gradio-container, .gradio-container *:not(svg):not(path) {
  --background-fill-primary: transparent;
  --background-fill-secondary: transparent;
  --border-color-primary: var(--ink);
  --body-text-color: var(--ink) !important;
  --body-text-color-subdued: var(--ink-soft) !important;
  --button-primary-background-fill: var(--crayon-orange);
  --button-primary-text-color: #fff;
  color-scheme: light;
}

/* ============================== RESPONSIVE ============================== */
@media (max-width: 820px) {
  body, gradio-app { background-attachment: scroll !important; }
  html, body { height: auto !important; overflow-y: auto !important; overflow-x: hidden !important; -webkit-overflow-scrolling: touch !important; }
  gradio-app { height: auto !important; min-height: 100vh !important; overflow-y: visible !important; overflow-x: hidden !important; }
  .gradio-container { max-width: 100vw !important; overflow-x: hidden !important; padding: 0 !important; }
  .gradio-container .gap { flex-wrap: wrap !important; }
  .input-card, .output-card {
    transform: none !important; flex: 1 1 100% !important; width: 100% !important;
    max-width: 100% !important; min-width: 0 !important; margin: 8px 0 !important;
    box-sizing: border-box !important;
  }
  .app-title { font-size: clamp(32px, 10vw, 52px) !important; }
  .book-page { max-width: 100%; margin: 20px auto; }
  .theme-pick label { font-size: 14px !important; padding: 6px 10px !important; }
  .feature-strip { gap: 6px 10px; }
  .feature-strip span { font-size: 14px !important; padding: 4px 10px !important; }
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
  .input-card, .output-card, .book-page, .book-cover, .btn-make, .btn-make button { transform: none !important; }
}
"""

COMBINED_JS = """
() => {
  // Lock to light mode first
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    window.location.replace(u.toString());
    return;
  }

  // Floating coloured sparkles
  const sparkColors = ['#ef6a3a','#f4c64a','#2ba39a','#4a9fd6','#d6517a','#74b85a'];
  const spWrap = document.createElement('div');
  spWrap.id = 'db-sparkles';
  spWrap.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;overflow:hidden;';
  for (let i = 0; i < 18; i++) {
    const s = document.createElement('span');
    s.className = 'db-spark';
    s.style.cssText = `left:${Math.random()*100}%;top:${10+Math.random()*80}%;`
      + `width:${5+Math.random()*6}px;height:${5+Math.random()*6}px;`
      + `background:${sparkColors[i % sparkColors.length]};`
      + `animation-delay:${Math.random()*9}s;animation-duration:${6+Math.random()*5}s;`;
    spWrap.appendChild(s);
  }
  document.body.prepend(spWrap);
}
"""


def create_layout(load_sample_fn=None, create_book_fn=None):
    """Build the single-tab DoodleBook Gradio Blocks layout."""

    _gr_major = int(gr.__version__.split(".")[0])
    design_kwargs = dict(
        css=CSS,
        head=HEAD,
        js=COMBINED_JS,
        theme=gr.themes.Base(),
    )
    blocks_kwargs = dict(title="DoodleBook")
    if _gr_major < 6:
        blocks_kwargs.update(design_kwargs)

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(SVG_DEFS)

        # ── HEADER ───────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
          <h1 class="app-title">D<span class="doodle-o">oo</span>dleBook</h1>
          <svg class="title-squiggle" viewBox="0 0 360 16" preserveAspectRatio="none">
            <path d="M2,11 C40,3 70,15 110,8 S190,2 230,9 320,14 358,5"/>
          </svg>
          <p class="app-subtitle">your child draws a hero &middot; AI writes the story &middot; hear it read aloud</p>
          <div class="feature-strip">
            <span>✏️ Draw your hero</span>
            <span>📖 AI writes 6–10 pages</span>
            <span>🎨 FLUX illustrates it</span>
            <span>🎙️ Narrated aloud</span>
            <span>🖨️ Print &amp; colour</span>
          </div>
        </div>
        """)

        # ── STORYBOOK ────────────────────────────────────────────────────────
        with gr.Row(equal_height=False):
            # INPUT CARD
            with gr.Column(scale=1, elem_classes=["input-card"]):
                gr.HTML('<p class="card-eyebrow">1 &middot; your character</p>')
                doodle = gr.Image(
                    sources=["upload", "webcam"],
                    label="Upload or snap the drawing",
                    type="numpy", height=240,
                    elem_classes=["doodle-input"],
                )
                gr.HTML('<p class="card-eyebrow">2 &middot; the details</p>')
                char_name = gr.Textbox(
                    label="Character name",
                    placeholder="Ziggy the robot",
                    elem_classes=["field"],
                )
                hero_name = gr.Textbox(
                    label="Hero's name in the story",
                    placeholder="Ziggy",
                    elem_classes=["field"],
                )
                theme = gr.Radio(
                    choices=THEMES, value=THEMES[0],
                    label="Story theme (pick one)",
                    elem_classes=["field", "theme-pick"],
                )
                story_idea = gr.Textbox(
                    label="Story spark (optional)",
                    placeholder="e.g. 'loves dinosaurs', 'wants to fly', 'learns to share'",
                    max_lines=2,
                    elem_classes=["field"],
                )
                num_pages = gr.Slider(
                    minimum=6, maximum=10, step=1, value=6,
                    label="Number of pages",
                    elem_classes=["field", "page-slider"],
                )
                voice = gr.Radio(
                    choices=VOICE_CHOICES, value=DEFAULT_VOICE,
                    label="Narrator voice",
                    elem_classes=["field", "theme-pick"],
                )
                custom_voice_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload your voice (5–60 s of clear speech)",
                    visible=False,
                    elem_classes=["custom-voice-field"],
                )
                make_coloring = gr.Checkbox(
                    label="Also make a coloring book",
                    value=False, elem_classes=["tiny-toggle"],
                )
                make_btn = gr.Button(
                    "Make my book!",
                    variant="primary",
                    elem_classes=["btn-make"],
                )
                status = gr.Textbox(
                    label="Status", interactive=False,
                    elem_classes=["status-display"],
                    value="Ready when you are! ✏️",
                )
                with gr.Row(elem_classes=["download-row"]):
                    pdf_download = gr.DownloadButton(
                        "⬇ Story PDF", visible=False, elem_classes=["btn-pdf"],
                    )
                    coloring_pdf_download = gr.DownloadButton(
                        "⬇ Coloring PDF", visible=False, elem_classes=["btn-pdf"],
                    )
                gr.Examples(
                    examples=[["assets/sample_doodle.jpg", "Ziggy", "Ziggy", "brave adventure"]],
                    inputs=[doodle, char_name, hero_name, theme],
                    label="Try an example",
                )

            # OUTPUT CARD
            with gr.Column(scale=2, elem_classes=["output-card"]):
                audio_narration = gr.Audio(
                    label="Listen to your story",
                    autoplay=False,
                    elem_classes=["audio-player"],
                )
                book_display = gr.HTML(
                    elem_classes=["book-stage"],
                    value="""
                    <div class="book-empty">
                      <span class="arrow">↑</span>
                      <p class="big">Your storybook appears here</p>
                      <p>Add a drawing, pick a theme, and tap <b>Make my book!</b></p>
                    </div>
                    """,
                )
                coloring_display = gr.HTML(visible=False, elem_classes=["book-stage"])

        # Behind the magic accordion
        with gr.Accordion("Behind the magic ✨", open=False, elem_classes=["behind-magic"]):
            with gr.Tabs():
                with gr.Tab("Story"):
                    story_info = gr.JSON(label="Generated story structure")
                with gr.Tab("Images"):
                    image_info = gr.Textbox(label="Illustration details", interactive=False, lines=5)
                with gr.Tab("Models"):
                    gr.Markdown(
                        """
| Model | Role | Size | Sponsor |
|---|---|---|---|
| **MiniCPM5-1B** | Story writer | 1B | OpenBMB |
| **VoxCPM2** | Voice narrator | 2B | OpenBMB |
| **FLUX.2-klein** | Illustrator | 4B | Black Forest Labs |

**Total: ~7B parameters.** The *brain* (story + voice) is a **3B small-model stack**. FLUX is the renderer. Tiny Titan.
                        """
                    )
                with gr.Tab("Trace"):
                    _tb_kwargs = dict(label="Generation trace", interactive=False, lines=8)
                    if _gr_major < 6:
                        _tb_kwargs["show_copy_button"] = True
                    trace_info = gr.Textbox(**_tb_kwargs)

        # ── FOOTER ───────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-footer">
          <p>stitched together with crayons &amp; code &middot; Build Small Hackathon 2026</p>
          <p class="badges">Off-Brand &middot; Open Trace &middot; Field Notes &middot; Tiny Titan &middot; Sponsor: OpenBMB &middot; Sponsor: Black Forest Labs</p>
        </div>
        """)

        # ── WIRING ───────────────────────────────────────────────────────────
        # Show/hide custom voice recorder when "My Voice" is selected
        voice.change(
            fn=lambda v: gr.update(visible=(v == "my_voice")),
            inputs=[voice],
            outputs=[custom_voice_audio],
        )

        if create_book_fn:
            make_btn.click(
                fn=create_book_fn,
                inputs=[doodle, char_name, theme, hero_name, voice, make_coloring,
                        num_pages, story_idea, custom_voice_audio],
                outputs=[book_display, status, audio_narration, pdf_download,
                         story_info, image_info, trace_info,
                         coloring_display, coloring_pdf_download],
            )
        if load_sample_fn:
            demo.load(fn=load_sample_fn, outputs=[book_display])

    demo.design_kwargs = design_kwargs if _gr_major >= 6 else {}
    return demo
