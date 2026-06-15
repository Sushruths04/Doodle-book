"""
UI layout — DoodleBook with two tabs: Storybook + Bedtime Voice.

Storybook tab: construction-paper scrapbook (unchanged aesthetic).
Bedtime Voice tab: moonlit night-sky palette, voice cloning, English + Kannada audio.
Both tabs share the same fonts, header, and footer.
"""

import gradio as gr
from config import VOICE_CHOICES, DEFAULT_VOICE, BEDTIME_GENRES, BEDTIME_MOODS

THEMES = [
    "brave adventure",
    "making a new friend",
    "overcoming a fear",
    "helping someone",
    "lost and found",
    "learning something new",
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
   DOODLEBOOK — CONSTRUCTION-PAPER SCRAPBOOK + BEDTIME VOICE
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
  /* Bedtime palette */
  --night:      #0e0e2e;
  --night-card: #14143a;
  --night-2:    #1a1a4a;
  --moon:       #c8a83c;
  --moon-glow:  #e8c84a;
  --starlight:  #ddd8ff;
  --star-soft:  #9090c0;
  --night-border: #4040a0;
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

.db-star {
  position: absolute;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.9);
  animation: db-twinkle ease-in-out infinite;
}
@keyframes db-twinkle {
  0%,100% { opacity: 0.04; transform: scale(0.7); }
  50%      { opacity: 0.95; transform: scale(1.35); }
}

/* ============================== TABS ============================== */
.gradio-container [role="tab"] {
  font-family: 'Gaegu', cursive !important;
  font-size: 20px !important;
  font-weight: 700 !important;
  border-radius: 14px 14px 0 0 !important;
  border: 2.5px solid var(--ink) !important;
  border-bottom: none !important;
  background: #ede0c8 !important;
  color: var(--ink-soft) !important;
  padding: 10px 24px !important;
  margin-right: 4px !important;
  transition: background .15s !important;
}
.gradio-container [role="tab"][aria-selected="true"] {
  background: #fffdf6 !important;
  color: var(--ink) !important;
}
.gradio-container [role="tab"]:hover:not([aria-selected="true"]) {
  background: #f5e8cf !important;
}

/* ============================== HEADER ============================== */
.app-header {
  text-align: center;
  padding: 34px 16px 10px;
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

/* ============================== BEDTIME VOICE CARDS ============================== */
.bedtime-input-card {
  position: relative;
  background: linear-gradient(160deg, var(--night-card) 0%, var(--night) 100%) !important;
  border-radius: 18px !important;
  padding: 30px 26px 26px !important;
  margin: 8px !important;
  box-shadow: 0 10px 26px rgba(0,0,0,.5), 0 0 50px rgba(80,80,180,.1) !important;
  transform: rotate(-0.5deg);
  color: var(--starlight) !important;
}
.bedtime-input-card:hover { transform: rotate(0deg); transition: transform .35s ease; }
.bedtime-output-card {
  position: relative;
  background: linear-gradient(160deg, var(--night-2) 0%, var(--night) 100%) !important;
  border-radius: 18px !important;
  padding: 30px 26px 26px !important;
  margin: 8px !important;
  box-shadow: 0 10px 26px rgba(0,0,0,.5) !important;
  transform: rotate(0.4deg);
}
.bedtime-output-card:hover { transform: rotate(0deg); transition: transform .35s ease; }

.bedtime-input-card .form, .bedtime-input-card .block,
.bedtime-input-card .panel, .bedtime-input-card .wrap,
.bedtime-input-card .gap, .bedtime-input-card .styler {
  background: transparent !important;
  border-color: rgba(64,64,160,.4) !important;
  box-shadow: none !important;
}
.bedtime-output-card .form, .bedtime-output-card .block,
.bedtime-output-card .panel, .bedtime-output-card .wrap,
.bedtime-output-card .gap, .bedtime-output-card .styler {
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}

.card-eyebrow-moon {
  font-family: 'Gaegu', cursive; font-weight: 700;
  font-size: 22px; color: var(--moon);
  margin: 0 0 10px; transform: rotate(-1.5deg); display: block;
}

.moon-orb {
  width: 64px; height: 64px;
  background: radial-gradient(circle at 38% 38%, #f5e87a, #c8a83c 60%, #8a6a10);
  border-radius: 50%;
  box-shadow: 0 0 30px rgba(200,168,60,.5), 0 0 60px rgba(200,168,60,.2);
  margin: 0 auto 16px;
  animation: moon-glow 4s ease-in-out infinite;
}
@keyframes moon-glow {
  0%,100% { box-shadow: 0 0 30px rgba(200,168,60,.4), 0 0 60px rgba(200,168,60,.2); }
  50%      { box-shadow: 0 0 50px rgba(200,168,60,.8), 0 0 90px rgba(200,168,60,.35); }
}

/* Night radio chips */
.theme-pick-night .wrap,
.theme-pick-night [role="radiogroup"] {
  display: flex !important; flex-wrap: wrap !important; gap: 8px !important;
}
.theme-pick-night label {
  background: rgba(80,80,180,.18) !important;
  border: 2px solid var(--night-border) !important;
  border-radius: 14px !important;
  padding: 8px 14px !important; margin: 0 !important; cursor: pointer !important;
  font-family: 'Gaegu', cursive !important; font-size: 16px !important;
  color: var(--starlight) !important;
  transition: transform .1s ease, background .1s ease !important;
}
.theme-pick-night label:hover { transform: translateY(-1px); background: rgba(100,100,220,.3) !important; }
.theme-pick-night label:has(input:checked) {
  background: var(--moon) !important;
  color: var(--night) !important;
  border-color: var(--moon-glow) !important;
  box-shadow: 0 0 8px rgba(200,168,60,.4) !important;
}
.theme-pick-night input[type="radio"] { accent-color: var(--moon); margin-right: 6px; }

/* Night text fields */
.field-night label span {
  font-family: 'Gaegu', cursive !important; font-weight: 700 !important;
  font-size: 19px !important; color: var(--moon) !important;
}
.field-night textarea,
.field-night input[type="text"],
.field-night input:not([type]) {
  font-family: 'Nunito', sans-serif !important; font-size: 17px !important;
  color: var(--starlight) !important;
  background: rgba(100,100,200,.15) !important;
  border: 2px solid var(--night-border) !important;
  border-radius: 12px !important;
  padding: 11px 14px !important;
}
.field-night textarea:focus, .field-night input:focus {
  border-color: var(--moon) !important;
  box-shadow: 0 0 8px rgba(200,168,60,.3) !important;
  outline: none !important;
}

/* Night audio recording input */
.bedtime-audio label span {
  font-family: 'Gaegu', cursive !important; font-weight: 700 !important;
  font-size: 19px !important; color: var(--moon) !important;
}
.bedtime-audio [data-testid="audio"],
.bedtime-audio .audio {
  background: rgba(60,60,160,.2) !important;
  border: 2px solid var(--night-border) !important;
  border-radius: 12px !important;
}
.bedtime-audio .upload-container,
.bedtime-audio .wrap { background: transparent !important; }

/* Bedtime button */
.btn-bedtime, .btn-bedtime button {
  font-family: 'Gaegu', cursive !important; font-weight: 700 !important;
  font-size: 24px !important; color: var(--night) !important;
  background: var(--moon) !important;
  border: 2.5px solid #8a6a10 !important; border-radius: 16px !important;
  padding: 13px 24px !important; width: 100% !important;
  transform: rotate(-0.8deg);
  box-shadow: 4px 5px 0 rgba(0,0,0,.5) !important;
  transition: transform .12s ease, box-shadow .12s ease !important;
}
.btn-bedtime:hover, .btn-bedtime button:hover {
  transform: rotate(-0.8deg) translate(-2px,-2px);
  box-shadow: 6px 7px 0 rgba(0,0,0,.5) !important;
  background: var(--moon-glow) !important;
}

/* Bedtime status */
.bedtime-status textarea {
  font-family: 'Caveat', cursive !important; font-size: 20px !important;
  color: var(--moon) !important;
  background: rgba(80,80,180,.12) !important;
  border: 2px dashed var(--night-border) !important;
  border-radius: 12px !important; text-align: center !important;
}
.bedtime-status, .bedtime-status .block, .bedtime-status .form,
.bedtime-status label span { background: transparent !important; color: var(--moon) !important; }

/* Night audio players */
.audio-player-night {
  background: rgba(60,60,160,.18) !important;
  border: 2px solid var(--night-border) !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
}
.audio-player-night label span {
  font-family: 'Gaegu', cursive !important; color: var(--moon) !important; font-size: 18px !important;
}

/* Bedtime story text display */
.bedtime-title {
  font-family: 'Gaegu', cursive; font-weight: 700;
  font-size: clamp(22px, 3.5vw, 36px); color: var(--moon);
  text-align: center; margin-bottom: 18px;
  text-shadow: 0 0 20px rgba(200,168,60,.3);
}
.bedtime-page {
  font-family: 'Caveat', cursive; font-size: clamp(18px, 2.5vw, 26px);
  line-height: 1.6; color: var(--starlight);
  text-align: center; padding: 14px 18px; margin: 10px 0;
  background: rgba(255,255,255,.05); border-radius: 12px;
  border-left: 3px solid rgba(200,168,60,.3);
}
.bedtime-empty {
  text-align: center; padding: 60px 24px;
  font-family: 'Gaegu', cursive; color: var(--star-soft);
}
.bedtime-empty .moon-icon { font-size: 48px; display: block; margin-bottom: 12px; animation: bob 2.2s ease-in-out infinite; }
.bedtime-empty .big { font-size: 26px; color: var(--starlight); margin-bottom: 6px; }

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
  .input-card, .output-card,
  .bedtime-input-card, .bedtime-output-card {
    transform: none !important; flex: 1 1 100% !important; width: 100% !important;
    max-width: 100% !important; min-width: 0 !important; margin: 8px 0 !important;
    box-sizing: border-box !important;
  }
  .app-title { font-size: clamp(32px, 10vw, 52px) !important; }
  .book-page { max-width: 100%; margin: 20px auto; }
  .theme-pick label, .theme-pick-night label { font-size: 14px !important; padding: 6px 10px !important; }
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
  .input-card, .output-card, .bedtime-input-card, .bedtime-output-card,
  .book-page, .book-cover, .btn-make, .btn-make button, .btn-bedtime, .btn-bedtime button,
  .moon-orb { transform: none !important; }
}
"""

# Combined JS: light-mode lock + kid-friendly floating animations
COMBINED_JS = """
() => {
  // Lock to light mode first
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    window.location.replace(u.toString());
    return;
  }

  // Floating coloured sparkles (kids love these)
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

  // Twinkling stars (subtle on paper, vivid on night-sky bedtime cards)
  const stWrap = document.createElement('div');
  stWrap.id = 'db-stars';
  stWrap.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;overflow:hidden;';
  for (let i = 0; i < 32; i++) {
    const s = document.createElement('span');
    s.className = 'db-star';
    s.style.cssText = `left:${Math.random()*100}%;top:${Math.random()*100}%;`
      + `width:${2+Math.random()*3}px;height:${2+Math.random()*3}px;`
      + `animation-delay:${Math.random()*6}s;animation-duration:${2+Math.random()*3}s;`;
    stWrap.appendChild(s);
  }
  document.body.prepend(stWrap);
}
"""


def create_layout(load_sample_fn=None, create_book_fn=None, create_bedtime_fn=None):
    """Build the DoodleBook Gradio Blocks layout with Storybook + Bedtime Voice tabs."""

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

        # ── HEADER (shared) ───────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
          <h1 class="app-title">D<span class="doodle-o">oo</span>dleBook</h1>
          <svg class="title-squiggle" viewBox="0 0 360 16" preserveAspectRatio="none">
            <path d="M2,11 C40,3 70,15 110,8 S190,2 230,9 320,14 358,5"/>
          </svg>
          <p class="app-subtitle">draw a character &middot; get a storybook &middot; hear it read aloud</p>
        </div>
        """)

        # ── TABS ─────────────────────────────────────────────────────────
        with gr.Tabs():

            # ================================================================
            # TAB 1 — STORYBOOK
            # ================================================================
            with gr.Tab("📖 Storybook"):
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
                        voice = gr.Radio(
                            choices=VOICE_CHOICES, value=DEFAULT_VOICE,
                            label="Narrator voice",
                            elem_classes=["field", "theme-pick"],
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

                # Behind the magic accordion (inside Storybook tab)
                with gr.Accordion("Behind the magic ✨", open=False, elem_classes=["behind-magic"]):
                    with gr.Tabs():
                        with gr.Tab("Story"):
                            story_info = gr.JSON(label="Generated story structure")
                        with gr.Tab("Images"):
                            image_info = gr.Textbox(label="Illustration details", interactive=False, lines=5)
                        with gr.Tab("Models"):
                            gr.Markdown(
                                """
| Model | Role | Size |
|---|---|---|
| **MiniCPM5-1B** | Story writer | 1B |
| **VoxCPM2** | Voice narrator | 2B |
| **FLUX.2-klein** | Illustrator | 4B |

The *brain* of DoodleBook — the story + the voice — is a **3B small-model stack**. FLUX is the printer. **Tiny Titan.**
                                """
                            )
                        with gr.Tab("Trace"):
                            _tb_kwargs = dict(label="Generation trace (Open Trace)", interactive=False, lines=8)
                            if _gr_major < 6:
                                _tb_kwargs["show_copy_button"] = True
                            trace_info = gr.Textbox(**_tb_kwargs)

            # ================================================================
            # TAB 2 — BEDTIME VOICE
            # ================================================================
            with gr.Tab("🌙 Bedtime Voice"):
                gr.HTML("""
                <div style="text-align:center;padding:20px 0 4px;">
                  <div class="moon-orb"></div>
                  <p style="font-family:'Caveat',cursive;font-size:22px;color:#9090c0;margin-top:4px;">
                    record your voice &middot; get a bedtime story narrated in it &middot; English + Kannada
                  </p>
                </div>
                """)

                with gr.Row(equal_height=False):
                    # BEDTIME INPUT CARD
                    with gr.Column(scale=1, elem_classes=["bedtime-input-card"]):
                        gr.HTML('<p class="card-eyebrow-moon">1 &middot; tonight\'s adventure</p>')
                        bedtime_genre = gr.Radio(
                            choices=BEDTIME_GENRES, value=BEDTIME_GENRES[0],
                            label="Genre",
                            elem_classes=["field-night", "theme-pick-night"],
                        )
                        gr.HTML('<p class="card-eyebrow-moon" style="margin-top:14px">2 &middot; the mood</p>')
                        bedtime_mood = gr.Radio(
                            choices=BEDTIME_MOODS, value=BEDTIME_MOODS[0],
                            label="Mood",
                            elem_classes=["field-night", "theme-pick-night"],
                        )
                        gr.HTML('<p class="card-eyebrow-moon" style="margin-top:14px">3 &middot; the hero</p>')
                        bedtime_hero = gr.Textbox(
                            label="Hero's name (optional)",
                            placeholder="Finn, Lily, little one…",
                            elem_classes=["field-night"],
                        )
                        gr.HTML('<p class="card-eyebrow-moon" style="margin-top:14px">4 &middot; your voice</p>')
                        bedtime_voice = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Record or upload 5–60 s of clear speech",
                            elem_classes=["bedtime-audio"],
                        )
                        bedtime_btn = gr.Button(
                            "Tuck in with a story 🌙",
                            variant="primary",
                            elem_classes=["btn-bedtime"],
                        )
                        bedtime_status = gr.Textbox(
                            label="Status", interactive=False,
                            elem_classes=["bedtime-status"],
                            value="Ready for bedtime… 🌙",
                        )

                    # BEDTIME OUTPUT CARD
                    with gr.Column(scale=2, elem_classes=["bedtime-output-card"]):
                        bedtime_en_audio = gr.Audio(
                            label="🌙 English narration",
                            autoplay=False,
                            elem_classes=["audio-player-night"],
                        )
                        bedtime_kn_audio = gr.Audio(
                            label="🌙 ಕನ್ನಡ ಕಥೆ  (Kannada story)",
                            autoplay=False,
                            elem_classes=["audio-player-night"],
                        )
                        bedtime_display = gr.HTML(
                            value="""
                            <div class="bedtime-empty">
                              <span class="moon-icon">🌙</span>
                              <p class="big">Your bedtime story appears here</p>
                              <p style="color:#9090c0">Record your voice, pick a genre, and tap <b>Tuck in with a story 🌙</b></p>
                            </div>
                            """,
                        )

        # ── FOOTER (shared) ───────────────────────────────────────────────
        gr.HTML("""
        <div class="app-footer">
          <p>stitched together with crayons &amp; code for the Build Small Hackathon 2026</p>
          <p class="badges">Well-Tuned &middot; Off-Brand &middot; Field Notes &middot; Open Trace</p>
        </div>
        """)

        # ── WIRING ───────────────────────────────────────────────────────
        if create_book_fn:
            make_btn.click(
                fn=create_book_fn,
                inputs=[doodle, char_name, theme, hero_name, voice, make_coloring],
                outputs=[book_display, status, audio_narration, pdf_download,
                         story_info, image_info, trace_info,
                         coloring_display, coloring_pdf_download],
            )
        if load_sample_fn:
            demo.load(fn=load_sample_fn, outputs=[book_display])
        if create_bedtime_fn:
            bedtime_btn.click(
                fn=create_bedtime_fn,
                inputs=[bedtime_voice, bedtime_hero, bedtime_genre, bedtime_mood],
                outputs=[bedtime_display, bedtime_status, bedtime_en_audio, bedtime_kn_audio],
            )

    demo.design_kwargs = design_kwargs if _gr_major >= 6 else {}
    return demo
