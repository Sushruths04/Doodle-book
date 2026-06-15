# DoodleDreams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new HF Space `build-small-hackathon/DoodleDreams` combining DoodleBook's draw-a-character → FLUX illustration pipeline with MoonlitVoice's bedtime story + voice-cloning pipeline.

**Architecture:** User uploads a doodle + optional voice recording → MiniCPM5-1B writes a bedtime story → FLUX.2-klein illustrates 6 pages keeping the doodle character consistent → VoxCPM2 (English) or IndicF5 (Kannada) narrates in the user's cloned voice → illustrated storybook HTML/PDF + audio. All models load on CUDA at module scope (ZeroGPU pattern).

**Tech Stack:** Python 3.11 · Gradio 6.18 · spaces (ZeroGPU) · FLUX.2-klein-4B · MiniCPM5-1B · VoxCPM2 2.0.3 · IndicF5 · indictrans2-en-indic-1B · diffusers · transformers · librosa · soundfile · voxcpm 2.0.3

---

## File Map

New project root: `D:\Project\Hugging_face_app\doodledreams\`

| File | Origin | Action |
|---|---|---|
| `config.py` | New | All model IDs, story themes, constants |
| `story.py` | moonlit-voice | Copy + add BEDTIME_GENRES from DoodleDreams context |
| `audio_utils.py` | moonlit-voice | Direct copy, no changes |
| `tts.py` | moonlit-voice | Direct copy, no changes |
| `indic_text.py` | moonlit-voice | Direct copy, no changes |
| `indic_tts.py` | moonlit-voice | Direct copy, no changes |
| `book_builder.py` | doodlebook | Direct copy, no changes |
| `ui/__init__.py` | New | Empty |
| `ui/layout.py` | New | DoodleBook scrapbook + moonlit-night design |
| `app.py` | New | Combined ZeroGPU orchestrator |
| `requirements.txt` | New | Merged deps from both apps |
| `README.md` | New | HF Space metadata + description |
| `assets/sample_doodle.jpg` | doodlebook | Copy |

---

### Task 1: Scaffold the project

**Files:** Create `D:\Project\Hugging_face_app\doodledreams\` and populate it.

- [ ] **Step 1: Create directories**

```powershell
New-Item -ItemType Directory -Force "D:\Project\Hugging_face_app\doodledreams\ui"
New-Item -ItemType Directory -Force "D:\Project\Hugging_face_app\doodledreams\assets"
```

- [ ] **Step 2: Clone moonlit-voice to a temp location**

```bash
git clone https://github.com/Sushruths04/moonlit-voice.git C:/Temp/moonlit-voice-src
```

- [ ] **Step 3: Copy shared source files**

```powershell
$src_mv = "C:\Temp\moonlit-voice-src"
$src_db = "D:\Project\Hugging_face_app\doodlebook"
$dst    = "D:\Project\Hugging_face_app\doodledreams"

# From moonlit-voice
"audio_utils.py","story.py","tts.py","indic_text.py","indic_tts.py" | ForEach-Object {
    Copy-Item "$src_mv\$_" "$dst\"
}
# From doodlebook
"book_builder.py" | ForEach-Object {
    Copy-Item "$src_db\$_" "$dst\"
}
Copy-Item "$src_db\assets\sample_doodle.jpg" "$dst\assets\"
```

- [ ] **Step 4: Create empty ui/__init__.py**

```powershell
New-Item -ItemType File "D:\Project\Hugging_face_app\doodledreams\ui\__init__.py"
```

- [ ] **Step 5: Git init + first commit**

```bash
cd D:/Project/Hugging_face_app/doodledreams
git init
git add .
git commit -m "chore: scaffold DoodleDreams from doodlebook + moonlit-voice"
```

---

### Task 2: config.py — all constants

**Files:**
- Create: `D:\Project\Hugging_face_app\doodledreams\config.py`

- [ ] **Step 1: Write config.py**

```python
from dataclasses import dataclass

@dataclass
class ModelSpec:
    hub_id: str

# Models
FLUX_MODEL        = ModelSpec("black-forest-labs/FLUX.2-klein-4B")
STORY_MODEL       = ModelSpec("openbmb/MiniCPM5-1B")
TTS_MODEL         = ModelSpec("openbmb/VoxCPM2")
TRANSLATION_MODEL = ModelSpec("ai4bharat/indictrans2-en-indic-1B")
KANNADA_TTS_MODEL = ModelSpec("ai4bharat/IndicF5")
KANNADA_FINETUNE  = "mitvho09/IndicF5-Kannada-Bedtime-v2"

BASE_SEED = 42

# Story options
GENRES = ["Animals", "Kingdom", "Space", "Dragons", "Ocean", "Forest"]
MOODS  = ["Magical", "Calming", "Dreamy", "Funny"]
LANGUAGES = ["English", "Kannada"]

STORY_LENGTHS = {
    "Short (~1 min)":  120,
    "Medium (~2 min)": 220,
    "Long (~3 min)":   320,
}

# FLUX generation settings (fast: 6 steps, cfg 1.0)
FLUX_STEPS    = 6
FLUX_GUIDANCE = 1.0
FLUX_SIZE     = 768

COLOR_ART_STYLE = (
    "hand-drawn crayon children's storybook illustration, soft waxy crayon "
    "texture, warm colorful shading, simple friendly shapes, looks drawn by hand"
)
COLOR_PAGE_SUFFIX = "full colorful background scene, character clearly visible."
```

- [ ] **Step 2: Verify import**

```bash
cd D:/Project/Hugging_face_app/doodledreams
python -c "from config import FLUX_MODEL, STORY_MODEL, TTS_MODEL; print('config OK')"
```

Expected: `config OK`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add config.py with all model IDs and constants"
```

---

### Task 3: Verify ported modules compile

**Files:** story.py, audio_utils.py, tts.py, indic_text.py, indic_tts.py (already copied)

- [ ] **Step 1: Smoke-import all ported modules (no GPU needed)**

```bash
python -c "
import importlib, sys
mods = ['story','audio_utils','tts','indic_text','indic_tts','book_builder']
for m in mods:
    try:
        importlib.import_module(m)
        print(f'  OK  {m}')
    except ImportError as e:
        print(f'  MISS {m}: {e}')
    except Exception as e:
        print(f'  ERR  {m}: {e}')
"
```

Expected: `OK` for all modules (ImportError for heavy deps like torch is fine at this stage — look for syntax errors or missing local files).

- [ ] **Step 2: Fix any local import issues**

`story.py` imports from `config` — it uses its own `MODEL_ID` constant. Open `story.py` and replace any hard-coded model ID at the top with an import from our config:

```python
# In story.py, find the model ID line (near top) and replace with:
from config import STORY_MODEL
_MODEL_ID = STORY_MODEL.hub_id
```

Then in `_get_model()` in story.py, ensure it references `_MODEL_ID`.

`tts.py` similarly has a hard-coded `"openbmb/VoxCPM2"` — replace:

```python
from config import TTS_MODEL
_TTS_HUB_ID = TTS_MODEL.hub_id
```

`indic_text.py` hard-codes `"ai4bharat/indictrans2-en-indic-1B"` — replace:

```python
from config import TRANSLATION_MODEL
_TRANS_HUB_ID = TRANSLATION_MODEL.hub_id
```

`indic_tts.py` hard-codes both `"ai4bharat/IndicF5"` and `"mitvho09/IndicF5-Kannada-Bedtime-v2"` — replace:

```python
from config import KANNADA_TTS_MODEL, KANNADA_FINETUNE
_BASE_HUB_ID = KANNADA_TTS_MODEL.hub_id
_CHECKPOINT  = KANNADA_FINETUNE
```

- [ ] **Step 3: Commit**

```bash
git add story.py tts.py indic_text.py indic_tts.py
git commit -m "fix: wire ported modules to shared config"
```

---

### Task 4: requirements.txt + README.md

**Files:**
- Create: `D:\Project\Hugging_face_app\doodledreams\requirements.txt`
- Create: `D:\Project\Hugging_face_app\doodledreams\README.md`

- [ ] **Step 1: Write requirements.txt**

```
gradio==6.18.0
voxcpm==2.0.3
diffusers>=0.31.0
transformers>=4.46.0
accelerate>=0.25.0
huggingface_hub>=0.20.0
soundfile>=0.12.0
librosa>=0.10.0
numpy<2.0.0
sentencepiece>=0.1.99
Pillow>=10.0.0
fpdf2>=2.7.0
IndicTransToolkit @ git+https://github.com/VarunGumma/IndicTransToolkit.git
# torch installed separately via CUDA index URL in Dockerfile / HF Space
```

- [ ] **Step 2: Write README.md (HF Space metadata + description)**

```markdown
---
title: DoodleDreams
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: "6.18.0"
app_file: app.py
pinned: false
tags:
  - hackathon
  - build-small
  - adventure-in-thousand-token-wood
  - gradio
  - flux
  - minicpm
  - voxcpm
  - storybook
  - voice-cloning
  - kannada
models:
  - black-forest-labs/FLUX.2-klein-4B
  - openbmb/MiniCPM5-1B
  - openbmb/VoxCPM2
  - ai4bharat/IndicF5
  - ai4bharat/indictrans2-en-indic-1B
---

# DoodleDreams 🌙🖍️

**Draw your character · Record your voice · Get a narrated bedtime storybook**

DoodleDreams merges two ideas: DoodleBook's draw-a-character illustration
pipeline and MoonlitVoice's bedtime story + voice cloning. The result: a
child's crayon drawing becomes a six-page illustrated bedtime story, narrated
in a parent's own cloned voice. English and Kannada supported.

Built for the **Build Small Hackathon 2026 · Adventure in Thousand Token Wood**.
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt README.md
git commit -m "feat: add requirements.txt and README with HF Space metadata"
```

---

### Task 5: ui/layout.py — DoodleDreams design

**Files:**
- Create: `D:\Project\Hugging_face_app\doodledreams\ui\layout.py`

The design inherits DoodleBook's construction-paper scrapbook DNA and adds a
moonlit-night layer: a deep indigo sky gradient at the very top fading into warm
construction paper, CSS twinkling stars, a moon glyph near the title, and gold
accent colour replacing DoodleBook's crayon-orange for interactive elements.
Fonts stay the same (Gaegu + Caveat + Nunito). Animations are kid-friendly:
stars twinkle, the moon pulses softly.

- [ ] **Step 1: Write ui/layout.py**

```python
"""
DoodleDreams UI — "Moonlit Scrapbook"
DoodleBook's construction-paper scrapbook + a night-sky layer.
"""
import gradio as gr
from config import GENRES, MOODS, LANGUAGES, STORY_LENGTHS

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
</svg>
"""

CSS = r"""
/* ============================================================
   DOODLEDREAMS — MOONLIT SCRAPBOOK
   ============================================================ */
:root {
  --paper:       #f6ecd4;
  --paper-2:     #efe0c2;
  --ink:         #2e2a26;
  --ink-soft:    #6b5d4f;
  --moon-gold:   #f4c64a;
  --moon-glow:   rgba(244,198,74,0.25);
  --night-deep:  #0d1040;
  --night-mid:   #1a2060;
  --crayon-teal: #2ba39a;
  --crayon-berry:#d6517a;
  --crayon-sky:  #4a9fd6;
  --crayon-leaf: #74b85a;
  --tape:        rgba(244,198,74,0.55);
  --star-color:  rgba(255,248,210,0.9);
}

/* Neutralize Gradio theme */
.gradio-container,
.gradio-container *:not(svg):not(path) {
  --block-background-fill: transparent;
  --block-border-width:    0px;
  --block-shadow:          none;
  --panel-background-fill: transparent;
  --input-background-fill: #fffdf6;
  --body-text-color:       var(--ink);
}

.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  background: transparent !important;
  font-family: 'Nunito', sans-serif !important;
  color: var(--ink);
}

/* Night-sky → paper gradient background */
body, gradio-app {
  background:
    radial-gradient(ellipse at 18% 8%, rgba(244,198,74,.12), transparent 36%),
    radial-gradient(ellipse at 80% 6%, rgba(74,159,214,.10), transparent 32%),
    linear-gradient(180deg,
      var(--night-deep) 0%,
      var(--night-mid)  14%,
      #3b3060           26%,
      #6b5d4f           42%,
      var(--paper)      58%,
      var(--paper)      100%) !important;
  background-attachment: fixed !important;
  min-height: 100vh;
}

/* Twinkling CSS stars */
.star-layer { position:fixed; top:0; left:0; width:100%; height:38vh; pointer-events:none; z-index:0; overflow:hidden; }
.star { position:absolute; background:var(--star-color); border-radius:50%; animation: dd-twinkle var(--dur,2s) ease-in-out infinite alternate; }
@keyframes dd-twinkle { from { opacity:.15; transform:scale(1); } to { opacity:1; transform:scale(1.4); } }

/* ============================== HEADER ============================== */
.app-header {
  text-align: center;
  padding: 44px 16px 12px;
  position: relative;
  z-index: 1;
}
.moon-orb {
  display: inline-block;
  width: 64px; height: 64px;
  background: radial-gradient(circle at 38% 38%, #fffbe6, var(--moon-gold) 60%, #c9a020);
  border-radius: 50%;
  box-shadow: 0 0 28px 10px var(--moon-glow), 0 0 6px 2px rgba(244,198,74,.5);
  margin-bottom: 10px;
  animation: dd-moon-glow 4s ease-in-out infinite alternate;
}
@keyframes dd-moon-glow {
  from { box-shadow: 0 0 22px 8px var(--moon-glow); }
  to   { box-shadow: 0 0 44px 18px var(--moon-glow), 0 0 8px 3px rgba(244,198,74,.6); }
}
.app-title {
  font-family: 'Gaegu', cursive !important;
  font-weight: 700 !important;
  font-size: clamp(44px, 7vw, 78px) !important;
  line-height: .95 !important;
  color: #fff !important;
  margin: 0 !important;
  letter-spacing: 1px;
  text-shadow: 2px 2px 0 var(--moon-gold), 4px 4px 0 rgba(244,198,74,.35);
}
.app-title .dream-o { color: var(--moon-gold); display:inline-block; transform:rotate(-6deg); }
.app-subtitle {
  font-family: 'Caveat', cursive !important;
  font-size: clamp(18px, 3vw, 26px) !important;
  color: rgba(255,248,210,.85) !important;
  margin-top: 8px !important;
}
.title-squiggle { display:block; margin:6px auto 0; width:min(360px,70%); height:16px; }
.title-squiggle path { stroke:var(--moon-gold); stroke-width:5; fill:none; stroke-linecap:round; filter:url(#wobble); }

/* ============================== CARDS ============================== */
.input-card, .output-card {
  position: relative;
  background: #fffdf6 !important;
  border-radius: 18px;
  padding: 30px 26px 26px;
  margin: 8px;
  box-shadow: 0 10px 26px rgba(13,16,64,.22), 0 2px 0 rgba(0,0,0,.04);
  z-index: 1;
}
.input-card  { transform: rotate(-0.7deg); }
.output-card { transform: rotate(0.5deg); }
.input-card:hover, .output-card:hover { transform: rotate(0deg); transition: transform .35s ease; }

.card-eyebrow {
  font-family: 'Gaegu', cursive; font-weight: 700;
  font-size: 22px; color: var(--moon-gold); filter: brightness(.85);
  margin: 0 0 10px; transform: rotate(-1.5deg);
}

/* ============================== FIELDS ============================== */
.field label span, .doodle-input label span,
.voice-input label span, .tiny-toggle label span {
  font-family: 'Gaegu', cursive !important;
  font-weight: 700 !important;
  font-size: 19px !important;
  color: var(--ink) !important;
}
.field textarea, .field input[type="text"],
.field input:not([type]), .field .secondary-wrap,
.field [class*="dropdown"] input {
  font-family: 'Nunito', sans-serif !important;
  font-size: 17px !important;
  color: var(--ink) !important;
  background: #fffdf6 !important;
  border: 2.5px solid var(--ink) !important;
  border-radius: 12px !important;
  padding: 11px 14px !important;
  box-shadow: 2px 3px 0 rgba(46,42,38,.10) !important;
}
.field textarea:focus, .field input:focus {
  border-color: var(--moon-gold) !important;
  box-shadow: 2px 3px 0 rgba(244,198,74,.40) !important;
  outline: none !important;
}

/* Doodle upload */
.doodle-input .image-container,
.doodle-input [data-testid="image"] {
  background: #ffffff !important;
  border: none !important;
  border-radius: 4px !important;
  padding: 12px 12px 34px !important;
  box-shadow: 0 8px 18px rgba(46,42,38,.18) !important;
  transform: rotate(-2deg);
}
.doodle-input img { border-radius: 2px !important; }
.doodle-input .upload-container,
.doodle-input [data-testid="image"] .wrap {
  border: 3px dashed var(--crayon-sky) !important;
  border-radius: 8px !important;
  background: #f3f9ff !important;
}

/* Voice recorder card */
.voice-input {
  border: 2.5px dashed var(--moon-gold) !important;
  border-radius: 14px !important;
  padding: 12px !important;
  background: rgba(244,198,74,.06) !important;
}

/* Genre / Mood chips */
.chip-pick .wrap,
.chip-pick [role="radiogroup"] {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
}
.chip-pick label {
  background: #fffdf6 !important;
  border: 2.5px solid var(--ink) !important;
  border-radius: 14px !important;
  padding: 7px 13px !important;
  margin: 0 !important;
  cursor: pointer !important;
  font-family: 'Gaegu', cursive !important;
  font-size: 15px !important;
  color: var(--ink) !important;
  box-shadow: 2px 3px 0 rgba(46,42,38,.12) !important;
  transition: transform .1s ease, background .1s ease !important;
}
.chip-pick label:hover { transform:translateY(-1px); background:#fff8d0 !important; }
.chip-pick label:has(input:checked) {
  background: var(--moon-gold) !important;
  color: var(--ink) !important;
  box-shadow: 2px 3px 0 var(--ink) !important;
}
.chip-pick input[type="radio"] { accent-color: var(--moon-gold); margin-right:5px; }

/* ============================== BUTTON ============================== */
.btn-make, .btn-make button {
  font-family: 'Gaegu', cursive !important;
  font-weight: 700 !important;
  font-size: 26px !important;
  color: var(--ink) !important;
  background: var(--moon-gold) !important;
  border: 2.5px solid var(--ink) !important;
  border-radius: 16px !important;
  padding: 14px 26px !important;
  width: 100% !important;
  transform: rotate(-1deg);
  box-shadow: 4px 5px 0 var(--ink) !important;
  transition: transform .12s ease, box-shadow .12s ease !important;
}
.btn-make:hover, .btn-make button:hover {
  transform: rotate(-1deg) translate(-2px,-2px);
  box-shadow: 6px 7px 0 var(--ink) !important;
}
.btn-make:active, .btn-make button:active {
  transform: rotate(-1deg) translate(2px,2px);
  box-shadow: 1px 2px 0 var(--ink) !important;
}

.btn-pdf, .btn-pdf button {
  font-family: 'Gaegu', cursive !important;
  font-weight: 700 !important;
  font-size: 19px !important;
  color: var(--ink) !important;
  background: var(--crayon-teal) !important;
  color: #fff !important;
  border: 2.5px solid var(--ink) !important;
  border-radius: 14px !important;
  box-shadow: 3px 4px 0 var(--ink) !important;
}
.download-row { margin-top:8px !important; gap:10px !important; }
.download-row > div { flex:1 1 0 !important; }

/* ============================== STATUS ============================== */
.status-display textarea {
  font-family: 'Caveat', cursive !important;
  font-size: 20px !important;
  color: var(--night-mid) !important;
  background: #f0f4ff !important;
  border: 2.5px dashed var(--night-mid) !important;
  border-radius: 12px !important;
  text-align: center !important;
}

/* ============================== AUDIO ============================== */
.audio-player {
  position: relative;
  background: rgba(13,16,64,.06) !important;
  border: 2.5px solid var(--night-mid) !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
  box-shadow: 3px 4px 0 rgba(46,42,38,.12) !important;
}
.audio-player label span { font-family:'Gaegu',cursive !important; font-size:18px !important; }

/* ============================== BOOK ============================== */
.book-stage { min-height:220px; }
.book-container { max-width:100%; padding:4px; }
.book-title {
  font-family:'Gaegu',cursive !important;
  font-weight:700 !important;
  font-size: clamp(30px,4.5vw,46px) !important;
  text-align:center !important;
  color: var(--ink) !important;
  margin:6px 0 22px !important;
  text-shadow: 2px 2px 0 var(--moon-gold);
}
.book-page {
  position:relative;
  background:#fffdf6;
  border-radius:14px;
  padding:22px 22px 26px;
  margin:30px auto;
  max-width:640px;
  box-shadow:0 8px 20px rgba(46,42,38,.13);
}
.book-page:nth-child(even) { transform:rotate(.9deg); }
.book-page:nth-child(odd)  { transform:rotate(-.9deg); }
.book-page:hover { transform:rotate(0deg) translateY(-3px); transition:transform .3s ease; }
.book-page::before {
  content:"";
  position:absolute; top:-11px; left:50%;
  width:96px; height:24px;
  transform:translateX(-50%) rotate(-2.5deg);
  background: repeating-linear-gradient(45deg,rgba(255,255,255,.4) 0 5px,transparent 5px 10px), var(--tape);
  box-shadow:0 2px 5px rgba(0,0,0,.12);
}
.book-page img { display:block; width:100%; border-radius:10px; }
.page-text {
  font-family:'Caveat',cursive !important;
  font-size: clamp(22px,3vw,30px) !important;
  line-height:1.45 !important;
  color: var(--ink) !important;
  text-align:center !important;
  margin:18px 6px 4px !important;
}
.page-num {
  display:block; text-align:center; margin-top:8px;
  font-family:'Gaegu',cursive; font-weight:700;
  font-size:16px; color: var(--ink-soft);
}
.book-empty {
  text-align:center; padding:54px 24px;
  font-family:'Gaegu',cursive; color: var(--ink-soft);
}
.book-empty .big  { font-size:30px; color: var(--ink); }
.book-empty .arrow { font-size:42px; display:block; margin-bottom:8px; animation:bob 1.6s ease-in-out infinite; }
@keyframes bob { 0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)} }
.page-loading {
  text-align:center; padding:40px 20px;
  font-family:'Gaegu',cursive; font-weight:700;
  font-size:22px; color: var(--night-mid);
  animation: bob 1.6s ease-in-out infinite;
}

/* ============================== ACCORDION ============================== */
.behind-magic {
  position:relative;
  margin:36px 8px 0 !important;
  background:#f0f4ff !important;
  border:2.5px dashed var(--night-mid) !important;
  border-radius:16px !important;
  padding:6px 14px !important;
  transform:rotate(-.4deg);
}
.behind-magic span, .behind-magic button {
  font-family:'Gaegu',cursive !important;
  font-weight:700 !important;
  color: var(--night-mid) !important;
  font-size:20px !important;
}

/* ============================== FOOTER ============================== */
.app-footer {
  text-align:center; padding:26px 16px 36px; margin-top:30px;
  font-family:'Caveat',cursive; color:rgba(255,248,210,.7); font-size:19px; z-index:1; position:relative;
}
.app-footer .badges { font-family:'Gaegu',cursive; font-weight:700; color:var(--moon-gold); }

/* ============================== LIGHT-MODE LOCK ============================== */
.gradio-container, .gradio-container *:not(svg):not(path) {
  --background-fill-primary:   transparent;
  --background-fill-secondary: transparent;
  --body-text-color:           var(--ink) !important;
  --body-text-color-subdued:   var(--ink-soft) !important;
  color-scheme: light;
}
.input-card .form, .input-card .block, .input-card .panel,
.input-card .wrap, .input-card .gap, .input-card .styler,
.output-card .form, .output-card .block, .output-card .panel,
.output-card .gap, .output-card .styler {
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}
.status-display, .status-display .block, .status-display .form,
.status-display .container, .status-display label span {
  background: transparent !important;
}

/* ============================== RESPONSIVE ============================== */
@media (max-width: 820px) {
  body, gradio-app { background-attachment: scroll !important; }
  html, body { height:auto !important; overflow-y:auto !important; overflow-x:hidden !important; -webkit-overflow-scrolling:touch !important; }
  gradio-app { height:auto !important; min-height:100vh !important; overflow-y:visible !important; overflow-x:hidden !important; }
  .gradio-container { max-width:100vw !important; overflow-x:hidden !important; padding:0 !important; }
  .gradio-container .gap { flex-wrap:wrap !important; }
  .input-card, .output-card { transform:none !important; flex:1 1 100% !important; width:100% !important; max-width:100% !important; min-width:0 !important; margin:8px 0 !important; box-sizing:border-box !important; }
  .app-title { font-size:clamp(32px,10vw,52px) !important; }
  .book-page { max-width:100%; margin:20px auto; }
  .chip-pick label { font-size:13px !important; padding:5px 9px !important; }
}

/* ============================== ACCESSIBILITY ============================== */
@media (prefers-reduced-motion: reduce) {
  * { animation:none !important; transition:none !important; }
  .star { animation:none !important; }
}
"""

FORCE_LIGHT_JS = """
() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    window.location.replace(u.toString());
  }
}
"""

# JS: inject CSS twinkling stars into the DOM (30 random dots in the top 38vh)
STARS_JS = """
() => {
  const layer = document.createElement('div');
  layer.className = 'star-layer';
  for (let i = 0; i < 30; i++) {
    const s = document.createElement('div');
    s.className = 'star';
    const size = (Math.random() * 3 + 1).toFixed(1) + 'px';
    s.style.cssText = `width:${size};height:${size};left:${(Math.random()*98).toFixed(1)}%;top:${(Math.random()*95).toFixed(1)}%;--dur:${(Math.random()*3+1.5).toFixed(1)}s;animation-delay:${(Math.random()*4).toFixed(1)}s`;
    layer.appendChild(s);
  }
  document.body.prepend(layer);
}
"""


def create_layout(create_book_fn=None):
    _gr_major = int(gr.__version__.split(".")[0])
    design_kwargs = dict(
        css=CSS,
        head=HEAD,
        js=FORCE_LIGHT_JS,
        theme=gr.themes.Base(),
    )
    blocks_kwargs = dict(title="DoodleDreams")
    if _gr_major < 6:
        blocks_kwargs.update(design_kwargs)

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(SVG_DEFS)

        # Inject stars after page load
        demo.load(fn=None, js=STARS_JS)

        # ---- HEADER ----
        gr.HTML("""
        <div class="app-header">
          <div class="moon-orb"></div>
          <h1 class="app-title">D<span class="dream-o">oo</span>dleDreams</h1>
          <svg class="title-squiggle" viewBox="0 0 360 16" preserveAspectRatio="none">
            <path d="M2,11 C40,3 70,15 110,8 S190,2 230,9 320,14 358,5"/>
          </svg>
          <p class="app-subtitle">draw your character &middot; record your voice &middot; hear your bedtime story</p>
        </div>
        """)

        with gr.Row(equal_height=False):
            # ---- INPUT CARD ----
            with gr.Column(scale=1, elem_classes=["input-card"]):
                gr.HTML('<p class="card-eyebrow">1 &middot; your character</p>')
                doodle = gr.Image(
                    sources=["upload", "webcam"],
                    label="Upload or snap the drawing",
                    type="numpy",
                    height=240,
                    elem_classes=["doodle-input"],
                )
                gr.HTML('<p class="card-eyebrow">2 &middot; your voice (optional)</p>')
                ref_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload your voice (5–60 s)",
                    elem_classes=["voice-input"],
                )
                gr.HTML('<p class="card-eyebrow">3 &middot; the story</p>')
                hero_name = gr.Textbox(
                    label="Hero's name",
                    placeholder="Luna, Ziggy, Aarav…",
                    elem_classes=["field"],
                )
                genre = gr.Radio(
                    choices=GENRES,
                    value=GENRES[0],
                    label="Story world",
                    elem_classes=["field", "chip-pick"],
                )
                mood = gr.Radio(
                    choices=MOODS,
                    value=MOODS[1],
                    label="Story mood",
                    elem_classes=["field", "chip-pick"],
                )
                language = gr.Radio(
                    choices=LANGUAGES,
                    value=LANGUAGES[0],
                    label="Language",
                    elem_classes=["field", "chip-pick"],
                )
                length = gr.Radio(
                    choices=list(STORY_LENGTHS.keys()),
                    value=list(STORY_LENGTHS.keys())[0],
                    label="Story length",
                    elem_classes=["field", "chip-pick"],
                )
                make_btn = gr.Button(
                    "🌙 Make my bedtime book!",
                    variant="primary",
                    elem_classes=["btn-make"],
                )
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["status-display"],
                    value="Ready to dream! 🌙",
                )
                with gr.Row(elem_classes=["download-row"]):
                    pdf_download = gr.DownloadButton(
                        "⬇ Story PDF",
                        visible=False,
                        elem_classes=["btn-pdf"],
                    )

            # ---- OUTPUT CARD ----
            with gr.Column(scale=2, elem_classes=["output-card"]):
                audio_narration = gr.Audio(
                    label="🔊 Your bedtime story",
                    autoplay=False,
                    elem_classes=["audio-player"],
                )
                book_display = gr.HTML(
                    elem_classes=["book-stage"],
                    value="""
                    <div class="book-empty">
                      <span class="arrow">🌙</span>
                      <p class="big">Your storybook appears here</p>
                      <p>Draw a character, pick a theme, and tap <b>Make my bedtime book!</b></p>
                    </div>
                    """,
                )

        # ---- BEHIND THE MAGIC ----
        with gr.Accordion("Behind the magic ✨", open=False, elem_classes=["behind-magic"]):
            with gr.Tabs():
                with gr.Tab("Story"):
                    story_info = gr.JSON(label="Generated story structure")
                with gr.Tab("Trace"):
                    _tb_kwargs = dict(label="Generation trace", interactive=False, lines=8)
                    if _gr_major < 6:
                        _tb_kwargs["show_copy_button"] = True
                    trace_info = gr.Textbox(**_tb_kwargs)

        gr.HTML("""
        <div class="app-footer">
          <p>sewn together with moonlight &amp; crayons for the Build Small Hackathon 2026</p>
          <p class="badges">Tiny Titan &middot; Off-Brand &middot; Open Trace &middot; Field Notes</p>
        </div>
        """)

        # ---- WIRING ----
        if create_book_fn:
            make_btn.click(
                fn=create_book_fn,
                inputs=[doodle, ref_audio, hero_name, genre, mood, language, length],
                outputs=[book_display, status, audio_narration, pdf_download,
                         story_info, trace_info],
            )

    demo.design_kwargs = design_kwargs if _gr_major >= 6 else {}
    return demo
```

- [ ] **Step 2: Smoke-check layout import**

```bash
python -c "from ui.layout import create_layout; print('layout OK')"
```

Expected: `layout OK`

- [ ] **Step 3: Commit**

```bash
git add ui/layout.py ui/__init__.py
git commit -m "feat: DoodleDreams moonlit-scrapbook UI (Gradio layout + CSS)"
```

---

### Task 6: app.py — combined ZeroGPU orchestrator

**Files:**
- Create: `D:\Project\Hugging_face_app\doodledreams\app.py`

- [ ] **Step 1: Write app.py**

```python
"""
DoodleDreams — combined ZeroGPU app
Draw a character + record your voice → illustrated bedtime story narrated in your voice.
"""
import os, sys, json, time, tempfile, logging, struct, threading
import torch

sys.path.insert(0, os.path.dirname(__file__))

try:
    import spaces
except ModuleNotFoundError:
    class _Shim:
        @staticmethod
        def GPU(*a, **k):
            return a[0] if a and callable(a[0]) else (lambda fn: fn)
    spaces = _Shim()

import gradio as gr
from config import (
    FLUX_MODEL, STORY_MODEL, TTS_MODEL, TRANSLATION_MODEL,
    KANNADA_TTS_MODEL, BASE_SEED, FLUX_STEPS, FLUX_GUIDANCE, FLUX_SIZE,
    COLOR_ART_STYLE, COLOR_PAGE_SUFFIX, STORY_LENGTHS,
)
from book_builder import build_book_html, export_pdf, magic_loader_html
from ui.layout import create_layout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ON_ZEROGPU = bool(os.environ.get("SPACES_ZERO_GPU"))

# ── module-level model caches ──────────────────────────────────────────
_FLUX       = None
_STORY_M    = None; _STORY_TOK = None
_TTS_M      = None
_TRANS_M    = None; _TRANS_TOK = None
_KAN_TTS_M  = None
_LOAD_ERRORS = {}


def _load_flux():
    global _FLUX
    if _FLUX is None:
        from diffusers import Flux2KleinPipeline
        _FLUX = Flux2KleinPipeline.from_pretrained(
            FLUX_MODEL.hub_id, torch_dtype=torch.bfloat16).to("cuda")
    return _FLUX


def _load_story():
    global _STORY_M, _STORY_TOK
    if _STORY_M is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _STORY_TOK = AutoTokenizer.from_pretrained(STORY_MODEL.hub_id, trust_remote_code=True)
        _STORY_M   = AutoModelForCausalLM.from_pretrained(
            STORY_MODEL.hub_id, torch_dtype=torch.float16, trust_remote_code=True,
        ).to("cuda").eval()
    return _STORY_M, _STORY_TOK


def _load_tts():
    global _TTS_M
    if _TTS_M is None:
        from voxcpm import VoxCPM
        _TTS_M = VoxCPM.from_pretrained(TTS_MODEL.hub_id, load_denoiser=False)
    return _TTS_M


def _load_translation():
    global _TRANS_M, _TRANS_TOK
    if _TRANS_M is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _TRANS_TOK = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL.hub_id, trust_remote_code=True)
        _TRANS_M   = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL.hub_id, trust_remote_code=True,
        ).to("cuda").eval()
    return _TRANS_M, _TRANS_TOK


def _load_kannada_tts():
    global _KAN_TTS_M
    if _KAN_TTS_M is None:
        from indic_tts import _get_model   # uses module-level cache inside indic_tts
        _KAN_TTS_M = _get_model()
    return _KAN_TTS_M


if ON_ZEROGPU:
    for _n, _fn in [("flux",_load_flux),("story",_load_story),
                    ("tts",_load_tts),("translation",_load_translation),
                    ("kannada_tts",_load_kannada_tts)]:
        try:
            _fn()
        except Exception as e:
            _LOAD_ERRORS[_n] = repr(e)
            logger.exception(f"Module-level load failed: {_n}")


# ── ZeroGPU inference functions ────────────────────────────────────────

@spaces.GPU(duration=60)
def _gen_story_gpu(hero_name: str, genre: str, mood: str, word_target: int) -> dict:
    from story import generate_story
    return generate_story(
        hero_name=hero_name, genre=genre, mood=mood, word_target=word_target)


@spaces.GPU(duration=150)
def _gen_images_gpu(char_desc: str, scenes: list,
                    doodle_bytes: bytes | None, seed: int) -> list:
    import io
    from PIL import Image
    pipe = _load_flux()
    canonical = None
    if doodle_bytes:
        try:
            ref = Image.open(io.BytesIO(doodle_bytes)).convert("RGB")
            canonical = pipe(
                prompt=(f"Turn this child's drawing into a clean, full-body cartoon "
                        f"character for a children's storybook. Keep the EXACT same creature. "
                        f"{COLOR_ART_STYLE}, plain white background, full character visible, centered."),
                image=ref,
                height=FLUX_SIZE, width=FLUX_SIZE,
                guidance_scale=FLUX_GUIDANCE,
                num_inference_steps=FLUX_STEPS,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
        except Exception as e:
            logger.warning(f"Canonical pass failed ({e}); text2img fallback")

    images = []
    for i, scene in enumerate(scenes):
        if canonical is not None:
            kw = dict(image=canonical,
                      prompt=f"The same character. {scene}. {COLOR_ART_STYLE}, {COLOR_PAGE_SUFFIX}")
        else:
            kw = dict(prompt=f"{char_desc}. Scene: {scene}. {COLOR_ART_STYLE}, centered.")
        kw.update(height=FLUX_SIZE, width=FLUX_SIZE,
                  guidance_scale=FLUX_GUIDANCE, num_inference_steps=FLUX_STEPS,
                  generator=torch.Generator("cuda").manual_seed(seed + i + 1))
        images.append(pipe(**kw).images[0])
        logger.info(f"Page {i+1}/{len(scenes)} illustrated")
    return images


@spaces.GPU(duration=120)
def _gen_tts_gpu(text: str, ref_audio_path: str | None,
                 mood: str, energy: float, language: str) -> bytes:
    import io, numpy as np
    if language == "Kannada":
        from indic_text import translate_to_kannada
        from indic_tts import narrate_kannada
        kannada_text = translate_to_kannada(text)
        ref_txt = "ಇದು ನನ್ನ ಧ್ವನಿ"   # fallback ref transcript
        return narrate_kannada(
            ref_audio_path or "", ref_txt, kannada_text, mood, energy)
    else:
        from tts import clone_and_speak
        return clone_and_speak(
            ref_audio_path=ref_audio_path,
            text=text, speed=1.0, mood=mood.lower(), energy=energy)


# ── heartbeat helper (same pattern as DoodleBook) ─────────────────────

def _with_heartbeat(blocking_fn, frame_fn, poll=4.0):
    box = {}
    def _run():
        try:   box["val"] = blocking_fn()
        except BaseException as e: box["err"] = e
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    t0 = time.time()
    while th.is_alive():
        th.join(timeout=poll)
        if th.is_alive():
            yield ("hb", frame_fn(int(time.time() - t0)))
    if "err" in box:
        raise box["err"]
    yield ("done", box["val"])


# ── main generator ─────────────────────────────────────────────────────

def create_book(doodle_image, ref_audio, hero_name, genre, mood, language, length_label):
    t0 = time.perf_counter()
    hero_name   = (hero_name or "").strip() or "Little Hero"
    word_target = STORY_LENGTHS.get(length_label, 120)
    energy      = 0.55   # mid-calm default

    trace = {
        "backend": "zerogpu", "hero": hero_name,
        "genre": genre, "mood": mood, "language": language,
        "seed": BASE_SEED, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if _LOAD_ERRORS:
        trace["load_errors"] = _LOAD_ERRORS

    _no   = gr.update(visible=False)
    _keep = gr.update()

    yield (magic_loader_html("story", hero_name),
           "Writing the bedtime story…", None, _no, {}, "")

    try:
        story = _gen_story_gpu(hero_name, genre, mood, word_target)
    except Exception as e:
        yield (f"<div class='page-loading'>Error: {e}</div>",
               f"Error: {e}", None, _no, {}, "")
        return

    title      = story.get("title", "A Bedtime Story")
    pages      = story.get("pages", [])
    char_desc  = story.get("character_description", "")
    scenes     = [p.get("scene","") for p in pages]
    page_texts = [p.get("text","")  for p in pages]
    full_text  = f"{title}. {' '.join(page_texts)}"
    trace.update(title=title, char_desc=char_desc)

    yield (magic_loader_html("images", hero_name),
           f"{title} — illustrating…", None, _no, story, json.dumps(trace, indent=2))

    doodle_bytes = None
    if doodle_image is not None:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(doodle_image).save(buf, format="PNG")
        doodle_bytes = buf.getvalue()

    # ── narration starts in parallel with illustration ──
    voice_box = {}
    def _do_voice():
        try:
            voice_box["bytes"] = _gen_tts_gpu(full_text, ref_audio, mood, energy, language)
        except Exception as e:
            voice_box["err"] = e

    voice_th = threading.Thread(target=_do_voice, daemon=True)
    voice_th.start()

    def _audio_now():
        if voice_box.get("bytes") and not voice_box.get("path"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(voice_box["bytes"]); voice_box["path"] = f.name
            except Exception: pass
        return voice_box.get("path")

    # ── image generation with heartbeat ──
    img_bytes, engine = None, "sketch"
    try:
        for kind, payload in _with_heartbeat(
            lambda: _gen_images_gpu(char_desc, scenes, doodle_bytes, BASE_SEED),
            lambda s: (
                magic_loader_html("images", hero_name),
                f"{title} — illustrating… {s}s"
                + ("  · narration ready ▶" if _audio_now() else "  · recording…"),
                _audio_now(), _no, story, json.dumps(trace, indent=2),
            ),
        ):
            if kind == "hb":
                yield payload
            else:
                import io
                raw = payload
                img_bytes = []
                for img in raw:
                    buf = io.BytesIO(); img.save(buf, format="PNG")
                    img_bytes.append(buf.getvalue())
                engine = "flux"
    except Exception as e:
        logger.exception("Image generation failed")
        trace["image_error"] = repr(e)
        from services.images import generate_placeholder_images
        img_bytes = generate_placeholder_images(char_desc, scenes, doodle_bytes)

    book_html = build_book_html(img_bytes, page_texts, title, engine)

    # ── wait for narration ──
    while voice_th.is_alive():
        voice_th.join(timeout=4)
        if voice_th.is_alive():
            yield (book_html, f"{title} — finishing narration…",
                   _audio_now(), _no, story, json.dumps(trace, indent=2))
    audio_path = _audio_now()
    if voice_box.get("err"):
        trace["tts_error"] = repr(voice_box["err"])

    # ── PDF ──
    pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = export_pdf(img_bytes, page_texts, title, f.name)
    except Exception as e:
        logger.warning(f"PDF failed: {e}")

    trace["total_sec"] = round(time.perf_counter() - t0, 2)
    trace["engine"] = engine

    pdf_update = gr.update(value=pdf_path, visible=True) if pdf_path else _keep

    yield (
        book_html,
        f"Done: {title} · {len(img_bytes)} pages · {language} · {trace['total_sec']}s",
        audio_path, pdf_update, story, json.dumps(trace, indent=2),
    )


# ── launch ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = create_layout(create_book_fn=create_book)
    demo.queue(default_concurrency_limit=2, max_size=8)
    demo.launch(share=False, allowed_paths=[tempfile.gettempdir()],
                **demo.design_kwargs)
```

- [ ] **Step 2: Verify import (no GPU/models needed)**

```bash
python -c "
import sys
sys.modules['spaces'] = type(sys)('spaces')
sys.modules['spaces'].GPU = lambda *a,**k: (a[0] if a and callable(a[0]) else lambda f:f)
import ast, pathlib
src = pathlib.Path('app.py').read_text()
ast.parse(src)
print('app.py syntax OK')
"
```

Expected: `app.py syntax OK`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: app.py — combined ZeroGPU orchestrator (doodle + voice + bedtime story)"
```

---

### Task 7: Push to GitHub + create HF Space

**Files:** README.md already written in Task 4.

- [ ] **Step 1: Create GitHub repo (do this manually or via gh CLI)**

```bash
# If gh CLI is available:
gh repo create Sushruths04/doodledreams --public --description "Draw a character, record your voice, get a bedtime storybook"
git remote add origin https://github.com/Sushruths04/doodledreams.git
git push -u origin main
```

- [ ] **Step 2: Add Hugging Face remote**

```bash
git remote add hf https://huggingface.co/spaces/build-small-hackathon/DoodleDreams.git
```

> The HF Space is auto-created when you push to its URL for the first time, IF you are a member of the `build-small-hackathon` org with write access.

- [ ] **Step 3: Push to HF**

```bash
git push hf main
```

Expected: HF picks up the push, Space rebuilds (yellow → green in ~2 min).

- [ ] **Step 4: Verify Space is live**

Open `https://huggingface.co/spaces/build-small-hackathon/DoodleDreams` — you should see the DoodleDreams UI with the moon/scrapbook design and a "Ready to dream!" status.

---

### Task 8: Smoke test — end-to-end check

- [ ] **Step 1: Test English story + FLUX illustration (no voice)**

In the live Space:
1. Upload `assets/sample_doodle.jpg`
2. Skip voice recording
3. Hero name: "Ziggy"
4. Genre: Animals · Mood: Calming · Language: English · Short
5. Tap "Make my bedtime book!"

Expected: story text appears, 6 illustrated pages, audio narration (default TTS voice), Story PDF download visible.

- [ ] **Step 2: Test voice cloning (English)**

1. Record 15–30 s of yourself speaking any sentence
2. Same inputs as Step 1
3. Tap "Make my bedtime book!"

Expected: narration audio is in the recorded voice (not a generic voice).

- [ ] **Step 3: Test Kannada**

1. Same inputs, Language: Kannada
2. Tap "Make my bedtime book!"

Expected: narration audio in Kannada. If IndicF5 fine-tune load fails, the app should still produce narration with the base IndicF5 model (fallback is coded in indic_tts.py).

- [ ] **Step 4: Test mobile layout**

Open the Space on a phone browser. Verify:
- Input card is full-width on top
- Output card is below (no horizontal scroll)
- Stars twinkle in the night sky header
- Moon orb glows

- [ ] **Step 5: Fix any issues and commit**

```bash
git add -A
git commit -m "fix: smoke test fixes"
git push hf main
git push origin main
```

---

## Known Risks

| Risk | Mitigation |
|---|---|
| `indic_tts.py` fine-tune checkpoint (`mitvho09/IndicF5-Kannada-Bedtime-v2`) not publicly accessible | `indic_tts.py` already has a try/except fallback to base IndicF5 — no code change needed |
| `IndicTransToolkit` pip install fails on HF Space | It's installed via `git+` in requirements.txt; the Space Dockerfile must have `git` available (Gradio SDK spaces do) |
| ZeroGPU 120 s slot too short for 6 FLUX pages + voice cloning | Images and narration run in separate `@spaces.GPU` calls; each has its own slot |
| `story.generate_story()` function signature mismatch | Confirm signature in story.py after copy; the call in app.py is `generate_story(hero_name, genre, mood, word_target)` — adjust if needed |
| Voice cloning with no ref audio | `tts.clone_and_speak(ref_audio_path=None, ...)` — confirm tts.py handles None gracefully (check its `prepare_reference` call) |
