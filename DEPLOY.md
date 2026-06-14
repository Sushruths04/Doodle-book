# DoodleBook — Hugging Face Deployment Guide

## Option 1: HF Spaces + Modal (Recommended)

### Setup Steps

1. **Create HF Space**
```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create space
huggingface-cli space create your-username/doodlebook --sdk gradio
```

2. **Set Modal Secrets**
```bash
# In HF Space Settings → Secrets:
MODAL_TOKEN_ID=your_modal_token
HF_TOKEN=your_hf_token
```

3. **Upload Code**
```bash
cd doodlebook
git init
git add .
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/your-username/doodlebook
git push -u origin main
```

### Inference Times

| Scenario | Cold Start | Warm |
|----------|-----------|------|
| Sample Book | 0s | 0s |
| First Generation | 2-3 min | 30-60s |
| Subsequent | 30-60s | 20-40s |

---

## Option 2: HF ZeroGPU (Free, No Modal)

### Changes Needed

Replace Modal calls with direct inference on ZeroGPU:

```python
# In modal_workers/modal_image_gen.py
# Remove Modal, use direct torch inference
```

### Inference Times

| Scenario | Cold Start | Warm |
|----------|-----------|------|
| Sample Book | 0s | 0s |
| First Generation | 3-5 min | 1-2 min |
| Subsequent | 1-2 min | 45-90s |

---

## Option 3: HF Inference API

### Setup

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.2-klein-4B"
headers = {"Authorization": "Bearer your_hf_token"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content
```

### Inference Times

| Scenario | Cold Start | Warm |
|----------|-----------|------|
| Single Image | 10-30s | 5-15s |

---

## Recommendation for Hackathon

**Use Option 1 (HF Spaces + Modal)** because:
- ✅ Sample book loads instantly (no compute)
- ✅ Warm generation ~30s (fast demo)
- ✅ Modal keeps model warm during judging
- ✅ Free HF Space (CPU only)
- ✅ Modal only charges during generation

### Cost Estimate
- HF Space: Free
- Modal: ~$0.50 per demo generation
- Total for hackathon: ~$5-10
