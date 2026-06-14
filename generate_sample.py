"""
Generate sample book for assets/sample_book/
Run this once to create the pre-generated sample (C3 non-negotiable).
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import SAMPLE_BOOK_PATH
from book_builder import build_book_html, export_pdf
from services.images import generate_placeholder_images


def generate_sample_book():
    """Generate and save the sample book."""
    # Sample story
    title = "Ziggy's Rainbow Adventure"
    char_desc = "A friendly little robot named Ziggy with big eyes and a round body, drawn in crayon style"
    
    pages = [
        {"page": 1, "text": "Once upon a time, there was a little robot named Ziggy who lived in a colorful forest.", "scene": "Ziggy in forest"},
        {"page": 2, "text": "Ziggy loved to explore and find new friends among the trees.", "scene": "Exploring forest"},
        {"page": 3, "text": "One day, Ziggy discovered a magical rainbow bridge stretching across the sky.", "scene": "Discovering bridge"},
        {"page": 4, "text": "A little bird was stuck on the other side. Ziggy knew he had to help.", "scene": "Bird needs help"},
        {"page": 5, "text": "Bravely, Ziggy crossed the rainbow bridge to rescue the bird.", "scene": "Crossing bridge"},
        {"page": 6, "text": "All the forest friends cheered. Ziggy was a hero. The end.", "scene": "Happy ending"}
    ]
    
    texts = [p["text"] for p in pages]
    
    # Generate images
    print("Generating placeholder images...")
    images = generate_placeholder_images()
    
    # Build HTML
    html = build_book_html(images, texts, title)
    
    # Save
    os.makedirs(SAMPLE_BOOK_PATH, exist_ok=True)
    
    html_path = os.path.join(SAMPLE_BOOK_PATH, "sample.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved HTML: {html_path}")
    
    # Save individual pages
    for i, img_bytes in enumerate(images):
        page_path = os.path.join(SAMPLE_BOOK_PATH, f"page_{i+1}.png")
        with open(page_path, "wb") as f:
            f.write(img_bytes)
    print(f"Saved {len(images)} pages to {SAMPLE_BOOK_PATH}")
    
    # Generate PDF
    pdf_path = os.path.join(SAMPLE_BOOK_PATH, "sample.pdf")
    export_pdf(images, texts, title, pdf_path)
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    generate_sample_book()
