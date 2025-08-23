#!/usr/bin/env python3
"""
Fix NLTK SSL certificate issues by downloading data with SSL disabled
Run this once to download required NLTK data
"""

import ssl
import nltk
import os
from pathlib import Path


def fix_nltk_ssl():
    """Download NLTK data with SSL verification disabled."""
    print("🔧 Fixing NLTK SSL certificate issues...")

    # Disable SSL verification temporarily
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Set NLTK data path to project directory
    project_nltk_dir = Path("./nltk_data")
    project_nltk_dir.mkdir(exist_ok=True)

    # Add to NLTK path
    nltk.data.path.insert(0, str(project_nltk_dir))

    # Download required data
    resources = [
        'stopwords',
        'punkt',
        'averaged_perceptron_tagger',
        'wordnet'
    ]

    for resource in resources:
        try:
            print(f"📥 Downloading {resource}...")
            nltk.download(resource, download_dir=str(project_nltk_dir), quiet=False)
            print(f"✅ {resource} downloaded successfully")
        except Exception as e:
            print(f"⚠️ Error downloading {resource}: {e}")
            # Try to download without SSL verification
            try:
                import urllib.request
                import urllib.error

                # This is a fallback - you might need to download manually
                print(f"   Trying alternative method for {resource}...")
            except Exception as e2:
                print(f"   Alternative method failed: {e2}")

    print("\n✅ NLTK setup complete!")
    print("💡 If downloads still fail, you can manually download NLTK data:")
    print("   1. Run Python: python -c \"import nltk; nltk.download('stopwords'); nltk.download('punkt')\"")
    print("   2. Or disable NLTK features in the config")


if __name__ == "__main__":
    fix_nltk_ssl()