#!/usr/bin/env python3
"""
Fix NLTK stopwords error for LlamaIndex.
Run this script to download required NLTK data.
"""

import nltk
import os
from pathlib import Path


def download_nltk_data():
    """Download required NLTK data."""
    print("=" * 60)
    print("DOWNLOADING NLTK DATA")
    print("=" * 60)

    # Set NLTK data path to project directory
    project_nltk_dir = Path("./nltk_data")
    project_nltk_dir.mkdir(exist_ok=True)

    # Add to NLTK path
    nltk.data.path.append(str(project_nltk_dir))

    # Download required data
    resources = [
        'stopwords',
        'punkt',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]

    for resource in resources:
        try:
            print(f"\nDownloading {resource}...")
            nltk.download(resource, download_dir=str(project_nltk_dir))
            print(f"✅ {resource} downloaded successfully")
        except Exception as e:
            print(f"⚠️ Error downloading {resource}: {e}")

    print("\n" + "=" * 60)
    print("✅ NLTK DATA DOWNLOADED")
    print("=" * 60)

    # Create a helper file to set NLTK path
    helper_content = '''"""
NLTK Data Path Helper
Add this to your imports to ensure NLTK finds the data.
"""

import nltk
from pathlib import Path

# Add project NLTK data directory to path
project_nltk = Path(__file__).parent.parent / "nltk_data"
if project_nltk.exists():
    nltk.data.path.insert(0, str(project_nltk))
'''

    helper_path = Path("backend/nltk_helper.py")
    helper_path.write_text(helper_content)
    print(f"\n✅ Created {helper_path}")

    print("\n📝 Next steps:")
    print("1. The NLTK data has been downloaded to ./nltk_data/")
    print("2. A helper file has been created at backend/nltk_helper.py")
    print("3. Run your app again - it should work now!")


if __name__ == "__main__":
    download_nltk_data()