#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt || exit 1

# Install SpaCy model
python -m spacy download en_core_web_sm || exit 1

echo "✅ Streamlit Cloud setup completed successfully!"
