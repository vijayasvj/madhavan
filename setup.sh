#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install SpaCy model
python -m spacy download en_core_web_sm

echo "✅ Streamlit Cloud setup completed successfully!"
