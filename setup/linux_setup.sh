#!/usr/bin/env bash

# install ollama client for linux (for local and cloud access)
curl -fsSL https://ollama.com/install.sh | sh

# install python dependencies in a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup .env file (replace with actual ollama api key)
echo "OLLAMA_API_KEY=your-ollama-cloud-key" > .env