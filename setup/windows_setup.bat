@echo off

:: install ollama client for windows (for local and cloud access)
echo Downloading and installing Ollama...
curl -L https://ollama.com/download/OllamaSetup.exe -o OllamaSetup.exe
start /wait OllamaSetup.exe
del OllamaSetup.exe

:: install python dependencies in a virtual environment
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt

:: Setup .env file (replace with actual ollama api key)
echo OLLAMA_API_KEY=your-ollama-cloud-key> .env
