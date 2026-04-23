# Setup

## Windows

1. Run `.setup/windows_setup.bat` to download ollama, create a Python virtual environment and install dependencies.
2. Update your `.env` file with API keys for any cloud models you want to use.

    Example `.env` contents:

    ```env
    OPENAI_API_KEY=sk-your-openai-api-key
    LITELLM_API_KEY=sk-your-litellm-proxy-key
    OLLAMA_API_KEY=your-ollama-cloud-key
    ```

3. Run `python src/test_llm_debate.py` inside the virtual environment to execute the example debate script.

## Linux

1. Run `.setup/linux_setup.sh` in the terminal to download ollama, create a Python virtual environment and install dependencies.
2. Update your `.env` file with API keys for any cloud models you want to use.

    Example `.env` contents:

    ```env
    OPENAI_API_KEY=sk-your-openai-api-key
    LITELLM_API_KEY=sk-your-litellm-proxy-key
    OLLAMA_API_KEY=your-ollama-cloud-key
    ```

3. Run `python src/test_llm_debate.py` inside the virtual environment to execute the example debate script.
