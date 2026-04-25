# CourtArena: A Multi-Agent Debate Simulation for Legal Reasoning

## Setup Instructions

### Windows

1. Run `./setup/windows_setup.bat` to download ollama, create a Python virtual environment and install dependencies.
2. Update your `.env` file with API keys for any cloud models you want to use.

    Example `.env` contents:

    ```env
    OPENAI_API_KEY=sk-your-openai-api-key
    LITELLM_API_KEY=sk-your-litellm-proxy-key
    OLLAMA_API_KEY=your-ollama-cloud-key
    ```

### Linux

1. Run `./setup/linux_setup.sh` in the terminal to download ollama, create a Python virtual environment and install dependencies.
2. Update your `.env` file with API keys for any cloud models you want to use.

    Example `.env` contents:

    ```env
    OPENAI_API_KEY=sk-your-openai-api-key
    LITELLM_API_KEY=sk-your-litellm-proxy-key
    OLLAMA_API_KEY=your-ollama-cloud-key
    ```

## Running the Project

1. Run `python src/get_courtreasoner_cases.py` to download and parse cases from the CourtReasoner repository. This will save the cases in `case_data/courtreasoner_cases.json`.
2. Configure the models to use in the arena by editing `src/model_config.py`. Verify and initialize your models by running `python src/setup_models.py`. This will pull any local models and verify connectivity to cloud models.
3. Run the arena with `python src/run_court_arena.py`. This will load the cases, run the agents, and log the process to `/logs`.

## References

- CourtReasoner: Can LLM Agents Reason Like Judges? Simeng Han, Yoshiki Takashima, Shannon Zejiang Shen, Chen Liu, Yixin Liu,  <https://aclanthology.org/2025.emnlp-main.1787.pdf>

- AgenticSimLaw: A Juvenile Courtroom Multi-Agent Debate Simulation for Explainable High-Stakes Tabular Decision Making <https://arxiv.org/pdf/2601.21936>
