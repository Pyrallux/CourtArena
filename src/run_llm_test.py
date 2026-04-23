import argparse
import asyncio
import logging
import os
import yaml
from llm_client import create_client

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_default_model():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            models = config.get('ollama_models', [])
            if models:
                return models[0]
    except Exception as e:
        logging.warning("Could not read default model from model_config.yaml: %s", e)
    return "llama3.2:1b"

async def main():
    parser = argparse.ArgumentParser(description="Test LLM debate client")
    parser.add_argument("--model", type=str, help="Model name to test")
    args = parser.parse_args()

    # Example using a model (matches config formats)
    model_name = args.model if args.model else get_default_model()
    
    # Create the client using the factory method from llm_client.py
    client = create_client(
        model_name=model_name,
        temperature=0.7,
        max_tokens=1024
    )

    # Prepare the message payload
    messages = [
        {"role": "system", "content": "You are an expert legal debater."},
        {"role": "user", "content": "Present a short argument about AI liability."}
    ]

    print(f"Sending request to {model_name}...")
    try:
        # Execute the chat request
        response = await client.chat(messages=messages)
        
        print("\n--- Response ---")
        print(response.content)
        
        print("\n--- Metadata ---")
        print(response.to_metadata_dict())
        
    except Exception as e:
        print(f"Error during LLM call: {e}")

if __name__ == "__main__":
    asyncio.run(main())
