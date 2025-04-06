from g4f import ChatCompletion, Provider
from flask import Flask, request, jsonify
from colorama import Fore, init
import logging

# Initialize colorama for colored output (optional for logs)
init()

# Set up Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of working providers that support both GPT-3.5 and GPT-4
working_providers = {
    "ChatGLM": Provider.ChatGLM,
    "Free2GPT": Provider.Free2GPT,
    "GizAI": Provider.GizAI,
    "Goabror": Provider.Goabror,
    "ImageLabs": Provider.ImageLabs,
    "MetaAI": Provider.MetaAI,
    "PollinationsAI": Provider.PollinationsAI,
    "PollinationsImage": Provider.PollinationsImage,
    "Qwen_QVQ_72B": Provider.Qwen_QVQ_72B,
    "Qwen_Qwen_2_5": Provider.Qwen_Qwen_2_5,
    "Qwen_Qwen_2_5M": Provider.Qwen_Qwen_2_5M,
    "Qwen_Qwen_2_5_Max": Provider.Qwen_Qwen_2_5_Max,
    "Qwen_Qwen_2_72B": Provider.Qwen_Qwen_2_72B,
    "Voodoohop_Flux1Schnell": Provider.Voodoohop_Flux1Schnell,
    "Websim": Provider.Websim,
    "Yqcloud": Provider.Yqcloud,
    "provider": Provider.provider
}

# Function to get a regular response
def get_response(prompt, provider_name, provider_class, model):
    try:
        response = ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            provider=provider_class
        )
        logger.info(f"{Fore.GREEN}Response generated successfully for {provider_name} with {model}")
        return response
    except Exception as e:
        logger.error(f"{Fore.RED}Error with {provider_name}: {str(e)}")
        return f"Error: {provider_name} - {str(e)}"

# Function to get a streaming response (collected as a string for API)
def get_streaming_response(prompt, provider_name, provider_class, model):
    try:
        response = ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            provider=provider_class,
            stream=True
        )
        full_response = ""
        for chunk in response:
            full_response += chunk
        logger.info(f"{Fore.GREEN}Streaming response generated successfully for {provider_name} with {model}")
        return full_response
    except Exception as e:
        logger.error(f"{Fore.RED}Error with {provider_name}: {str(e)}")
        return f"Error: {provider_name} - {str(e)}"

# API endpoint for generating responses
@app.route('/api/generate', methods=['POST'])
def generate_response():
    data = request.get_json()
    
    # Extract parameters from the request
    prompt = data.get('prompt', '')
    model = data.get('model', 'gpt-3.5-turbo')  # Default to GPT-3.5
    provider_name = data.get('provider', 'ChatGLM')  # Default to ChatGLM
    mode = data.get('mode', 'normal')  # Default to normal

    # Validate inputs
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    if model not in ["gpt-3.5-turbo", "gpt-4"]:
        logger.warning(f"{Fore.YELLOW}Invalid model: {model}. Defaulting to gpt-3.5-turbo")
        model = "gpt-3.5-turbo"
    
    if provider_name not in working_providers:
        logger.warning(f"{Fore.YELLOW}Invalid provider: {provider_name}. Defaulting to ChatGLM")
        provider_name = "ChatGLM"
    
    if mode not in ["normal", "streaming"]:
        logger.warning(f"{Fore.YELLOW}Invalid mode: {mode}. Defaulting to normal")
        mode = "normal"

    selected_provider = working_providers[provider_name]
    logger.info(f"{Fore.WHITE}Processing request: model={model}, provider={provider_name}, mode={mode}")

    # Generate response based on mode
    if mode == "streaming":
        result = get_streaming_response(prompt, provider_name, selected_provider, model)
    else:
        result = get_response(prompt, provider_name, selected_provider, model)

    # Return JSON response
    return jsonify({
        "model": model,
        "provider": provider_name,
        "mode": mode,
        "response": result
    })

# Run the Flask app
if __name__ == "__main__":
    print(f"{Fore.CYAN}Starting GPT API Server locally...")
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    print(f"{Fore.CYAN}Starting GPT API Server on Render...")
