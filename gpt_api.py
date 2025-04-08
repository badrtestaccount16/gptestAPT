from g4f import ChatCompletion, Provider
from flask import Flask, request, jsonify, Response
from colorama import Fore, init
import logging
import json
from functools import wraps

# Initialize colorama for colored output
init()

# Set up Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# List of working providers
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

# Supported models
SUPPORTED_MODELS = ["gpt-3.5-turbo", "gpt-4"]

# Middleware to validate JSON input
def require_json(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        return f(*args, **kwargs)
    return decorated_function

# Root endpoint for API documentation
@app.route('/', methods=['GET'])
def home():
    return """
    <h1>GPT API</h1>
    <p>Welcome to the GPT API powered by g4f.</p>
    <h3>Endpoint: POST /api/generate</h3>
    <p>Generate responses using GPT-3.5 or GPT-4 with various providers.</p>
    <h4>Request Body (JSON):</h4>
    <pre>
    {
        "prompt": "Your question or text here (required)",
        "model": "gpt-3.5-turbo or gpt-4 (default: gpt-3.5-turbo)",
        "provider": "ChatGLM, MetaAI, etc. (default: ChatGLM)",
        "mode": "normal or streaming (default: normal)"
    }
    </pre>
    <h4>Example:</h4>
    <pre>
    curl -X POST https://your-api-url/api/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is AI?", "model": "gpt-4", "provider": "ChatGLM", "mode": "normal"}'
    </pre>
    <h4>Response (JSON):</h4>
    <pre>
    {
        "model": "gpt-4",
        "provider": "ChatGLM",
        "mode": "normal",
        "response": "Artificial Intelligence (AI) is..."
    }
    </pre>
    """

# Function to get a normal response
def get_normal_response(prompt, provider_name, provider_class, model):
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

# Function to get a streaming response (for SSE)
def get_streaming_response(prompt, provider_name, provider_class, model):
    try:
        response = ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            provider=provider_class,
            stream=True
        )
        for chunk in response:
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        logger.info(f"{Fore.GREEN}Streaming completed for {provider_name} with {model}")
    except Exception as e:
        logger.error(f"{Fore.RED}Error with {provider_name}: {str(e)}")
        yield f"data: {json.dumps({'error': f'Error: {provider_name} - {str(e)}'})}\n\n"

# API endpoint for generating responses
@app.route('/api/generate', methods=['POST'])
@require_json
def generate_response():
    data = request.get_json()
    
    # Extract and validate parameters
    prompt = data.get('prompt', '')
    model = data.get('model', 'gpt-3.5-turbo')
    provider_name = data.get('provider', 'ChatGLM')
    mode = data.get('mode', 'normal')

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if model not in SUPPORTED_MODELS:
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
        return Response(
            get_streaming_response(prompt, provider_name, selected_provider, model),
            mimetype='text/event-stream'
        )
    else:
        result = get_normal_response(prompt, provider_name, selected_provider, model)
        return jsonify({
            "model": model,
            "provider": provider_name,
            "mode": mode,
            "response": result
        })

if __name__ == "__main__":
    logger.info(f"{Fore.CYAN}Starting GPT API Server locally...")
    app.run(host='0.0.0.0', port=5000, debug=True)
