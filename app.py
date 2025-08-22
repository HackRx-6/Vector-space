from flask import Flask, request, jsonify
import asyncio
from agent import run_agent_with_prompt
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/')
def home():
    return "Flask server is running inside Docker!"

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        result = asyncio.run(run_agent_with_prompt(prompt))
        print(result)
        return jsonify({
            "message": "Execution successful",
            "output": str(result)
        })
    except Exception as e:
        print(str(e))
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Run on all interfaces, required for Docker
    app.run(host='0.0.0.0', port=7860)
