from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# API keys from .env
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
CHATGPT_URL = "https://api.openai.com/v1/completions"

CORS(app)  # Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

@app.route("/get-stock-data", methods=["POST"])
def get_stock_data():
    """
    Fetch stock data for a given symbol and date range.
    """
    data = request.json
    symbol = data.get("symbol")
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    # Validation
    if not symbol or not start_date or not end_date:
        return jsonify({"error": "Symbol, start_date, and end_date are required"}), 400

    # Build the Polygon.io API URL
    url = f"{BASE_URL}/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&apiKey={POLYGON_API_KEY}"

    try:
        # Make the request to Polygon.io
        response = requests.get(url)
        response_data = response.json()

        if response.status_code == 200:
            return jsonify({
                "symbol": symbol.upper(),
                "start_date": start_date,
                "end_date": end_date,
                "data": response_data.get("results", [])
            })
        else:
            return jsonify({
                "error": response_data.get("message", "Error fetching stock data")
            }), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask-chatgpt", methods=["POST"])
def ask_chatgpt():
    """
    Query ChatGPT for general or financial-related questions.
    """
    data = request.json
    question = data.get("question")

    # Validation
    if not question:
        return jsonify({"error": "A question is required"}), 400

    headers = {
        "Authorization": f"Bearer {CHATGPT_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",  # Updated model
        "messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": question}
        ],
        "max_tokens": 150,  # Adjust token limit based on requirements
        "temperature": 0.7
    }

    try:
        # Make the request to OpenAI
        response = requests.post(CHATGPT_URL, headers=headers, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            return jsonify({"response": response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()})
        else:
            return jsonify({"error": response_data.get("error", {}).get("message", "Error querying ChatGPT")}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
