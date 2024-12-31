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
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"


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



if __name__ == "__main__":
    app.run(debug=True)
