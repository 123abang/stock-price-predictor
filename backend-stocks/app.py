from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Fetch all stocks data route
@app.route('/api/stocks', methods=['GET'])
def get_all_stocks():
    url = f'https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/2023-11-01?adjusted=true&apiKey={API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            return jsonify(data["results"])  # Return the list of stocks
        else:
            return jsonify({"error": "No results found"}), 404
    return jsonify({'error': 'Failed to fetch data'}), 500

# Fetch individual stock data for charting route
@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2023-01-01/2023-12-31?adjusted=true&apiKey={API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            stock_data = []
            for result in data["results"]:
                stock_data.append({
                    "Date": result["t"],  # Millisecond timestamp
                    "Open": result["o"],
                    "Close": result["c"],
                    "High": result["h"],
                    "Low": result["l"],
                    "Volume": result["v"]
                })
            return jsonify({"ticker": ticker, "data": stock_data})  # Return the stock data in a structured way
        else:
            return jsonify({"error": "Data not found"}), 404
    return jsonify({'error': 'Failed to fetch data'}), 500

# Fetch stock details including company name, description, and industry
@app.route('/api/stock/<ticker>/details', methods=['GET'])
def get_stock_details(ticker):
    # Fetch the stock details including company name from Polygon API
    url = f'https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            # Get the company details from the API response
            company_details = {
                "ticker": ticker,
                "companyName": data["results"].get("name", "N/A"), 
                "description": data["results"].get("description", "N/A"),
                "industry": data["results"].get("sic_description", "N/A"),
                "sector": data["results"].get("sector", "N/A"),
                "market_cap": data["results"].get("market_cap", "N/A"),
                "employees": data["results"].get("employees", "N/A"),
                "current_price": data["results"].get("last_price", "N/A"), 
                "open": data["results"].get("open", "N/A"),
                "close": data["results"].get("close", "N/A"),
                "high": data["results"].get("high", "N/A"),
                "low": data["results"].get("low", "N/A"),
                "volume": data["results"].get("volume", "N/A")
            }
            return jsonify(company_details)
        else:
            return jsonify({"error": "Company details not found"}), 404
    return jsonify({'error': 'Failed to fetch company details'}), 500

# Fetch historical data for a given stock within a specific date range
@app.route('/api/stock/<ticker>/history', methods=['GET'])
def get_stock_history(ticker):
    # Get date range from query parameters
    start_date = request.args.get('start', '2023-01-01')  # Default start date
    end_date = request.args.get('end', '2023-12-31')      # Default end date
    
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&apiKey={API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            historical_data = [
                {
                    "Date": result["t"],  # Millisecond timestamp
                    "Open": result["o"],
                    "Close": result["c"],
                    "High": result["h"],
                    "Low": result["l"],
                    "Volume": result["v"]
                }
                for result in data["results"]
            ]
            return jsonify({"ticker": ticker, "history": historical_data})
        else:
            return jsonify({"error": "No historical data found"}), 404
    return jsonify({'error': 'Failed to fetch historical data'}), 500

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
