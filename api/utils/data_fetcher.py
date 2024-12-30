# backend/utils/data_fetcher.py

import os
import requests
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Retrieve the API key from .env
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

def fetch_multiple_stocks_data(symbols):
    """
    Fetch historical stock data for multiple symbols from Polygon.io.

    Parameters:
    - symbols (list of str): List of stock symbols (e.g., ['AAPL', 'MSFT'])

    Returns:
    - pandas.DataFrame: Combined stock data with date and symbol as columns
    """
    all_data = []

    for symbol in symbols:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={POLYGON_API_KEY}"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print(data)  # Print the full response for debugging
            
            # Check if the response contains the expected keys
            if 'results' in data and data['results']:
                stock_data = data['results'][0]  # Get the first result
                # Extract relevant data
                stock_info = {
                    "Symbol": symbol,
                    "Date": stock_data["t"],
                    "Open": stock_data["o"],
                    "Close": stock_data["c"],
                    "High": stock_data["h"],
                    "Low": stock_data["l"],
                    "Volume": stock_data["v"]
                }
                all_data.append(stock_info)
            else:
                print(f"No results found for symbol: {symbol}")
        else:
            print(f"Error fetching data for {symbol}: {response.text}")

    # Create DataFrame from all stocks data
    df = pd.DataFrame(all_data)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], unit='ms')  # Convert date from milliseconds to datetime
        df.set_index("Date", inplace=True)
    return df

# Test fetching data for multiple stocks
if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = fetch_multiple_stocks_data(symbols)
    print(data)
