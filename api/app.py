from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


class StockPredictor:
    def __init__(self, model_type='linear'):
        """
        Initialize the stock predictor
        Args:
            model_type (str): 'linear' or 'knn'
        """
        self.model_type = model_type
        self.model = LinearRegression() if model_type == 'linear' else KNeighborsRegressor(n_neighbors=5)
        self.scaler = MinMaxScaler()
        
    def fetch_historical_data(self, symbol, start_date, end_date):
        """
        Fetch historical data for a single symbol with extended date range
        """
        url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&apiKey={API_KEY}'
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if "results" in data and data["results"]:
                df = pd.DataFrame(data["results"])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df = df.rename(columns={
                    'o': 'open',
                    'c': 'close',
                    'h': 'high',
                    'l': 'low',
                    'v': 'volume',
                    'timestamp': 'date'
                })
                df.set_index('date', inplace=True)
                return df[['open', 'close', 'high', 'low', 'volume']]
        return None

    def prepare_features(self, df):
        """
        Prepare technical indicators and features
        """
        df = df.copy()
        
        # Technical indicators
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['Price_Change'] = df['close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=5).std()
        df['RSI'] = self._calculate_rsi(df['close'])
        df['MACD'] = self._calculate_macd(df['close'])
        
        # Create target variable (next day's closing price)
        df['target'] = df['close'].shift(-1)
        
        # Drop NaN values
        df = df.dropna()
        
        return df

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        return exp1 - exp2

    def train_model(self, symbol, start_date, end_date):
        """
        Train the model using historical data
        """
        # Fetch and prepare data
        df = self.fetch_historical_data(symbol, start_date, end_date)
        if df is None:
            return None
        
        df = self.prepare_features(df)
        
        # Prepare features for training
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 
                         'SMA_5', 'SMA_20', 'Price_Change', 'Volatility', 
                         'RSI', 'MACD']
        
        X = df[feature_columns]
        y = df['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        return {
            'model_score': self.model.score(X_test, y_test),
            'last_date': df.index[-1],
            'last_price': df['close'].iloc[-1]
        }

    def predict_future(self, df, days=150):
        """
        Predict future stock prices
        """
        predictions = []
        current_data = df.iloc[-1:].copy()
        
        for _ in range(days):
            # Prepare features for prediction
            pred_data = self.prepare_features(current_data)
            features = ['open', 'high', 'low', 'close', 'volume', 
                       'SMA_5', 'SMA_20', 'Price_Change', 'Volatility', 
                       'RSI', 'MACD']
            
            # Scale features
            scaled_data = self.scaler.transform(pred_data[features].iloc[-1:])
            
            # Make prediction
            pred_price = self.model.predict(scaled_data)[0]
            predictions.append(pred_price)
            
            # Update current data for next prediction
            new_row = current_data.iloc[-1:].copy()
            new_row['close'] = pred_price
            new_row['open'] = pred_price
            new_row['high'] = pred_price * 1.01
            new_row['low'] = pred_price * 0.99
            current_data = new_row
        
        return predictions

    def analyze_trends(self, predictions, dates):
        """
        Analyze predicted trends and identify significant changes
        """
        trends = []
        for i in range(1, len(predictions)):
            change = ((predictions[i] - predictions[i-1]) / predictions[i-1]) * 100
            
            if abs(change) >= 2:  # Significant change threshold
                trends.append({
                    'date': dates[i].strftime('%Y-%m-%d'),
                    'price': round(predictions[i], 2),
                    'change': round(change, 2),
                    'type': 'RISE' if change > 0 else 'DIP'
                })
        
        return trends

    def get_full_prediction(self, symbol, start_date, end_date):
        """
        Get complete prediction analysis for a stock
        """
        # Train model
        training_result = self.train_model(symbol, start_date, end_date)
        if training_result is None:
            return None
            
        # Get historical data for feature generation
        df = self.fetch_historical_data(symbol, start_date, end_date)
        
        # Generate future dates
        last_date = training_result['last_date']
        future_dates = [last_date + timedelta(days=x) for x in range(1, 151)]
        
        # Make predictions
        predictions = self.predict_future(df)
        
        # Analyze trends
        trends = self.analyze_trends(predictions, future_dates)
        
        return {
            'symbol': symbol,
            'model_accuracy': round(training_result['model_score'] * 100, 2),
            'last_actual_price': round(training_result['last_price'], 2),
            'predictions': [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'price': round(price, 2)
                }
                for date, price in zip(future_dates, predictions)
            ],
            'trends': trends
        }

# Flask Routes

@app.route('/api/stocks', methods=['GET'])
def get_all_stocks():
    url = f'https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/2024-10-09?adjusted=true&apiKey={API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            return jsonify(data["results"])
        else:
            return jsonify({"error": "No results found"}), 404
    return jsonify({'error': 'Failed to fetch data'}), 500

@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2024-10-09/2024-10-31?adjusted=true&apiKey={API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            stock_data = []
            for result in data["results"]:
                stock_data.append({
                    "Date": result["t"],
                    "Open": result["o"],
                    "Close": result["c"],
                    "High": result["h"],
                    "Low": result["l"],
                    "Volume": result["v"]
                })
            return jsonify({"ticker": ticker, "data": stock_data})
        else:
            return jsonify({"error": "Data not found"}), 404
    return jsonify({'error': 'Failed to fetch data'}), 500

@app.route('/api/stock/<ticker>/details', methods=['GET'])
def get_stock_details(ticker):
    url = f'https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
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

@app.route('/api/stock/<ticker>/history', methods=['GET'])
def get_stock_history(ticker):
    start_date = request.args.get('start', '2024-10-09')
    end_date = request.args.get('end', '2024-10-31')
    
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&apiKey={API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            historical_data = [
                {
                    "Date": result["t"],
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

@app.route('/api/stock/<ticker>/predict', methods=['GET'])
def predict_stock(ticker):
    """
    New endpoint for stock price prediction
    """
    start_date = request.args.get('start', '2024-10-09')
    end_date = request.args.get('end', '2024-10-31')
    model_type = request.args.get('model', 'linear')  # 'linear' or 'knn'
    
    predictor = StockPredictor(model_type=model_type)
    prediction_results = predictor.get_full_prediction(ticker, start_date, end_date)
    
    if prediction_results is None:
        return jsonify({'error': 'Failed to generate prediction'}), 500
        
    return jsonify(prediction_results)

if __name__ == '__main__':
    app.run(debug=True)