import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
from textblob import TextBlob
import time

# Initialize Bybit API connection
bybit = ccxt.bybit({
    'apiKey': 'zfwyJlkyD32CZjaf0G',
    'secret': 'lirMdi0jPr4ynPdUcBwkznnNvBW7lHogtKh8'
})

class CryptoTradingBot:
    def __init__(self):
        self.exchange = bybit
        self.model = self._build_lstm_model()
        self.scaler = MinMaxScaler()
        
    def _build_lstm_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fetch_market_data(self, symbol='BTC/USD', timeframe='1h', limit=1000):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

    def calculate_indicators(self, df):
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2*df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2*df['close'].rolling(window=20).std()

        return df

    def get_sentiment_analysis(self):
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            response = requests.get(url)
            news_data = response.json()['Data']
            
            sentiment_scores = []
            for news in news_data[:10]:
                analysis = TextBlob(news['title'] + " " + news['body'])
                sentiment_scores.append(analysis.sentiment.polarity)
                
            return np.mean(sentiment_scores)
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return 0

    def predict_price(self, data):
        try:
            scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
            sequences = []
            for i in range(len(scaled_data) - 60):
                sequences.append(scaled_data[i:(i + 60)])
            sequences = np.array(sequences)
            
            prediction = self.model.predict(sequences[-1:])
            return self.scaler.inverse_transform(prediction)[0][0]
        except Exception as e:
            print(f"Error in price prediction: {e}")
            return None

    def execute_trade(self, symbol, side, amount, stop_loss=None, take_profit=None):
        try:
            params = {}
            if stop_loss:
                params['stopLoss'] = stop_loss
            if take_profit:
                params['takeProfit'] = take_profit
                
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
                params=params
            )
            return order
        except Exception as e:
            print(f"Error executing trade: {e}")
            return None

    def risk_assessment(self, position_size, current_price, historical_data):
        try:
            # Calculate historical volatility
            returns = np.log(historical_data['close'] / historical_data['close'].shift(1))
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Value at Risk calculation
            confidence_level = 0.95
            var = position_size * current_price * volatility * np.sqrt(1/252)
            
            # Maximum drawdown
            rolling_max = historical_data['close'].rolling(window=252, min_periods=1).max()
            drawdown = (historical_data['close'] - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            return {
                'var': var,
                'volatility': volatility,
                'max_drawdown': max_drawdown
            }
        except Exception as e:
            print(f"Error in risk assessment: {e}")
            return None

class TradingAgent:
    def __init__(self, bot):
        self.bot = bot
        self.min_confidence = 0.7
        self.position_size = 0.01  # Default position size in BTC
        self.max_var = 1000  # Maximum Value at Risk in USD

    def analyze_market(self, symbol):
        df = self.bot.fetch_market_data(symbol)
        if df is None:
            return None
        
        df = self.bot.calculate_indicators(df)
        sentiment = self.bot.get_sentiment_analysis()
        predicted_price = self.bot.predict_price(df['close'].values)
        
        current_price = df['close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal = df['Signal_Line'].iloc[-1]
        bb_upper = df['BB_upper'].iloc[-1]
        bb_lower = df['BB_lower'].iloc[-1]
        
        # Risk assessment
        risk_metrics = self.bot.risk_assessment(self.position_size, current_price, df)
        if risk_metrics and risk_metrics['var'] > self.max_var:
            return None
            
        confidence = 0
        trade_signal = None
        stop_loss = None
        take_profit = None
        
        # Long signal conditions
        if (rsi < 30 and 
            macd > signal and 
            sentiment > 0 and 
            current_price < bb_lower and
            predicted_price > current_price):
            
            confidence = 0.8
            trade_signal = 'buy'
            stop_loss = current_price * 0.95  # 5% stop loss
            take_profit = current_price * 1.1  # 10% take profit
            
        # Short signal conditions
        elif (rsi > 70 and 
              macd < signal and 
              sentiment < 0 and 
              current_price > bb_upper and
              predicted_price < current_price):
              
            confidence = 0.8
            trade_signal = 'sell'
            stop_loss = current_price * 1.05  # 5% stop loss
            take_profit = current_price * 0.9  # 10% take profit
            
        if confidence >= self.min_confidence:
            return {
                'signal': trade_signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        return None

# Initialize and run the bot
if __name__ == "__main__":
    bot = CryptoTradingBot()
    agent = TradingAgent(bot)
    
    while True:
        try:
            trade_info = agent.analyze_market('BTC/USD')
            if trade_info:
                order = bot.execute_trade(
                    'BTC/USD', 
                    trade_info['signal'], 
                    agent.position_size,
                    trade_info['stop_loss'],
                    trade_info['take_profit']
                )
                if order:
                    print(f"Executed {trade_info['signal']} order: {order}")
                    print(f"Stop Loss: {trade_info['stop_loss']}")
                    print(f"Take Profit: {trade_info['take_profit']}")
            
            # Wait for 5 minutes before next analysis
            time.sleep(300)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)  # Wait 1 minute on error before retrying
