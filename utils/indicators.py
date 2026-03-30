import pandas as pd
import numpy as np
from typing import Dict, Optional
import ta

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
    
    df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
    
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    df['BB_Upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BB_Lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['BB_Mid'] = ta.volatility.BollingerBands(df['Close']).bollinger_mavg()
    
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    return df

def get_technical_summary(df: pd.DataFrame) -> Dict[str, str]:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    summary = {}
    
    if latest['RSI'] > 70:
        summary['RSI'] = f"Oversold ({latest['RSI']:.1f})"
    elif latest['RSI'] < 30:
        summary['RSI'] = f"Overbought ({latest['RSI']:.1f})"
    else:
        summary['RSI'] = f"Neutral ({latest['RSI']:.1f})"
    
    if latest['Close'] > latest['SMA_50']:
        summary['Trend'] = "Bullish (Price above 50-day SMA)"
    else:
        summary['Trend'] = "Bearish (Price below 50-day SMA)"
    
    if latest['MACD'] > latest['MACD_Signal']:
        summary['MACD'] = "Bullish Crossover"
    else:
        summary['MACD'] = "Bearish Crossover"
    
    if latest['Close'] > latest['BB_Upper']:
        summary['Bollinger'] = "Overbought (Above upper band)"
    elif latest['Close'] < latest['BB_Lower']:
        summary['Bollinger'] = "Oversold (Below lower band)"
    else:
        summary['Bollinger'] = "Within bands"
    
    return summary
