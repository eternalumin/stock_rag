from typing import Dict, Any
import re
from utils.stock_data import get_stock_history
from utils.indicators import calculate_indicators, get_technical_summary
import config

class AnalysisAgent:
    def __init__(self):
        self.name = "Analysis Agent"
    
    def extract_ticker(self, query: str) -> str:
        pattern = r'\b([A-Z]{1,5})\b'
        words = query.upper().split()
        for word in words:
            if re.match(r'^[A-Z]{1,5}$', word) and len(word) <= 5:
                if word not in ['AND', 'OR', 'THE', 'FOR', 'WITH', 'FROM', 'WHAT', 'HOW', 'WHY', 'WHEN', 'WHO']:
                    return word
        return None
    
    def process(self, query: str) -> Dict[str, Any]:
        ticker = self.extract_ticker(query)
        
        if not ticker:
            return {
                "agent": self.name,
                "response": "Please provide a stock ticker symbol for analysis (e.g., 'Analyze NVDA')."
            }
        
        df = get_stock_history(ticker, period="6mo")
        
        if df is None or df.empty:
            return {
                "agent": self.name,
                "response": f"Could not fetch historical data for {ticker}."
            }
        
        df_with_indicators = calculate_indicators(df)
        technical_summary = get_technical_summary(df_with_indicators)
        
        latest = df_with_indicators.iloc[-1]
        
        response = f"## Technical Analysis: {ticker}\n\n"
        
        response += "### Price Summary\n"
        response += f"- **Current Price**: ${latest['Close']:.2f}\n"
        response += f"- **20-Day SMA**: ${latest['SMA_20']:.2f}\n"
        response += f"- **50-Day SMA**: ${latest['SMA_50']:.2f}\n"
        response += f"- **200-Day SMA**: ${latest['SMA_200']:.2f}\n\n"
        
        response += "### Technical Indicators\n"
        for indicator, value in technical_summary.items():
            response += f"- **{indicator}**: {value}\n"
        
        response += "\n### Momentum\n"
        response += f"- **RSI (14)**: {latest['RSI']:.2f}\n"
        response += f"- **MACD**: {latest['MACD']:.4f}\n"
        response += f"- **MACD Signal**: {latest['MACD_Signal']:.4f}\n"
        
        response += "\n### Volatility\n"
        response += f"- **ATR**: ${latest['ATR']:.2f}\n"
        response += f"- **BB Upper**: ${latest['BB_Upper']:.2f}\n"
        response += f"- **BB Lower**: ${latest['BB_Lower']:.2f}\n"
        
        return {
            "agent": self.name,
            "response": response,
            "ticker": ticker,
            "data": df_with_indicators.tail(30).to_dict(),
            "summary": technical_summary
        }
