import re
from typing import Dict, Any
from utils.stock_data import get_stock_info, get_stock_price

class StockAgent:
    def __init__(self):
        self.name = "Stock Data Agent"
    
    def extract_ticker(self, query: str) -> list:
        pattern = r'\b([A-Z]{1,5})\b'
        words = query.upper().split()
        tickers = []
        for word in words:
            if re.match(r'^[A-Z]{1,5}$', word) and len(word) <= 5:
                if word not in ['AND', 'OR', 'THE', 'FOR', 'WITH', 'FROM', 'WHAT', 'HOW', 'WHY', 'WHEN', 'WHO']:
                    tickers.append(word)
        return tickers
    
    def process(self, query: str) -> Dict[str, Any]:
        tickers = self.extract_ticker(query)
        
        if not tickers:
            return {
                "agent": self.name,
                "response": "I couldn't find any stock ticker symbols in your query. Please provide a valid ticker symbol (e.g., AAPL, NVDA, TSLA).",
                "tickers": []
            }
        
        results = {}
        for ticker in tickers:
            info = get_stock_info(ticker)
            if "error" not in info and info.get("current_price"):
                results[ticker] = info
            else:
                results[ticker] = {"error": f"Could not fetch data for {ticker}"}
        
        response = self._format_response(results)
        
        return {
            "agent": self.name,
            "response": response,
            "data": results,
            "tickers": tickers
        }
    
    def _format_response(self, results: Dict) -> str:
        lines = []
        for ticker, info in results.items():
            if "error" in info:
                lines.append(f"**{ticker}**: {info['error']}")
                continue
            
            lines.append(f"## {ticker}")
            if info.get("current_price"):
                lines.append(f"- **Current Price**: ${info['current_price']}")
            if info.get("market_cap"):
                lines.append(f"- **Market Cap**: ${info['market_cap']/1e9:.2f}B")
            if info.get("pe_ratio"):
                lines.append(f"- **P/E Ratio**: {info['pe_ratio']:.2f}")
            if info.get("dividend_yield"):
                lines.append(f"- **Dividend Yield**: {info['dividend_yield']*100:.2f}%")
            if info.get("52w_high") and info.get("52w_low"):
                lines.append(f"- **52-Week Range**: ${info['52w_low']:.2f} - ${info['52w_high']:.2f}")
            if info.get("recommendation"):
                lines.append(f"- **Recommendation**: {info['recommendation'].title()}")
            if info.get("target_mean"):
                lines.append(f"- **Target Price**: ${info['target_mean']:.2f}")
            lines.append("")
        
        return "\n".join(lines)
