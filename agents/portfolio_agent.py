from typing import Dict, Any, List
import re
from utils.stock_data import get_multiple_stocks
import json

class PortfolioAgent:
    def __init__(self):
        self.name = "Portfolio Agent"
    
    def extract_holdings(self, query: str) -> Dict[str, float]:
        holdings = {}
        
        patterns = [
            r'([A-Z]{1,5})\s*:?\s*(\d+\.?\d*)\s*[%sh]',
            r'([A-Z]{1,5})\s+(\d+\.?\d*)',
            r'holdings?[:\s]+([A-Z,\s]+\d+)',
        ]
        
        words = query.upper().replace(',', ' ').split()
        current_ticker = None
        
        for word in words:
            if re.match(r'^[A-Z]{1,5}$', word) and len(word) <= 5:
                if word not in ['AND', 'OR', 'THE', 'FOR', 'WITH', 'FROM']:
                    current_ticker = word
            elif current_ticker and word.replace('.', '').isdigit():
                try:
                    value = float(word)
                    if value > 0:
                        holdings[current_ticker] = value
                        current_ticker = None
                except:
                    pass
        
        return holdings
    
    def process(self, query: str) -> Dict[str, Any]:
        holdings = self.extract_holdings(query)
        
        if not holdings:
            example_response = """Please provide your portfolio holdings in format like:
- AAPL 10 shares
- NVDA: 5 shares  
- MSFT 20%

Or text like: "My portfolio: AAPL 10, NVDA 5, MSFT 20%"
"""
            return {
                "agent": self.name,
                "response": example_response,
                "holdings": {}
            }
        
        stock_data = get_multiple_stocks(list(holdings.keys()))
        
        total_value = 0
        analysis = []
        
        for ticker, shares in holdings.items():
            info = stock_data.get(ticker, {})
            
            if "error" not in info and info.get("current_price"):
                value = info["current_price"] * shares
                total_value += value
                analysis.append({
                    "ticker": ticker,
                    "shares": shares,
                    "price": info["current_price"],
                    "value": value,
                    "pe_ratio": info.get("pe_ratio"),
                    "recommendation": info.get("recommendation")
                })
        
        response = f"## Portfolio Analysis\n\n"
        response += f"### Holdings\n"
        
        sector_exposure = {}
        
        for item in analysis:
            pct = (item["value"] / total_value * 100) if total_value > 0 else 0
            item["allocation"] = pct
            response += f"- **{item['ticker']}**: {item['shares']} shares @ ${item['price']:.2f} = ${item['value']:.2f} ({pct:.1f}%)\n"
            
            info = stock_data.get(item["ticker"], {})
            sector = info.get("sector", "Unknown")
            if sector not in sector_exposure:
                sector_exposure[sector] = 0
            sector_exposure[sector] += pct
        
        response += f"\n### Total Value: ${total_value:,.2f}\n\n"
        
        response += "### Sector Exposure\n"
        for sector, pct in sector_exposure.items():
            response += f"- **{sector}**: {pct:.1f}%\n"
        
        response += "\n### Recommendations\n"
        
        num_holdings = len(holdings)
        if num_holdings < 5:
            response += "- Consider diversifying with more holdings (5-10 recommended)\n"
        
        sector_count = len(sector_exposure)
        if sector_count < 3:
            response += "- Your portfolio is concentrated in few sectors. Consider diversifying across more sectors.\n"
        
        for item in analysis:
            rec = item.get("recommendation", "none")
            if rec == "strongBuy" or rec == "buy":
                response += f"- **{item['ticker']}**: Strong buy indicator\n"
            elif rec == "sell" or rec == "strongSell":
                response += f"- **{item['ticker']}**: Consider reviewing - sell signal\n"
        
        return {
            "agent": self.name,
            "response": response,
            "holdings": holdings,
            "total_value": total_value,
            "analysis": analysis
        }
