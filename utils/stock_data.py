import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any

def get_stock_info(ticker: str) -> Dict[str, Any]:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "volume": info.get("volume"),
            "avg_volume": info.get("averageVolume"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "beta": info.get("beta"),
            "recommendation": info.get("recommendationKey"),
            "target_mean": info.get("targetMeanPrice"),
        }
    except Exception as e:
        return {"error": str(e)}

def get_stock_history(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception:
        return None

def get_stock_price(ticker: str) -> Optional[float]:
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get("currentPrice", stock.info.get("regularMarketPrice"))
    except Exception:
        return None

def get_multiple_stocks(tickers: list) -> Dict[str, Dict]:
    result = {}
    for ticker in tickers:
        info = get_stock_info(ticker)
        result[ticker] = info
    return result
