"""
RAG (Retrieval-Augmented Generation) Agent

This agent handles knowledge-based queries using ChromaDB vector store
for semantic search and Groq LLM for response generation.
"""
import os
import logging
from typing import Dict, Any, List
from functools import lru_cache

import chromadb
from chromadb.config import Settings

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG Agent for financial knowledge Q&A.
    
    Uses ChromaDB vector store with sentence-transformer embeddings
    to retrieve relevant context from knowledge base documents.
    """
    
    def __init__(self):
        self.name = "RAG Agent"
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )
        self.llm = ChatGroq(
            model=config.LLM_MODEL,
            temperature=0.3,
            max_tokens=1024
        )
        self.vectorstore = None
        self._init_vectorstore()
    
    def _init_vectorstore(self) -> None:
        """Initialize vector store from persistent storage or create new."""
        persist_dir = config.VECTOR_STORE_DIR
        
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            logger.info("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
        else:
            logger.info("Creating new vector store...")
            self._create_vectorstore()
    
    def _create_vectorstore(self) -> None:
        """Create vector store from knowledge base documents."""
        kb_dir = config.KNOWLEDGE_BASE_DIR
        
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir, exist_ok=True)
            self._create_sample_docs()
        
        try:
            loader = DirectoryLoader(kb_dir, glob="*.txt")
            documents = loader.load()
            
            if not documents:
                self._create_sample_docs()
                loader = DirectoryLoader(kb_dir, glob="*.txt")
                documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            
            logger.info(f"Indexing {len(texts)} text chunks...")
            
            self.vectorstore = Chroma.from_documents(
                texts,
                self.embeddings,
                persist_directory=config.VECTOR_STORE_DIR
            )
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            self.vectorstore = None
    
    def _create_sample_docs(self) -> None:
        """
        Create comprehensive knowledge base documents covering:
        - Stock fundamentals
        - Investment strategies
        - Technical analysis
        - Risk management
        - Market indicators
        - Financial metrics
        """
        kb_dir = config.KNOWLEDGE_BASE_DIR
        os.makedirs(kb_dir, exist_ok=True)
        
        docs = [
            ("01_stock_fundamentals.txt", """Stock Fundamentals: A stock represents partial ownership in a company. 
When you purchase shares, you become a shareholder entitled to company profits and voting rights.

Key Stock Metrics:
- Market Capitalization (Market Cap): Total value of all shares. Calculated as share price times number of outstanding shares.
- P/E Ratio (Price-to-Earnings): Stock price divided by earnings per share. Indicates how much investors pay per dollar of earnings.
- EPS (Earnings Per Share): Company's profit divided by outstanding shares. Higher is generally better.
- Dividend Yield: Annual dividend payment divided by stock price. Expressed as percentage.
- 52-Week High/Low: Highest and lowest prices in the past year.

Stock exchanges include NYSE (New York Stock Exchange) and NASDAQ. Companies list on exchanges to raise capital for growth, acquisitions, and operations."""),

            ("02_investment_strategies.txt", """Investment Strategies:

1. VALUE INVESTING:
   - Focus on undervalued stocks trading below intrinsic value
   - Look for low P/E and P/B ratios
   - Requires fundamental analysis of financial statements
   - Famous proponents: Warren Buffett, Benjamin Graham

2. GROWTH INVESTING:
   - Invest in companies with high growth potential
   - Focus on revenue and earnings growth rates
   - Accept higher valuations for high-potential stocks
   - Typically technology and emerging market companies

3. INDEX INVESTING:
   - Buy index funds tracking S&P 500, Total Market, etc.
   - Provides broad diversification at low cost
   - Beats most active investors over time
   - Examples: VOO, SPY, VTI

4. DOLLAR-COST AVERAGING (DCA):
   - Invest fixed dollar amount at regular intervals
   - Reduces impact of market volatility
   - Removes emotional decision-making
   - Ideal for long-term investors

5. SECTOR ROTATION:
   - Move investments between sectors based on economic cycles
   - Rotate to defensive sectors (utilities, healthcare) in downturns
   - Rotate to cyclical sectors (tech, consumer) in expansions"""),

            ("03_technical_indicators.txt", """Technical Analysis Indicators:

1. RSI (Relative Strength Index):
   - Momentum oscillator measuring speed/change of price movements
   - Scale: 0-100
   - RSI above 70 = Overbought (potential sell signal)
   - RSI below 30 = Oversold (potential buy signal)
   - Standard period: 14 days

2. MACD (Moving Average Convergence Divergence):
   - Trend-following momentum indicator
   - MACD Line = 12-period EMA - 26-period EMA
   - Signal Line = 9-period EMA of MACD Line
   - Bullish signal: MACD crosses above signal line
   - Bearish signal: MACD crosses below signal line

3. BOLLINGER BANDS:
   - Volatility bands above and below moving average
   - Upper band = SMA + (2 x standard deviation)
   - Lower band = SMA - (2 x standard deviation)
   - Price near upper band: potentially overbought
   - Price near lower band: potentially oversold

4. MOVING AVERAGES:
   - SMA (Simple Moving Average): Equal weight to all periods
   - EMA (Exponential Moving Average): More weight to recent prices
   - 50-day SMA: Medium-term trend indicator
   - 200-day SMA: Long-term trend indicator
   - Golden Cross: 50-day crosses above 200-day (bullish)
   - Death Cross: 50-day crosses below 200-day (bearish)"""),

            ("04_risk_management.txt", """Risk Management Principles:

1. POSITION SIZING:
   - Never allocate more than 2-5% to a single stock
   - Reduces impact of any single position losing value
   - Adjust position sizes based on conviction level

2. DIVERSIFICATION:
   - Spread investments across sectors and asset classes
   - Don't put all eggs in one basket
   - Correlated assets move together - diversify wisely
   - Target 20-30 stocks for adequate diversification

3. STOP-LOSS ORDERS:
   - Automatically sell when stock falls below threshold
   - Limits potential losses on any position
   - Typical stop-loss: 7-10% below purchase price

4. REBALANCING:
   - Periodically review and adjust portfolio allocations
   - Sell winners, buy laggards to maintain targets
   - Recommended frequency: Quarterly or annually

5. EMERGENCY FUND:
   - Keep 3-6 months expenses in liquid savings
   - Don't invest money needed for short-term expenses
   - Separate emergency fund from investment portfolio

6. RISK TOLERANCE:
   - Understand your investment timeline
   - Young investors can take more risk
   - Near retirement = more conservative allocation"""),

            ("05_market_indicators.txt", """Market Indicators and Indices:

MAJOR INDICES:
- S&P 500: 500 large US companies, market cap weighted
- Dow Jones Industrial Average: 30 major companies, price weighted
- NASDAQ Composite: All NASDAQ stocks, tech-heavy
- Russell 2000: Small-cap stocks index

ECONOMIC INDICATORS:
- GDP (Gross Domestic Product): Total economic output
- Unemployment Rate: Percentage of labor force without jobs
- Inflation (CPI): Consumer price changes over time
- Interest Rates: Federal Reserve policy rate
- Treasury Yields: Government debt returns

MARKET SENTIMENT:
- VIX (CBOE Volatility Index): "Fear Index" - measures market volatility
- High VIX = fear/uncertainty, low VIX = complacency
- Put/Call Ratio: Options activity indicating sentiment

SECTOR INDICES:
- Technology (XLK)
- Healthcare (XLV)
- Financials (XLF)
- Energy (XLE)
- Consumer Discretionary (XLY)"""),

            ("06_financial_metrics.txt", """Financial Metrics and Ratios:

VALUATION METRICS:
- P/E Ratio: Price / Earnings per Share
  - Forward P/E uses projected earnings
  - Relative P/E compares to industry average
  
- P/B Ratio: Price / Book Value per Share
  - Below 1 = potentially undervalued
  - Industry comparison important

- EV/EBITDA: Enterprise Value / Earnings Before Interest, Taxes, Depreciation
  - Useful for comparing companies with different debt levels

PROFITABILITY METRICS:
- ROE (Return on Equity): Net Income / Shareholder's Equity
- ROA (Return on Assets): Net Income / Total Assets
- Gross Margin: (Revenue - COGS) / Revenue
- Operating Margin: Operating Income / Revenue

GROWTH METRICS:
- Revenue Growth: Year-over-year revenue increase
- EPS Growth: Year-over-year earnings growth
- PEG Ratio: P/E Ratio / Expected Earnings Growth
  - PEG < 1 = potentially undervalued
  - PEG > 2 = potentially overvalued

LIQUIDITY METRICS:
- Current Ratio: Current Assets / Current Liabilities (>1.5 is good)
- Quick Ratio: (Current Assets - Inventory) / Current Liabilities
"""),

            ("07_trading_psychology.txt", """Trading Psychology and Behavioral Finance:

COMMON BIASES:
1. LOSS AVERSION: Pain of losing is 2x stronger than pleasure of winning
2. CONFIRMATION BIAS: Seeking information that confirms existing beliefs
3. HERD MENTALLY: Following the crowd, especially at extremes
4. OVERCONFIDENCE: Overestimating ability to predict market
5. ANCHORING: Fixating on specific price points

AVOID THESE MISTAKES:
- Chasing stocks that have already risen significantly
- Panic selling during market downturns
- Trying to time the market (buy low, sell high)
- Trading too frequently (costs add up)
- Ignoring diversification

GOOD PRACTICES:
- Have a written investment plan
- Stick to your strategy during volatility
- Review portfolio regularly but don't overreact
- Keep emotions out of decisions
- Learn from mistakes, move on

MARKET MYTHS TO IGNORE:
- "The market is always right"
- "This time is different"
- "I need to do something"
- "Everyone else is making money"
""),
        ]
        
        for filename, content in docs:
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"Created {len(docs)} knowledge base documents")
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Process user query using RAG pipeline.
        
        Args:
            query: User question about stocks or investing
            
        Returns:
            Dictionary with agent response and source citations
        """
        if not self.vectorstore:
            self._create_vectorstore()
        
        if not self.vectorstore:
            return {
                "agent": self.name,
                "response": "Knowledge base not available. Please ensure documents exist in the knowledge base directory.",
                "sources": []
            }
        
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                verbose=False
            )
            
            result = qa_chain({"query": query})
            
            sources = []
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "Unknown")
                    if source:
                        sources.append(os.path.basename(source))
            
            response = result["result"]
            
            if sources:
                response += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in set(sources)])
            
            logger.info(f"RAG query processed: {query[:30]}...")
            
            return {
                "agent": self.name,
                "response": response,
                "sources": list(set(sources))
            }
            
        except Exception as e:
            logger.error(f"RAG query error: {str(e)}")
            return {
                "agent": self.name,
                "response": f"I encountered an error searching my knowledge base: {str(e)}",
                "sources": []
            }
