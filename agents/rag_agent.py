import os
import chromadb
from chromadb.config import Settings
from typing import Dict, Any, List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import config

class RAGAgent:
    def __init__(self):
        self.name = "RAG Agent"
        self.embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.llm = ChatGroq(model=config.LLM_MODEL, temperature=0.3)
        self.vectorstore = None
        self._init_vectorstore()
    
    def _init_vectorstore(self):
        persist_dir = config.VECTOR_STORE_DIR
        
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
        else:
            self._create_vectorstore()
    
    def _create_vectorstore(self):
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
                chunk_overlap=50
            )
            texts = text_splitter.split_documents(documents)
            
            self.vectorstore = Chroma.from_documents(
                texts,
                self.embeddings,
                persist_directory=config.VECTOR_STORE_DIR
            )
        except Exception as e:
            self.vectorstore = None
    
    def _create_sample_docs(self):
        kb_dir = config.KNOWLEDGE_BASE_DIR
        os.makedirs(kb_dir, exist_ok=True)
        
        docs = [
            ("stock_basics.txt", """Stock Basics: A stock represents ownership in a company. When you buy shares, you own a portion of that company. 
Stock prices fluctuate based on supply and demand. Key metrics include P/E ratio (price to earnings), market cap (total value), and dividend yield.
Companies issue stock to raise capital for growth and operations. Stocks trade on exchanges like NYSE and NASDAQ."""),
            
            ("investment_strategies.txt", """Investment Strategies: 1. Value investing - Find undervalued stocks with strong fundamentals.
2. Growth investing - Focus on companies with high growth potential.
3. Index investing - Buy index funds for diversified market exposure.
4. Dollar-cost averaging - Invest fixed amounts regularly to reduce timing risk.
5. Sector rotation - Move investments between sectors based on economic cycles."""),
            
            ("technical_analysis.txt", """Technical Analysis: RSI (Relative Strength Index) measures momentum. RSI above 70 = overbought, below 30 = oversold.
MACD (Moving Average Convergence Divergence) shows trend direction. Bullish when MACD crosses above signal line.
Bollinger Bands indicate volatility. Price near upper band = potentially overbought, near lower band = potentially oversold.
SMA (Simple Moving Average) smooths price data. 50-day and 200-day SMAs are commonly used for trend analysis."""),
            
            ("risk_management.txt", """Risk Management: Never invest more than you can afford to lose.
Diversify across sectors and asset classes to reduce portfolio risk.
Use stop-loss orders to limit potential losses.
Keep an emergency fund separate from investments.
Rebalance your portfolio periodically to maintain target allocations.
Understand your risk tolerance and invest accordingly."""),
            
            ("market_indicators.txt", """Market Indicators: The S&P 500 tracks 500 large US companies.
The Dow Jones Industrial Average includes 30 major companies.
NASDAQ Composite focuses on tech and growth stocks.
VIX measures market volatility (fear index).
Treasury yields indicate interest rate expectations.
GDP growth shows economic health.
Employment data indicates labor market strength.""")
        ]
        
        for filename, content in docs:
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
    
    def process(self, query: str) -> Dict[str, Any]:
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
                return_source_documents=True
            )
            
            result = qa_chain({"query": query})
            
            sources = []
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    sources.append(doc.metadata.get("source", "Unknown"))
            
            response = result["result"]
            
            if sources:
                response += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in set(sources)])
            
            return {
                "agent": self.name,
                "response": response,
                "sources": list(set(sources))
            }
            
        except Exception as e:
            return {
                "agent": self.name,
                "response": f"Error processing query: {str(e)}",
                "sources": []
            }
