# StockMind 🤖📈

> AI-Powered Stock Analysis Assistant with Multi-Agent RAG Architecture

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-LLM%20Agents-yellow.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203-orange.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A production-ready **Retrieval-Augmented Generation (RAG)** application demonstrating advanced AI agent orchestration, real-time financial data integration, and interactive data visualization.

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Orchestration** | LangGraph-powered workflow with intelligent intent classification routing to 4 specialized agents |
| **Real-Time Market Data** | Live stock prices, fundamentals, and analyst recommendations via yfinance API |
| **Technical Analysis** | RSI, MACD, Bollinger Bands, SMA/EMA, ATR with interactive Plotly charts |
| **RAG Q&A System** | ChromaDB vector store with sentence-transformer embeddings for financial knowledge |
| **Portfolio Analytics** | Holdings analysis, sector exposure, diversification recommendations |
| **Production Patterns** | Caching, error handling, type hints, modular architecture |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    📊 INTENT CLASSIFIER NODE                            │
│         (Routes query to appropriate specialized agent)                 │
└─────────────────────────────────────────────────────────────────────────┘
            │                    │                    │                    │
            ▼                    ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │   STOCK     │      │    RAG      │      │  ANALYSIS   │      │ PORTFOLIO   │
   │   AGENT     │      │   AGENT     │      │   AGENT     │      │   AGENT     │
   │             │      │             │      │             │      │             │
   │ • Price     │      │ • P/E ratio │      │ • RSI/MACD  │      │ • Holdings  │
   │ • Market cap│      │ • Strategies│      │ • Bollinger │      │ • Allocation│
   │ • Dividends │      │ • Investing │      │ • Trends    │      │ • Risk      │
   └─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
            │                    │                    │                    │
            └────────────────────┴────────────────────┴────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            FINAL RESPONSE                                │
│              (Formatted with sources, citations, charts)                │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                    │
│                        Streamlit + Plotly                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH ORCHESTRATION                         │
│                    Intent Classification → Agent Routing                │
└─────────────────────────────────────────────────────────────────────────┘
            │                    │                    │                    │
            ▼                    ▼                    ▼                    ▼
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │   yfinance API  │  │   ChromaDB +    │  │  Technical     │  │  Holdings      │
   │   (Real-time)  │  │   RAG Pipeline  │  │  Indicators    │  │  Calculator    │
   └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            BACKEND                                       │
│              Groq (LLaMA-3-70B) + Sentence-Transformers                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Google Colab (Recommended for Quick Demo)

```python
# Step 1: Install dependencies
!pip install streamlit langchain langgraph langchain-groq chromadb yfinance sentence-transformers plotly ta

# Step 2: Set Groq API key
import os
os.environ["GROQ_API_KEY"] = "gsk_your_api_key"

# Step 3: Run Streamlit with tunneling
!streamlit run app.py & npx localtunnel --port 8501
```

### Local Development

```bash
# Clone repository
git clone https://github.com/eternalumin/stock_rag.git
cd stock_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export GROQ_API_KEY="your_groq_api_key"

# Run application
streamlit run app.py
```

> **Get Free API Key**: Visit [groq.com](https://console.groq.com/) for a free API key

## 💬 Example Queries

| Query Type | Example |
|------------|---------|
| **Stock Data** | "What's the current price of AAPL?" |
| **Technical Analysis** | "Analyze NVDA technically, show RSI and MACD" |
| **Fundamentals** | "What's the P/E ratio of Microsoft?" |
| **RAG Knowledge** | "Explain what P/E ratio means for beginners" |
| **Portfolio** | "My portfolio: AAPL 10 shares, NVDA 5, MSFT 20%" |
| **Strategies** | "What is dollar-cost averaging?" |

## 📁 Project Structure

```
stock_rag/
├── app.py                        # Main Streamlit UI application
├── config.py                     # Configuration & API settings
├── requirements.txt              # Python dependencies with versions
├── Dockerfile                    # Container configuration
├── .gitignore                    # Git ignore patterns
├── README.md                     # This file
├── agents/
│   ├── __init__.py              # Agent package exports
│   ├── coordinator.py           # LangGraph workflow orchestration
│   ├── stock_agent.py           # Real-time market data retrieval
│   ├── rag_agent.py             # RAG pipeline with ChromaDB
│   ├── analysis_agent.py        # Technical indicators computation
│   └── portfolio_agent.py       # Portfolio holdings analysis
├── utils/
│   ├── __init__.py              # Utils package exports
│   ├── stock_data.py            # yfinance wrapper functions
│   └── indicators.py            # Technical analysis calculations
└── data/
    └── knowledge_base/          # RAG document storage
```

## 🎓 Skills Demonstrated

- ✅ **LLM Integration** - Groq API, prompt engineering, response parsing
- ✅ **RAG Architecture** - Vector databases, embeddings, retrieval chains
- ✅ **Agent Systems** - LangGraph state machines, tool orchestration
- ✅ **Financial Data** - APIs, real-time data, market fundamentals
- ✅ **Technical Analysis** - Trading indicators, pattern recognition
- ✅ **Data Visualization** - Interactive charts with Plotly
- ✅ **Production Engineering** - Error handling, caching, type hints
- ✅ **Cloud Deployment** - Docker, Streamlit sharing, Colab

## 🔧 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM | Required |
| `LLM_MODEL` | LLM model name | `llama-3-70b-8192` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |

## 📊 Demo Screenshots

The application features:
- Chat interface with agent attribution
- Interactive stock price candlestick charts
- Technical indicator visualizations (RSI, MACD, Bollinger Bands)
- Portfolio allocation pie charts
- Knowledge base Q&A with source citations

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Author

**StockMind** - Built with ❤️ using LangGraph, Streamlit, and Groq

---

<p align="center">
  <sub>⭐ Star this repo if you found it useful!</sub>
</p>
