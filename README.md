# StockMind - AI Stock Analysis Assistant

A powerful stock RAG (Retrieval-Augmented Generation) application that combines real-time stock data, technical analysis, and knowledge-base Q&A using LangGraph, Groq, and Streamlit.

## Features

- **🤖 Multi-Agent System**: LangGraph-powered orchestration of specialized agents
- **📊 Real-Time Stock Data**: Live prices, fundamentals, and analyst recommendations via yfinance
- **📈 Technical Analysis**: RSI, MACD, Bollinger Bands, SMA/EMA indicators
- **💬 RAG-Powered Q&A**: Ask questions about investing, stocks, and strategies
- **💼 Portfolio Analysis**: Analyze your holdings and get diversification insights

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq (Llama-3-70B) |
| Agents | LangGraph |
| UI | Streamlit |
| Stock Data | yfinance |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers |

## Getting Started

### Option 1: Google Colab (Recommended)

1. **Create a new Colab notebook**

2. **Install dependencies**:
   ```python
   !pip install streamlit langchain langgraph langchain-groq chromadb yfinance sentence-transformers plotly ta
   ```

3. **Set your Groq API Key**:
   - Get a free API key from [https://console.groq.com/](https://console.groq.com/)
   - Set it in the sidebar or as environment variable:
   ```python
   import os
   os.environ["GROQ_API_KEY"] = "your-api-key-here"
   ```

4. **Run Streamlit**:
   ```python
   !streamlit run app.py & npx localtunnel --port 8501
   ```

5. **Access the app**: Click the localtunnel URL provided

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/eternalumin/stock_rag.git
cd stock_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
export GROQ_API_KEY="your-api-key-here"

# Run
streamlit run app.py
```

## Project Structure

```
stock_rag/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration and API keys
├── requirements.txt          # Python dependencies
├── agents/
│   ├── coordinator.py        # LangGraph workflow
│   ├── stock_agent.py        # Real-time stock data
│   ├── rag_agent.py          # Document Q&A
│   ├── analysis_agent.py     # Technical analysis
│   └── portfolio_agent.py    # Portfolio insights
├── utils/
│   ├── stock_data.py         # yfinance helpers
│   └── indicators.py         # Technical indicators
└── data/
    └── knowledge_base/       # RAG documents
```

## Example Questions

- "What's the price of AAPL?"
- "Analyze NVDA technically"
- "Explain P/E ratio"
- "My portfolio: AAPL 10, NVDA 5, MSFT 20"
- "What is dollar-cost averaging?"

## Architecture

```
User Query → Intent Classifier → [Stock Agent | RAG Agent | Analysis Agent | Portfolio Agent] → Response
```

The LangGraph workflow routes queries to the appropriate specialized agent based on intent classification.

## License

MIT License
