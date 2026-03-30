import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from agents.coordinator import run_agent, classify_intent
from utils.stock_data import get_stock_history, get_stock_info
from utils.indicators import calculate_indicators
import config
import os

st.set_page_config(
    page_title="StockMind - AI Stock Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockMind - AI Stock Analysis Assistant")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get your free API key from https://console.groq.com/",
        value=config.GROQ_API_KEY if config.GROQ_API_KEY != "gsk_garbage_key_replace_later" else ""
    )
    
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        config.GROQ_API_KEY = api_key
        st.session_state.api_key_set = True
        st.success("API Key configured!")
    else:
        st.warning("Please enter your Groq API key")
        st.session_state.api_key_set = False
    
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    examples = [
        "What's the price of AAPL?",
        "Explain P/E ratio",
        "Analyze NVDA",
        "My portfolio: AAPL 10, NVDA 5, MSFT 20"
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.input_query = ex

st.markdown("### 💬 Ask about stocks")

if hasattr(st.session_state, 'input_query'):
    default_value = st.session_state.input_query
    del st.session_state.input_query
else:
    default_value = ""

query = st.text_input(
    "Ask a question:",
    placeholder="e.g., What's AAPL price? or Explain stock dividends",
    value=default_value,
    label_visibility="collapsed"
)

col1, col2 = st.columns([6, 1])
with col2:
    ask_button = st.button("Ask", type="primary")

if ask_button and query:
    with st.spinner("Analyzing..."):
        try:
            result = run_agent(query)
            
            st.session_state.messages.append({
                "role": "user",
                "content": query
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "agent": result["agent"]
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure your Groq API key is correct!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "agent" in message:
            st.caption(f"Powered by: {message['agent']}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Stock Charts", "📈 Technical Analysis", "📚 Knowledge Base"])

with tab1:
    st.subheader("Stock Price Charts")
    
    ticker_input = st.text_input("Enter ticker symbol:", value="AAPL").upper()
    
    if st.button("Load Chart"):
        if ticker_input:
            df = get_stock_history(ticker_input, period="1y")
            if df is not None and not df.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=ticker_input
                ))
                fig.update_layout(
                    title=f"{ticker_input} Stock Price",
                    yaxis_title="Price ($)",
                    xaxis_title="Date",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Quick Stats")
                info = get_stock_info(ticker_input)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Price", f"${info.get('current_price', 'N/A')}")
                with c2:
                    st.metric("P/E Ratio", f"{info.get('pe_ratio', 'N/A')}")
                with c3:
                    st.metric("Market Cap", f"${info.get('market_cap', 0)/1e9:.2f}B" if info.get('market_cap') else "N/A")
                with c4:
                    st.metric("Recommendation", f"{info.get('recommendation', 'N/A')}")
            else:
                st.error(f"Could not load data for {ticker_input}")

with tab2:
    st.subheader("Technical Indicators")
    
    ticker_input2 = st.text_input("Enter ticker for analysis:", value="AAPL", key="ticker2").upper()
    
    if st.button("Analyze", key="analyze_btn"):
        if ticker_input2:
            df = get_stock_history(ticker_input2, period="6mo")
            if df is not None and not df.empty:
                df_indicators = calculate_indicators(df)
                latest = df_indicators.iloc[-1]
                
                st.subheader(f"Technical Summary for {ticker_input2}")
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("RSI (14)", f"{latest['RSI']:.2f}")
                with c2:
                    st.metric("MACD", f"{latest['MACD']:.2f}")
                with c3:
                    st.metric("SMA 50", f"${latest['SMA_50']:.2f}")
                with c4:
                    st.metric("ATR", f"${latest['ATR']:.2f}")
                
                st.subheader("Price with SMA")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['Close'], name="Price", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['SMA_20'], name="SMA 20", line=dict(color="orange")))
                fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['SMA_50'], name="SMA 50", line=dict(color="green")))
                fig.update_layout(title=f"{ticker_input2} Price with Moving Averages", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("RSI Chart")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['RSI'], name="RSI", line=dict(color="purple")))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(title="RSI (14)", yaxis_range=[0, 100], template="plotly_dark")
                st.plotly_chart(fig_rsi, use_container_width=True)
            else:
                st.error(f"Could not analyze {ticker_input2}")

with tab3:
    st.subheader("📚 Stock Knowledge Base")
    st.markdown("""
    This knowledge base contains information about:
    - Stock basics and fundamentals
    - Investment strategies
    - Technical analysis indicators
    - Risk management
    - Market indicators
    """)
    
    st.info("💡 Try asking questions like:")
    st.markdown("""
    - What is P/E ratio?
    - How does RSI work?
    - What is dollar-cost averaging?
    - Explain value investing
    """)

st.markdown("---")
st.caption("StockMind - Powered by LangGraph, Groq, and Streamlit")
