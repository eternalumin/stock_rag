from typing import TypedDict, Sequence
from langgraph.graph import StateGraph, END
from .stock_agent import StockAgent
from .rag_agent import RAGAgent
from .analysis_agent import AnalysisAgent
from .portfolio_agent import PortfolioAgent
import re

class AgentState(TypedDict):
    query: str
    intent: str
    response: str
    agent_used: str

def classify_intent(query: str) -> str:
    query_lower = query.lower()
    
    portfolio_keywords = ['portfolio', 'holdings', 'hold', 'shares', 'allocation', 'diversify', 'my stocks']
    if any(kw in query_lower for kw in portfolio_keywords):
        return "portfolio"
    
    analysis_keywords = ['analyze', 'technical', 'indicator', 'rsi', 'macd', 'sma', 'trend', 'chart']
    if any(kw in query_lower for kw in analysis_keywords):
        return "analysis"
    
    stock_keywords = ['price', 'value', 'market cap', 'pe ratio', 'dividend', 'earnings', 'recommendation', 'target']
    if any(kw in query_lower for kw in stock_keywords):
        return "stock"
    
    knowledge_keywords = ['what is', 'explain', 'how does', 'define', 'meaning', 'strategy', 'investing', 'basics']
    if any(kw in query_lower for kw in knowledge_keywords):
        return "rag"
    
    if re.search(r'\b[a-z]{1,5}\b', query) and not re.search(r'\b[A-Z]{2,}\b', query):
        return "rag"
    
    return "stock"

def stock_node(state: AgentState) -> AgentState:
    agent = StockAgent()
    result = agent.process(state["query"])
    state["response"] = result.get("response", "No response")
    state["agent_used"] = agent.name
    return state

def rag_node(state: AgentState) -> AgentState:
    agent = RAGAgent()
    result = agent.process(state["query"])
    state["response"] = result.get("response", "No response")
    state["agent_used"] = agent.name
    return state

def analysis_node(state: AgentState) -> AgentState:
    agent = AnalysisAgent()
    result = agent.process(state["query"])
    state["response"] = result.get("response", "No response")
    state["agent_used"] = agent.name
    return state

def portfolio_node(state: AgentState) -> AgentState:
    agent = PortfolioAgent()
    result = agent.process(state["query"])
    state["response"] = result.get("response", "No response")
    state["agent_used"] = agent.name
    return state

def router(state: AgentState) -> str:
    return state["intent"]

def create_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("stock", stock_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("portfolio", portfolio_node)
    
    workflow.set_entry_point("classify")
    
    workflow.add_conditional_edges(
        "classify",
        router,
        {
            "stock": "stock",
            "rag": "rag", 
            "analysis": "analysis",
            "portfolio": "portfolio"
        }
    )
    
    workflow.add_edge("stock", END)
    workflow.add_edge("rag", END)
    workflow.add_edge("analysis", END)
    workflow.add_edge("portfolio", END)
    
    return workflow.compile()

def run_agent(query: str) -> dict:
    workflow = create_workflow()
    
    intent = classify_intent(query)
    
    initial_state = {
        "query": query,
        "intent": intent,
        "response": "",
        "agent_used": ""
    }
    
    result = workflow.invoke(initial_state)
    
    return {
        "response": result["response"],
        "intent": result["intent"],
        "agent": result["agent_used"]
    }
