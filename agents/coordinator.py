"""
LangGraph Workflow Coordinator

This module orchestrates multiple specialized agents using LangGraph's
state machine approach for intelligent query routing.
"""
import logging
from typing import TypedDict, Optional
from functools import lru_cache

from langgraph.graph import StateGraph, END

from .stock_agent import StockAgent
from .rag_agent import RAGAgent
from .analysis_agent import AnalysisAgent
from .portfolio_agent import PortfolioAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for LangGraph workflow."""
    query: str
    intent: str
    response: str
    agent_used: str
    error: Optional[str]


def classify_intent(query: str) -> str:
    """
    Classify user query into appropriate intent category.
    
    Routes to:
    - portfolio: holdings, allocation, diversification
    - analysis: technical indicators, charts, trends
    - stock: prices, market cap, fundamentals
    - rag: general knowledge, explanations, strategies
    """
    query_lower = query.lower()
    
    portfolio_keywords = [
        'portfolio', 'holdings', 'hold', 'shares', 
        'allocation', 'diversify', 'my stocks', 'my portfolio'
    ]
    if any(kw in query_lower for kw in portfolio_keywords):
        logger.debug("Intent classified: portfolio")
        return "portfolio"
    
    analysis_keywords = [
        'analyze', 'technical', 'indicator', 'rsi', 'macd', 
        'sma', 'trend', 'chart', 'analysis'
    ]
    if any(kw in query_lower for kw in analysis_keywords):
        logger.debug("Intent classified: analysis")
        return "analysis"
    
    stock_keywords = [
        'price', 'value', 'market cap', 'pe ratio', 'dividend', 
        'earnings', 'recommendation', 'target', 'current'
    ]
    if any(kw in query_lower for kw in stock_keywords):
        logger.debug("Intent classified: stock")
        return "stock"
    
    knowledge_keywords = [
        'what is', 'explain', 'how does', 'define', 'meaning', 
        'strategy', 'investing', 'basics', 'learn about'
    ]
    if any(kw in query_lower for kw in knowledge_keywords):
        logger.debug("Intent classified: rag")
        return "rag"
    
    logger.debug("Intent classified: stock (default)")
    return "stock"


def stock_node(state: AgentState) -> AgentState:
    """Process query through Stock Data Agent."""
    try:
        agent = StockAgent()
        result = agent.process(state["query"])
        state["response"] = result.get("response", "No response")
        state["agent_used"] = agent.name
        logger.info(f"Stock Agent processed query: {state['intent']}")
    except Exception as e:
        logger.error(f"Stock Agent error: {str(e)}")
        state["error"] = str(e)
        state["response"] = f"I encountered an error fetching stock data: {str(e)}"
    return state


def rag_node(state: AgentState) -> AgentState:
    """Process query through RAG Agent for knowledge base queries."""
    try:
        agent = RAGAgent()
        result = agent.process(state["query"])
        state["response"] = result.get("response", "No response")
        state["agent_used"] = agent.name
        logger.info(f"RAG Agent processed query: {state['intent']}")
    except Exception as e:
        logger.error(f"RAG Agent error: {str(e)}")
        state["error"] = str(e)
        state["response"] = f"I encountered an error searching the knowledge base: {str(e)}"
    return state


def analysis_node(state: AgentState) -> AgentState:
    """Process query through Technical Analysis Agent."""
    try:
        agent = AnalysisAgent()
        result = agent.process(state["query"])
        state["response"] = result.get("response", "No response")
        state["agent_used"] = agent.name
        logger.info(f"Analysis Agent processed query: {state['intent']}")
    except Exception as e:
        logger.error(f"Analysis Agent error: {str(e)}")
        state["error"] = str(e)
        state["response"] = f"I encountered an error performing technical analysis: {str(e)}"
    return state


def portfolio_node(state: AgentState) -> AgentState:
    """Process query through Portfolio Analysis Agent."""
    try:
        agent = PortfolioAgent()
        result = agent.process(state["query"])
        state["response"] = result.get("response", "No response")
        state["agent_used"] = agent.name
        logger.info(f"Portfolio Agent processed query: {state['intent']}")
    except Exception as e:
        logger.error(f"Portfolio Agent error: {str(e)}")
        state["error"] = str(e)
        state["response"] = f"I encountered an error analyzing your portfolio: {str(e)}"
    return state


def router(state: AgentState) -> str:
    """Route to appropriate agent based on classified intent."""
    return state["intent"]


def create_workflow():
    """
    Create LangGraph state machine workflow.
    
    Flow:
    1. Intent Classifier determines query type
    2. Conditional routing to specialized agent
    3. Agent processes query and returns response
    4. End state with formatted response
    """
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


@lru_cache(maxsize=1)
def get_workflow():
    """Cache compiled workflow for performance."""
    return create_workflow()


def run_agent(query: str) -> dict:
    """
    Main entry point for agent processing.
    
    Args:
        query: User input string
        
    Returns:
        Dictionary with response, intent classification, and agent used
    """
    logger.info(f"Processing query: {query[:50]}...")
    
    workflow = get_workflow()
    
    intent = classify_intent(query)
    
    initial_state = {
        "query": query,
        "intent": intent,
        "response": "",
        "agent_used": "",
        "error": None
    }
    
    result = workflow.invoke(initial_state)
    
    logger.info(f"Query processed by {result['agent_used']}")
    
    return {
        "response": result["response"],
        "intent": result["intent"],
        "agent": result["agent_used"],
        "error": result.get("error")
    }
