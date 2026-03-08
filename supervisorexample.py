from langgraph.graph import StateGraph, END,START
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import random
load_dotenv()

valid_routes = {'orders', 'billing', 'technical', 'general'}    
llm = ChatOpenAI(model="gpt-4.1-nano")

class MultiAgentState(TypedDict):
    user_request: str # The original user message
    route: str # 'orders' | 'billing' | 'technical' | 'general'
    agent_used: str # Which specialist ran
    specialist_result: str # Raw output from specialist
    final_response: str # Synthesized response for user

@tool
def get_order_status(order_id: str) -> str:
    """ Get order status by order id """
    return f"Order {order_id} is in {random.choice(['pending', 'shipped', 'delivered'])} status"

@tool
def process_return(order_id: str) -> str:
    """ Process return by order id """
    return f"Return {order_id} is processed. Return ID is {random.randint(1000, 9999)}"

@tool
def check_payment_status(order_id: str) -> str:
    """ Check payment status by order id """
    return f"Payment {order_id} is {random.choice(['pending', 'completed', 'failed'])}"


@tool
def issue_refund(order_id: str) -> str:
    """ Issue refund by order id """
    return f"Refund for order {order_id} is issued. Refund ID is REF#{random.randint(1000, 9999)}"

@tool
def create_bug_report(issue: str) -> str:
    """ Create bug report by issue """
    return f"Bug report for issue {issue} is created. Bug ID is JIRA#{random.randint(1000, 9999)}"

@tool
def create_feature_request(feature: str) -> str:
    """ Create feature request by feature """
    return f"Feature request for feature {feature} is created. Feature ID is JIRA#{random.randint(1000, 9999)}"

@tool
def check_inventory(product_id: str) -> str:
    """ Check inventory by product id """
    return f"Inventory {product_id} is {random.choice(['in stock', 'out of stock'])}"

# Node definitions

def supervisor_node(state: MultiAgentState) -> dict:
    """ Supervisor node to route the user request to the appropriate specialist """
    classify_messages = [
        SystemMessage(content="""Classify the request into ONLY one category:
- orders: returns, order status, tracking
- billing: payments, refunds, charges
- technical: app bugs, login issues
- general: everything else
Respond with ONLY the category name. Nothing else."""),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(classify_messages)
    route = response.content.strip().lower()
    return {"route": route if route in valid_routes else "general"}

def order_agent_node(state: MultiAgentState) -> dict:
    """ Specialist agent for order-related queries with order specific tools """
    order_messages = [
        SystemMessage(content="""You are a specialist agent for order-related queries.
You have access to the following tools:
- get_order_status: to get the status of an order
- process_return: to process a return (returns only, no other queries)
"""),
        HumanMessage(content=state["user_request"]),
    ]   
    llm_with_tools = llm.bind_tools([get_order_status, process_return])
    response = llm_with_tools.invoke(order_messages)
    return {"specialist_result": response.content, "agent_used": "order_agent"}

def billing_agent_node(state: MultiAgentState) -> dict:
    """ Specialist agent for billing-related queries with billing specific tools """
    billing_messages = [
        SystemMessage(content="""You are a specialist agent for billing-related queries.
You have access to the following tools: - check_payment_status: to check the payment status of an order - issue_refund: to issue a refund"""),
        HumanMessage(content=state["user_request"]),
    ]
    llm_with_tools = llm.bind_tools([check_payment_status, issue_refund])
    response = llm_with_tools.invoke(billing_messages)
    return {"specialist_result": response.content, "agent_used": "billing_agent"}

def technical_agent_node(state: MultiAgentState) -> dict:
    """ Specialist agent for technical-related queries with technical specific tools """
    technical_messages = [
        SystemMessage(content="""You are a specialist agent for technical-related queries.
You have access to the following tools: - create_bug_report: to create a bug report - create_feature_request: to create a feature request"""),
        HumanMessage(content=state["user_request"]),
    ]
    llm_with_tools = llm.bind_tools([create_bug_report, create_feature_request])
    response = llm_with_tools.invoke(technical_messages)
    return {"specialist_result": response.content, "agent_used": "technical_agent"}

def general_agent_node(state: MultiAgentState) -> dict:
    """ Specialist agent for general-related queries with general specific tools """
    general_messages = [
        SystemMessage(content="""You are a specialist agent for general-related queries.
You have access to the following tools: none"""),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(general_messages)
    return {"final_response": response.content, "agent_used": "general_agent"}

def synthesize_response_node(state: MultiAgentState) -> dict:
    """ Synthesize the response from the specialist agents """
    
    print(f"[synthesize_response_node] Synthesizing response from {state['agent_used']}")
    return {"final_response": state["specialist_result"]}
    
# Build the graph
graph = StateGraph(MultiAgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("order_agent", order_agent_node)
graph.add_node("billing_agent", billing_agent_node)
