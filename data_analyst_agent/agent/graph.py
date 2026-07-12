from langgraph.graph import StateGraph, START, END
from agent.state import AnalystState
from agent.nodes import (
    input_processor_node,
    analyst_router_node,
    code_generator_node,
    execute_code_node,
    code_refiner_node,
    summarizer_reporter_node,
    general_responder_node
)

def router_routing(state: AnalystState) -> str:
    """Routing function for the analyst router node."""
    return state.get("next_step", "general_responder")

def execution_routing(state: AnalystState) -> str:
    """Routing function for the execute code node."""
    if not state.get("error_feedback"):
        return "summarizer_reporter"
    
    # Retry logic
    retry_count = state.get("retry_count", 0)
    if retry_count < 3:
        return "code_refiner"
    
    return "summarizer_reporter"

# Construct the workflow graph
workflow = StateGraph(AnalystState)

# Add nodes
workflow.add_node("input_processor", input_processor_node)
workflow.add_node("analyst_router", analyst_router_node)
workflow.add_node("code_generator", code_generator_node)
workflow.add_node("execute_code", execute_code_node)
workflow.add_node("code_refiner", code_refiner_node)
workflow.add_node("summarizer_reporter", summarizer_reporter_node)
workflow.add_node("general_responder", general_responder_node)

# Set entry point and processor connection
workflow.add_edge(START, "input_processor")
workflow.add_edge("input_processor", "analyst_router")

# Router routing
workflow.add_conditional_edges(
    "analyst_router",
    router_routing,
    {
        "code_generator": "code_generator",
        "general_responder": "general_responder"
    }
)

# Execute code
workflow.add_edge("code_generator", "execute_code")

# Verify execution result and route
workflow.add_conditional_edges(
    "execute_code",
    execution_routing,
    {
        "summarizer_reporter": "summarizer_reporter",
        "code_refiner": "code_refiner"
    }
)

# Refiner goes back to execute
workflow.add_edge("code_refiner", "execute_code")

# Final reports go to END
workflow.add_edge("summarizer_reporter", END)
workflow.add_edge("general_responder", END)

# Compile the final executable LangGraph app
app = workflow.compile()
