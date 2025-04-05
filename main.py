from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from utilities import create_tool_node_with_fallback
from langchain_core.messages import AIMessage
from state import State
import os
from agents import part_1_tools ,part_1_assistant_runnable, Assistant
from utilities import tracer



#Logging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


#User Graph
builder = StateGraph(State)
builder.add_node("user", Assistant(part_1_assistant_runnable, tracer))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))

builder.add_edge(START, "user")
builder.add_conditional_edges(
    "user",
    tools_condition,
)
builder.add_edge("tools", "user")

memory = MemorySaver()
user_graph = builder.compile(checkpointer = memory)



def process_user_query(messages: str, thread_id: str, vector_store) -> dict:
    """
    Process the user query using the compiled user graph.

    Args:
        user_query (str): The query from the user.
        thread_id (str): The unique session ID.

    Returns:
        dict: A response object containing assistant message.
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "vector_store": vector_store
        },
        "callbacks":[tracer]
    }
    print("Thread Id", thread_id)

    # logger.info(f"User State: {user_state}")
    # logger.info(f"Config: {config}")

    memory.get(config=config)

    result = user_graph.invoke({"messages": messages}, config=config)
    # memory.save(config=config)

    logger.info(f"Result: {result}")

    return result