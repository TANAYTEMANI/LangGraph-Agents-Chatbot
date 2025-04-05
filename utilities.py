from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langchain_core.tracers import LangChainTracer
from langchain_core.callbacks import CallbackManager
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()



# llm = ChatGroq(
#     api_key=os.getenv("GROQ_KEY"),  # or use os.environ["GROQ_API_KEY"]
#     model_name="llama-3.3-70b-versatile",  # Or any other Groq-supported model
# )

# os.environ["AZURE_OPENAI_API_KEY"] = "331b74aa95dc4614ba1d3d0833c6d1b3"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://h3251433-ai.openai.azure.com/"

 


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


# LangSmith & Azure OpenAI
def setup_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    os.environ["LANGCHAIN_PROJECT"] = "WeatherRag"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_API_KEY"] = (
        "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"
    )
    tracer = LangChainTracer(project_name="WeatherRag")
    callback_manager = CallbackManager([tracer])
    return tracer, callback_manager


tracer, callback_manager = setup_langsmith()


llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",  # or your deployment
    api_version="2024-08-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    callbacks = [tracer]
)