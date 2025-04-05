import datetime
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import Runnable, RunnableConfig
import tools

from state import State
from utilities import llm, tracer


llm = llm


class Assistant:
    def __init__(self, runnable: Runnable, tracer):
        self.runnable = runnable
        self.tracer = tracer

    def __call__(self, state: State, config: RunnableConfig):
        for _ in range(3):
            while True:
                config["callbacks"] = [self.tracer]
                result = self.runnable.invoke(state)
                if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
                ):
                    messages = state["messages"] + [
                        ("user", "Respond with a real output.")
                    ]
                    state = {**state, "messages": messages}
                else:
                    break
            return {"messages": result}


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " Use the provided tools to generate the response."
            " End the cycle if you feel the Tool has successfully given its answer."
            " Do not ask follow-up questions."
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.datetime.now)


part_1_tools = [tools.weather_tool, tools.rag_tool]

part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)
