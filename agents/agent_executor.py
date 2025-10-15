from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from .tools import tools
from .prompts import prompt
from config.settings import Config

def initialize_agent():
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=Config.GROQ_API_KEY
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_execution_time=20
    )
    return executor


def agent_search(executor, query: str):
    result = executor.invoke({"input": query})
    if "intermediate_steps" in result and result["intermediate_steps"]:
        tool_result = result["intermediate_steps"][-1][1]
        if isinstance(tool_result, list):
            tool_result = [img for img in tool_result if img]
        return tool_result
    return []
