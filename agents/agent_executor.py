import logging
from config.settings import Config
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor

from agents.tools import tools
from agents.prompts import prompt
from agents.tools import search_by_feature, search_by_metadata, search_hybrid, random_search


logger = logging.getLogger(__name__)


def initialize_agent():
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=Config.GROQ_API_KEY
    )

    # llm = ChatOllama(
    #     model="gpt-oss:20b",
    #     temperature=0,
    # )

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_execution_time=20
    )
    return executor


#######################################
## For Groq based Tool Calling Agent ##
#######################################
def agent_search(executor, query: str):
    result = executor.invoke({"input": query})

    if "intermediate_steps" in result: 
        if result["intermediate_steps"]:
            logger.info("Using intermediate steps for tool result.")
            tool_result = result["intermediate_steps"][-1][1]
            if isinstance(tool_result, list):
                tool_result = [img for img in tool_result if img]
            return tool_result
        else:
            tool_name = result.get("output", "").strip()
            tool_mapping = {
                "search_by_feature": search_by_feature,
                "search_by_metadata": search_by_metadata,
                "search_hybrid": search_hybrid,
                "random_search": random_search
            }

            logger.info("No intermediate steps found, using final tool call.")
            if tool_name in tool_mapping:
                tool_result = tool_mapping[tool_name].invoke({"query": query})
                return tool_result

    return []


# #########################################
# ## For Ollama based Tool Calling Agent ##
# #########################################
# def agent_search(executor, query: str):
#     result = executor.invoke({"input": query})
#     tool_name = result.get("output", "").strip()

#     tool_mapping = {
#         "search_by_feature": search_by_feature,
#         "search_by_metadata": search_by_metadata,
#         "search_hybrid": search_hybrid,
#         "random_search": random_search
#     }

#     if tool_name in tool_mapping:
#         tool_result = tool_mapping[tool_name].invoke({"query": query})
#         return tool_result
#     return []

