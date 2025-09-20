from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, get_key
from postgres_database.postgress_database import checkpointer
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from langmem.short_term import SummarizationNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Any
from langchain_core.messages.utils import count_tokens_approximately

load_dotenv()

model_name = get_key(".env", "AZURE_OPENAI_MODEL_NAME")
llm = init_chat_model(
            model=model_name,
            temperature=0.1,
            model_provider="azure_openai"
        )

tools = []

class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=llm,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)

system_prompt = """You are a helpful assistant that provides analysis on various topics."""
react_agent_graph = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
            checkpointer=checkpointer,
            pre_model_hook=summarization_node,
            state_schema=State,
        )

def get_messages(thread_id: str) -> list:
    config = RunnableConfig(configurable={"thread_id": thread_id})
    state = react_agent_graph.get_state(config=config)
    if isinstance(state, tuple):
            state = state[0]
            
    if not state or "messages" not in state:
        return []
        
    messages: list[BaseMessage] = state["messages"]
    
    simplified_messages = []
    for message in messages:
        simplified_messages.append((message.type, message.content))
    return simplified_messages







