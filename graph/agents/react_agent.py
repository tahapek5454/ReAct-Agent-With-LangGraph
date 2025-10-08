from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, get_key
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from langmem.short_term import SummarizationNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Any
from langchain_core.messages.utils import count_tokens_approximately
from graph.tools.tools import tool_node
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.state import CompiledStateGraph

load_dotenv()


class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]

class ReactAgentManager:
    def __init__(self, checkpointer: PostgresSaver):
        self.checkpointer = checkpointer
        self.model_name = get_key(".env", "AZURE_OPENAI_MODEL_NAME")
        self.llm = init_chat_model(
                    model=self.model_name,
                    temperature=0.1,
                    model_provider="azure_openai"
                )
        self.summarization_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=self.llm,
            max_tokens=8182,
            max_summary_tokens=1224,
            output_messages_key="llm_input_messages",
        )
        self.react_agent_graph : CompiledStateGraph | None = None


    def prompt(self, state: State, config: RunnableConfig) -> list[AnyMessage]:  
        user_name = config["configurable"].get("user_name") or "User" # type: ignore

        system_msg = f"Sen bir yardımcı asistanısın. Kullanıcı sorularını yanıtlamaya ve ihtiyaç duyduklarında araçları kullanmaya hazırsın. Karşındaki kullanıcıya {user_name} olarak hitap et."
    
        return [{"role": "system", "content": system_msg}] + state["messages"] # type: ignore

    def create_agent(self):
        self.react_agent_graph = create_react_agent(
                    model=self.llm,
                    tools=tool_node,
                    prompt=self.prompt, # type: ignore
                    checkpointer=self.checkpointer,
                    pre_model_hook=self.summarization_node,
                    state_schema=State,
                    debug=False,
                    version='v2',
                    name="ReactAgent-Demo"
                )
        
        return self.react_agent_graph

    def get_messages(self, thread_id: str) -> list:
        if self.react_agent_graph is None:
            return  []
        
        config = RunnableConfig(configurable={"thread_id": thread_id})
        state = self.react_agent_graph.get_state(config=config)
        if isinstance(state, tuple):
                state = state[0]
                
        if not state or "messages" not in state:
            return []
            
        messages: list[BaseMessage] = state["messages"]
        
        simplified_messages = []
        for message in messages:
            simplified_messages.append((message.type, message.content))
        return simplified_messages







