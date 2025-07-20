import operator

from typing import TypedDict, List, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.globals import set_debug, set_verbose

from config import GEMINI_API_KEY

set_debug(True)
set_verbose(True)


# Set up MCP client
client = MultiServerMCPClient(
    {
        "healthcare_insurance_plan": {
            "url": "http://localhost:8000/healthcare-insurance-plan/mcp",
            "transport": "streamable_http",
        },
        "vehicle_insurance_claims": {
            "url": "http://localhost:8000/vehicle-insurance-claims/mcp",
            "transport": "streamable_http",
        },
    }
)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


class MultiAgents:
    def __init__(self):
        self.tools = None
        self.llm = None
        self.memory = None
        self.agent = None
        self._initialized = False

    async def _initialize(self):
        if not self._initialized:
            self.tools = await client.get_tools()
            self.llm = self._setup_gemini_llm()
            self.agent = self._create_graph()
            self._initialized = True

    def _setup_gemini_llm(self):
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            api_key=GEMINI_API_KEY,
        ).bind_tools(self.tools)

    def _create_graph(self):
        workflow = StateGraph(AgentState)
        tool_node = ToolNode(self.tools)

        workflow.add_edge(START, "agent")
        workflow.add_node("agent", self._call_llm)
        workflow.add_node("tools", tool_node)

        workflow.add_conditional_edges("agent", self._should_continue)

        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _should_continue(self, state: AgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return END

    async def _call_llm(self, state: AgentState):
        response = await self.llm.ainvoke(state["messages"])
        return {"messages": [response]}

    async def chat(self, messages):
        state = AgentState(messages=messages)
        response = await self.agent.ainvoke(state)
        return response["messages"][-1].content


multi_agent = MultiAgents()
