import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.globals import set_debug, set_verbose

from config import GEMINI_API_KEY

set_debug(True)
set_verbose(True)


# Set up MCP client
client = MultiServerMCPClient(
    {
        "healthcare_insurance_plan": {
            "url": "http://localhost:3000/healthcare-insurance-plan/mcp",
            "transport": "streamable_http",
        },
        "vehicle_insurance_claims": {
            "url": "http://localhost:3000/vehicle-insurance-claims/mcp",
            "transport": "streamable_http",
        },
    }
)

class MultiAgents:
    def __init__(self):
        self.tools = asyncio.run(client.get_tools())
        self.llm = self._setup_gemini_llm()
        self.agent = self._create_graph()

    def _setup_gemini_llm(self):
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            api_key=GEMINI_API_KEY,
        ).bind_tools(self.tools)

    def _create_graph(self):
        workflow = StateGraph(MessagesState)
        tool_node = ToolNode(self.tools)

        workflow.add_edge(START, "agent")
        workflow.add_node("agent", self._call_llm)
        workflow.add_node("tools", tool_node)

        workflow.add_conditional_edges("agent", self._should_continue)

        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _should_continue(self, state: MessagesState) -> str:
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return END

    async def _call_llm(self, state: MessagesState):
        response = await self.llm.ainvoke(state["messages"])
        return {"messages": [response]}

    async def chat(self, message):
        state = MessagesState(messages=[HumanMessage(content=message)])
        response = await self.agent.ainvoke(state)
        return response["messages"][-1].content


multi_agent = MultiAgents()
