from agents.multi_agent import MultiAgents

app = MultiAgents()
app.agent.get_graph().draw_mermaid_png(output_file_path="workflow.png")
