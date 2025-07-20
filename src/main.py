from agents.multi_agent import multi_agent

import chainlit as cl


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    response = await multi_agent.chat(message.content)
    await msg.stream_token(response)
    await msg.send()
