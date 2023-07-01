import os
from typing import Optional, Tuple
import gradio as gr
from chatWithCache import llm_qa
from threading import Lock
import os
from dotenv import load_dotenv
load_dotenv()
import settings
settings.init()

class ChatWrapper:
    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, inp: str, history: Optional[Tuple[str, str]], chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        settings.init()
        try:
            history = history or []
            output, latency = llm_qa(inp, history)
            if settings.cache_hit == 1:
                output_str = f"Cache hit \n{latency}"
            else:
                output_str = f"Cache miss \n{latency}"
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history, output_str

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Ask your questions on your data</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask questions about your data",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    output_text = gr.Textbox(
        label="Details Of the response"
    ).style(color='red')
    

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state, agent_state], outputs=[chatbot, state, output_text])
    message.submit(chat, inputs=[message, state, agent_state], outputs=[chatbot, state, output_text])

block.launch()