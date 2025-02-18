import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# demo.launch(server_name="0.0.0.0", share=True, server_port= 7860)
demo.launch(server_name="0.0.0.0", share=True, server_port= 7860)