import os

from demo_gradio import demo, launch


# Expose the demo object for Gradio/HF style runners.
if __name__ == "__main__":
    # Default share=True to bypass localhost restrictions unless explicitly disabled.
    share = os.environ.get("GRADIO_SHARE", "1") in ("1", "true", "True")
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "8080"))
    # Set share= True via env if localhost is blocked in the runtime.
    launch(server_name=server_name, server_port=server_port, share=share)

