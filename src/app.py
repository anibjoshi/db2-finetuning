import gradio as gr
from pathlib import Path
from typing import Dict
from inference.inference_manager import InferenceManager

# Model paths configuration
MODEL_PATHS: Dict[int, Path] = {
    1: Path("src/model/model1/best_model"),
    2: Path("src/model/model2/best_model"),
    3: Path("src/model/model3/best_model"),
    4: Path("src/model/model4/best_model"),
    5: Path("src/model/model5/best_model")
}

# Supported DB2 versions
DB2_VERSIONS = ["11.1", "11.5", "12.1"]

def generate_response(question: str, db2_version: str, model_version: int) -> str:
    """Generate response using the selected model version."""
    try:
        model_path = MODEL_PATHS.get(model_version)
        if not model_path:
            return f"Error: Invalid model version {model_version}"

        inferencer = InferenceManager(model_path)
        response = inferencer.generate_response(
            question=question,
            db2_version=db2_version
        )
        return response if response else "No response generated"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(
            label="Question",
            placeholder="Enter your DB2 related question here..."
        ),
        gr.Dropdown(
            choices=DB2_VERSIONS,
            label="DB2 Version",
            value="11.5"
        ),
        gr.Dropdown(
            choices=list(MODEL_PATHS.keys()),
            label="Model Version",
            value=1
        )
    ],
    outputs=gr.Textbox(label="Response"),
    title="DB2 Assistant",
    description="Ask questions about DB2 database management and troubleshooting",
    examples=[
        ["How do I create a new database in DB2?", "11.5", 1],
        ["What are the common backup strategies in DB2?", "12.1", 1],
        ["How to optimize query performance in DB2?", "11.1", 1]
    ]
)

if __name__ == "__main__":
    demo.launch(share=False) 