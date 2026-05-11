import os
import time
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = os.getenv("HF_SPACE_MODEL_ID", "distilbert-base-uncased")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=14,
        )

    model.to(DEVICE)
    model.eval()

    label_names = [f"label_{i}" for i in range(int(getattr(model.config, "num_labels", 14)))]
    return tokenizer, model, label_names


tokenizer, model, label_names = _load_model_and_tokenizer()


@torch.no_grad()
def predict_text(text: str):
    text = str(text or "").strip()
    if not text:
        return "Please enter text.", 0.0, 0.0

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_id = int(probs.argmax())
    pred_label = label_names[pred_id] if pred_id < len(label_names) else f"label_{pred_id}"

    return f"Label={pred_label} (id={pred_id})", float(probs[pred_id]), float(elapsed_ms)


demo = gr.Interface(
    fn=predict_text,
    inputs=gr.Textbox(lines=6, label="Input Text", placeholder="Enter text for classification..."),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Confidence"),
        gr.Number(label="Inference time (ms)"),
    ],
    title="DBPedia Text Classification Demo",
    description="Set HF_SPACE_MODEL_ID to your fine-tuned model on Hugging Face Hub.",
)


if __name__ == "__main__":
    demo.launch()
