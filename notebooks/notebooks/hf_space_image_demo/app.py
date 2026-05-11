
import gradio as gr
import torch
import pandas as pd
import timm
from PIL import Image
from pathlib import Path
import json

# --- Cấu hình chung ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DF = pd.read_csv("results.csv")
with open("labels.json", "r", encoding="utf-8") as f:
    ID2LABEL = {int(k): v for k, v in json.load(f).items()}

# Biến toàn cục để giữ model hiện tại, tránh load lại liên tục
CURRENT_MODEL = None
CURRENT_MODEL_NAME = None
CURRENT_TFM = None

class ImageClassifier(torch.nn.Module):
    def __init__(self, timm_name, n_classes, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, n_classes),
        )
    def forward(self, x): return self.head(self.backbone(x))

def load_selected_model(run_name):
    global CURRENT_MODEL, CURRENT_MODEL_NAME, CURRENT_TFM

    if CURRENT_MODEL_NAME == run_name:
        return CURRENT_MODEL, CURRENT_TFM

    # Lấy thông tin từ CSV
    row = DF[DF["run_name"] == run_name].iloc[0]
    model_type = row["model"]
    ckpt_path = row["ckpt"]

    # Khởi tạo model
    model = ImageClassifier(model_type, n_classes=len(ID2LABEL)).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # Khởi tạo transforms
    data_cfg = timm.data.resolve_model_data_config(timm.create_model(model_type, pretrained=False))
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
    ])

    CURRENT_MODEL = model
    CURRENT_MODEL_NAME = run_name
    CURRENT_TFM = tfm
    return model, tfm

def predict(image, run_name):
    if image is None: return None
    model, tfm = load_selected_model(run_name)

    x = tfm(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    return {ID2LABEL[i]: float(prob[i]) for i in range(len(ID2LABEL))}

# --- Giao diện Gradio ---
with gr.Blocks(title="Crop Pest & Disease Multi-Model Demo") as demo:
    gr.Markdown("# 🌿 Crop Pest & Disease Detection")
    gr.Markdown("Chọn Model phía dưới và upload ảnh để thử nghiệm.")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil")
            # Dropdown cho phép chọn model từ cột run_name trong CSV
            model_dropdown = gr.Dropdown(
                choices=DF["run_name"].tolist(), 
                value=DF["run_name"].tolist()[0], 
                label="Chọn kiến trúc Model (Backbone)"
            )
            btn = gr.Button("Dự đoán", variant="primary")

        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Kết quả phân loại")

    btn.click(fn=predict, inputs=[img_input, model_dropdown], outputs=label_output)

demo.launch()
