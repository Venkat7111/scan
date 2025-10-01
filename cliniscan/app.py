import io
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from src.utils.gradcam import GradCAM


@st.cache_resource
def load_model(weights_path: Path, class_map_path: Path) -> Tuple[nn.Module, dict, int]:
	with open(class_map_path, "r", encoding="utf-8") as f:
		class_idx_to_name = json.load(f)
	image_size = 224
	model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features, len(class_idx_to_name))
	state = torch.load(weights_path, map_location="cpu")
	model.load_state_dict(state["model_state_dict"]) if isinstance(state, dict) and "model_state_dict" in state else model.load_state_dict(state)
	model.eval()
	return model, class_idx_to_name, state.get("image_size", image_size) if isinstance(state, dict) else image_size


def build_transform(image_size: int):
	return transforms.Compose([
		transforms.Grayscale(num_output_channels=3),
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


def pil_to_bgr(img: Image.Image) -> np.ndarray:
	rgb = np.array(img.convert("RGB"))
	bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
	return bgr


def softmax_confidence(logits: torch.Tensor) -> Tuple[int, float]:
	probs = torch.softmax(logits, dim=1)
	conf, idx = torch.max(probs, dim=1)
	return int(idx.item()), float(conf.item())


def main():
	st.set_page_config(page_title="Cliniscan: Pneumonia X-ray Classifier", page_icon="🩺", layout="centered")
	st.title("Cliniscan: Pneumonia X-ray Classifier with Grad-CAM")

	models_dir = Path("models")
	weights_path = models_dir / "resnet18_pneumonia.pt"
	class_map_path = models_dir / "class_index.json"

	if not weights_path.exists() or not class_map_path.exists():
		st.warning("Model files not found. Train the model first from the README instructions.")
		st.stop()

	model, class_idx_to_name, image_size = load_model(weights_path, class_map_path)
	transform = build_transform(image_size)
	gradcam = GradCAM(model, target_layer="layer4")

	uploaded = st.file_uploader("Upload a chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"]) 
	if uploaded is None:
		st.info("Upload an image to get started.")
		st.stop()

	image_bytes = uploaded.read()
	pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
	bgr_img = pil_to_bgr(pil_img)

	# Preprocess
	input_tensor = transform(pil_img).unsqueeze(0)

	# Predict
	with torch.no_grad():
		logits = model(input_tensor)
	pred_idx, conf = softmax_confidence(logits)
	pred_label = class_idx_to_name[str(pred_idx)] if isinstance(class_idx_to_name, dict) and str(pred_idx) in class_idx_to_name else (class_idx_to_name[pred_idx] if isinstance(class_idx_to_name, dict) is False else str(pred_idx))

	st.subheader("Prediction")
	st.write(f"{pred_label} — confidence {conf:.2%}")

	# Grad-CAM
	heatmap = gradcam.generate(input_tensor, class_idx=pred_idx)
	heatmap_resized = cv2.resize(heatmap, (bgr_img.shape[1], bgr_img.shape[0]))
	overlay = gradcam.overlay(heatmap_resized, bgr_img, alpha=0.45)

	col1, col2 = st.columns(2)
	with col1:
		st.image(pil_img, caption="Input X-ray", use_column_width=True)
	with col2:
		st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Grad-CAM Overlay", use_column_width=True)


if __name__ == "__main__":
	main()
