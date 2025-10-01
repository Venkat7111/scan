from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn
from typing import Tuple


class GradCAM:
	"""Simple Grad-CAM for CNN models. Default target layer for ResNet18: layer4."""
	def __init__(self, model: nn.Module, target_layer: str = "layer4") -> None:
		self.model = model
		self.model.eval()
		self.target_layer_name = target_layer

		self.activations = None
		self.gradients = None

		# Hook the target layer
		target_layer_module = dict([*self.model.named_modules()])[self.target_layer_name]

		def forward_hook(module, inp, out):
			self.activations = out.detach()

		def backward_hook(module, grad_in, grad_out):
			self.gradients = grad_out[0].detach()

		target_layer_module.register_forward_hook(forward_hook)
		target_layer_module.register_full_backward_hook(backward_hook)

	@torch.no_grad()
	def _normalize(self, arr: np.ndarray) -> np.ndarray:
		arr_min = arr.min()
		arr_max = arr.max()
		if arr_max - arr_min < 1e-12:
			return np.zeros_like(arr)
		return (arr - arr_min) / (arr_max - arr_min)

	def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
		"""Generate Grad-CAM heatmap for input_tensor (1xCxHxW)."""
		self.model.zero_grad(set_to_none=True)
		logits = self.model(input_tensor)
		if class_idx is None:
			class_idx = int(torch.argmax(logits, dim=1).item())

		target = logits[0, class_idx]
		target.backward(retain_graph=True)

		# activations: [B, C, H, W], gradients: [B, C, H, W]
		weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
		cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
		cam = torch.relu(cam)
		cam = cam[0, 0].cpu().numpy()

		cam = self._normalize(cam)
		return cam

	def overlay(self, heatmap: np.ndarray, image_bgr: np.ndarray, alpha: float = 0.4) -> np.ndarray:
		"""Overlay heatmap onto an OpenCV BGR image."""
		heatmap_uint8 = np.uint8(255 * heatmap)
		heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
		overlay = cv2.addWeighted(heatmap_color, alpha, image_bgr, 1 - alpha, 0)
		return overlay
