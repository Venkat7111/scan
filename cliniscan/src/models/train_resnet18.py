import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
	train_tf = transforms.Compose([
		transforms.Grayscale(num_output_channels=3),
		transforms.Resize((image_size, image_size)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	val_tf = transforms.Compose([
		transforms.Grayscale(num_output_channels=3),
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	test_tf = val_tf
	return train_tf, val_tf, test_tf


def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int, image_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
	train_tf, val_tf, test_tf = build_transforms(image_size)
	train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
	val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tf)
	test_ds = datasets.ImageFolder(data_dir / "test", transform=test_tf)

	class_idx_to_name = {i: c for i, c in enumerate(train_ds.classes)}

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	return train_loader, val_loader, test_loader, class_idx_to_name


def build_model(num_classes: int) -> nn.Module:
	model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features, num_classes)
	return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
	model.eval()
	correct = 0
	total = 0
	loss_sum = 0.0
	criterion = nn.CrossEntropyLoss()
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss_sum += float(loss.item()) * images.size(0)
			_, predicted = torch.max(outputs, 1)
			correct += int((predicted == labels).sum().item())
			total += int(labels.size(0))
	avg_loss = loss_sum / max(total, 1)
	acc = correct / max(total, 1)
	return avg_loss, acc


def train(
	data_dir: Path,
	out_dir: Path,
	epochs: int,
	batch_size: int,
	lr: float,
	num_workers: int,
	image_size: int,
	device: torch.device,
) -> None:
	train_loader, val_loader, test_loader, class_idx_to_name = build_dataloaders(data_dir, batch_size, num_workers, image_size)
	model = build_model(num_classes=len(class_idx_to_name))
	model.to(device)

	optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
	scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
	criterion = nn.CrossEntropyLoss()

	best_val_acc = 0.0
	out_dir.mkdir(parents=True, exist_ok=True)
	weights_path = out_dir / "resnet18_pneumonia.pt"
	classes_path = out_dir / "class_index.json"

	for epoch in range(1, epochs + 1):
		model.train()
		pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
		for images, labels in pbar:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			pbar.set_postfix(loss=float(loss.item()))

		val_loss, val_acc = evaluate(model, val_loader, device)
		scheduler.step()
		print(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save({
				"model_state_dict": model.state_dict(),
				"num_classes": len(class_idx_to_name),
				"image_size": image_size,
				"class_idx_to_name": class_idx_to_name,
			}, weights_path)
			with open(classes_path, "w", encoding="utf-8") as f:
				json.dump(class_idx_to_name, f, indent=2)
			print(f"Saved best model to {weights_path}")

	# Final test
	test_loss, test_acc = evaluate(model, test_loader, device)
	print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, required=True, help="Path to chest_xray folder containing train/val/test")
	parser.add_argument("--out_dir", type=str, default="models")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--image_size", type=int, default=224)
	args = parser.parse_args()

	data_dir = Path(args.data_dir)
	out_dir = Path(args.out_dir)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	train(data_dir=data_dir, out_dir=out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers, image_size=args.image_size, device=device)


if __name__ == "__main__":
	main()
