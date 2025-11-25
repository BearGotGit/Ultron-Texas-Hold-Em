"""Minimal training script (PyTorch) that trains a tiny MLP on the generated dataset.

Usage:
  python -m training.train_model --data data/processed/train.jsonl --epochs 1 --batch 32

This is a lightweight example to get you started. It expects `training/data_loader.py`.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training.data_loader import PokerStepDataset, collate_fn


class TinyMLP(nn.Module):
	def __init__(self, input_size, hidden=64, num_actions=4):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_size, hidden),
			nn.ReLU(),
			nn.Linear(hidden, hidden),
			nn.ReLU(),
			nn.Linear(hidden, num_actions),
		)

	def forward(self, x):
		return self.net(x)


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--data", required=True)
	p.add_argument("--epochs", type=int, default=1)
	p.add_argument("--batch", type=int, default=32)
	p.add_argument("--lr", type=float, default=1e-3)
	p.add_argument("--save-path", type=str, default="models/saved/model.pt", help="Where to save the trained model")
	return p.parse_args()


def train(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ds = PokerStepDataset(args.data)

	# split into train / val (80/20)
	total = len(ds)
	if total == 0:
		print("No data found in", args.data)
		return

	indices = list(range(total))
	split = int(total * 0.8)
	train_idx = indices[:split]
	val_idx = indices[split:]

	from torch.utils.data import Subset

	train_ds = Subset(ds, train_idx)
	val_ds = Subset(ds, val_idx) if len(val_idx) > 0 else None

	loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
	val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn) if val_ds is not None else None

	# infer input size from the dataset (robust to different PokerStepDataset versions)
	try:
		sample_x, _ = ds[0]
		input_size = int(sample_x.numel())
	except Exception:
		# fallback to previous default
		input_size = 123

	model = TinyMLP(input_size).to(device)
	opt = optim.Adam(model.parameters(), lr=args.lr)
	loss_fn = nn.CrossEntropyLoss()

	model.train()
	for epoch in range(args.epochs):
		total_loss = 0.0
		total = 0
		correct = 0
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)

			logits = model(x)
			loss = loss_fn(logits, y)

			opt.zero_grad()
			loss.backward()
			opt.step()

			total_loss += loss.item() * x.size(0)
			total += x.size(0)
			preds = logits.argmax(dim=1)
			correct += (preds == y).sum().item()

		avg_loss = total_loss / max(1, total)
		acc = correct / max(1, total)
		print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f} - acc: {acc:.4f}")

		# run validation pass
		if val_loader is not None:
			model.eval()
			v_total = 0
			v_loss = 0.0
			v_correct = 0
			with torch.no_grad():
				for vx, vy in val_loader:
					vx = vx.to(device)
					vy = vy.to(device)
					v_logits = model(vx)
					v_l = loss_fn(v_logits, vy)
					v_loss += v_l.item() * vx.size(0)
					preds = v_logits.argmax(dim=1)
					v_correct += (preds == vy).sum().item()
					v_total += vx.size(0)

			v_avg = v_loss / max(1, v_total)
			v_acc = v_correct / max(1, v_total) if v_total > 0 else 0.0
			print(f"  Val - loss: {v_avg:.4f} - acc: {v_acc:.4f}")
			model.train()
	# ensure directory exists and save model
	try:
		import os
		os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
		torch.save(model, args.save_path)
		print(f"Saved trained model to {args.save_path}")
		# Also save state_dict for safer loading across environments
		try:
			sd_path = args.save_path + ".state_dict.pt"
			torch.save(model.state_dict(), sd_path)
			print(f"Saved model state_dict to {sd_path}")
		except Exception as e:
			print(f"Failed to save state_dict: {e}")
	except Exception as e:
		print(f"Failed to save model: {e}")


def main():
	args = parse_args()
	train(args)


if __name__ == "__main__":
	main()

