import torch
import clip

device = "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

print("CLIP works!")