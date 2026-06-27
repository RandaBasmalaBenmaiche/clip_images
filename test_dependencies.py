import torch
import clip
from PIL import Image

print("✅ Imports successful")

# Check device
device = "cpu"
print(f"Using device: {device}")

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
print("✅ CLIP model loaded")

# Create a dummy image (no file needed)
from PIL import ImageDraw

img = Image.new("RGB", (224, 224), color="white")
draw = ImageDraw.Draw(img)
draw.text((50, 100), "Test", fill="black")

# Preprocess
image = preprocess(img).unsqueeze(0).to(device)

# Get embedding
with torch.no_grad():
    embedding = model.encode_image(image)

# Normalize
embedding = embedding / embedding.norm(dim=-1, keepdim=True)

print("✅ Embedding computed")
print("Embedding shape:", embedding.shape)

print("\n🎉 Everything works correctly!")