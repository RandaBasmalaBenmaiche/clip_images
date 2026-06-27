import os
import json
import torch
import clip
from PIL import Image

device = "cpu"

model, preprocess = clip.load(
    "ViT-L/14",
    device=device
)

DATA_DIR = "data"
OUTPUT = "embeddings.json"


def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")

    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_input)

    # normalize for cosine similarity
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding[0].cpu().tolist()


database = []

for class_name in os.listdir(DATA_DIR):

    class_path = os.path.join(DATA_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"Processing {class_name}")

    for filename in os.listdir(class_path):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_path = os.path.join(
                class_path,
                filename
            )

            vector = get_embedding(image_path)

            database.append({
                "class": class_name,
                "image": filename,
                "embedding": vector
            })




with open(OUTPUT, "w") as f:
    json.dump(database, f)


print(
    f"Saved {len(database)} embeddings"
)