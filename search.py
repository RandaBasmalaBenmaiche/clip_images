import json
import torch
import clip
from PIL import Image
import numpy as np



device = "cpu"

model, preprocess = clip.load(
    "ViT-B/32",
    device=device
)


def get_embedding(image_path):

    image = Image.open(image_path).convert("RGB")

    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_image(image_input)

    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True
    )

    return embedding[0].numpy()


# load database

with open("embeddings.json") as f:
    database = json.load(f)


query_image = "test.jpg"


query_vector = get_embedding(query_image)


results = []


for item in database:

    vector = np.array(
        item["embedding"]
    )

    similarity = np.dot(
        query_vector,
        vector
    )

    results.append(
        (
            item["class"],
            item["image"],
            similarity
        )
    )


# highest similarity first

results.sort(
    key=lambda x: x[2],
    reverse=True
)


print("\nTop matches:")

for r in results[:5]:
    print(
        f"{r[0]} | {r[1]} | score={r[2]:.3f}"
    )


# final prediction

best_class = results[0][0]

print(
    "\nPredicted class:",
    best_class
)

print(results)