import json
import torch
import clip
from PIL import Image
import numpy as np
from collections import Counter, defaultdict


device = "cpu"

model, preprocess = clip.load(
    "ViT-L/14",
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
TOP_K = 3  # number of nearest neighbors used for the majority-vote prediction

# Minimum similarity score required before trusting a prediction.
# Below this, the query is treated as "unknown" -- not confidently
# any of the existing classes. Cosine similarity ranges from -1 to 1;
# for CLIP image embeddings, scores below ~0.5 usually mean the query
# doesn't closely resemble anything in the database.
# Tune this based on your own benchmark results: look at the similarity
# scores of CORRECT matches vs scores for things that don't belong to
# any class, and set the threshold somewhere between the two.
CONFIDENCE_THRESHOLD = 0.5


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


best_score = results[0][2]
is_confident = best_score >= CONFIDENCE_THRESHOLD


# --- Top-1 prediction (single best match, original behavior) ---
top1_class = results[0][0] if is_confident else "unknown"


# --- Top-K majority vote prediction ---
top_k_matches = results[:TOP_K]

vote_counts = Counter(cls for cls, _, _ in top_k_matches)
best_count = max(vote_counts.values())
tied_classes = [cls for cls, count in vote_counts.items() if count == best_count]

if len(tied_classes) == 1:
    topk_class = tied_classes[0]
else:
    # Tie-break: whichever tied class has the highest total similarity among the top-k
    sim_sum = defaultdict(float)
    for cls, _, sim in top_k_matches:
        if cls in tied_classes:
            sim_sum[cls] += sim
    topk_class = max(tied_classes, key=lambda cls: sim_sum[cls])

if not is_confident:
    topk_class = "unknown"


print(f"\nBest match score: {best_score:.3f}  (threshold: {CONFIDENCE_THRESHOLD})")
if not is_confident:
    print("=> Below confidence threshold -- this doesn't look like a confident match to any known class.")

print(f"\nTop-{TOP_K} votes:", dict(vote_counts))
print("Predicted class (Top-1):", top1_class)
print(f"Predicted class (Top-{TOP_K} majority vote):", topk_class)
