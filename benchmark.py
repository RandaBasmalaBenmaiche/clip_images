"""
benchmark.py

Leave-one-out accuracy benchmark for the CLIP embedding search system.

For every embedding in embeddings.json, this script temporarily removes it
from the database, treats it as a query, and checks whether the nearest
neighbor(s) among the *remaining* embeddings predict the correct class.

This requires no separate test set -- it reuses the same embeddings.json
that create_embeddings.py already produces, which makes it well suited to
small datasets (few images per class) where holding out a separate test
split would leave too little data to train/build the index from.

Two prediction strategies are evaluated side by side:
  - Top-1:  predicted class = class of the single nearest neighbor
  - Top-3:  predicted class = majority vote among the 3 nearest neighbors
            (ties broken by highest total similarity)

Run:
    python benchmark.py

Outputs:
  - Printed summary: overall accuracy (top-1 and top-3), per-class
    accuracy, and the most common confusion pairs.
  - benchmark_results.json: full per-image detail, so results can be
    diffed/compared across runs as you add classes or images.
"""

import json
from collections import Counter, defaultdict

import numpy as np

EMBEDDINGS_FILE = "embeddings.json"
RESULTS_FILE = "benchmark_results.json"


def load_database(path):
    with open(path) as f:
        data = json.load(f)

    classes = np.array([item["class"] for item in data])
    images = np.array([item["image"] for item in data])
    vectors = np.array([item["embedding"] for item in data])  # (N, D), already L2-normalized

    return classes, images, vectors


def topk_predictions(similarities, classes, k):
    """
    Given a similarity row (excluding self) and the corresponding class
    labels, return:
      - top1_class: class of the single best match
      - topk_class: majority-vote class among the top-k matches
      - top_matches: list of (class, similarity) for the top-k, for logging
    """
    order = np.argsort(-similarities)  # descending
    top_idx = order[:k]

    top_matches = [(str(classes[i]), float(similarities[i])) for i in top_idx]

    top1_class = top_matches[0][0]

    vote_counts = Counter(c for c, _ in top_matches)
    best_count = max(vote_counts.values())
    tied_classes = [c for c, cnt in vote_counts.items() if cnt == best_count]

    if len(tied_classes) == 1:
        topk_class = tied_classes[0]
    else:
        # Tie-break: sum similarity per tied class, pick highest
        sim_sum = defaultdict(float)
        for c, s in top_matches:
            if c in tied_classes:
                sim_sum[c] += s
        topk_class = max(tied_classes, key=lambda c: sim_sum[c])

    return top1_class, topk_class, top_matches


def run_benchmark(k=3):
    classes, images, vectors = load_database(EMBEDDINGS_FILE)
    n = len(classes)

    if n < 3:
        print(f"Only {n} embeddings found -- need at least a few per class for a meaningful benchmark.")
        return

    # Full similarity matrix (vectors are already normalized -> dot product = cosine similarity)
    sim_matrix = vectors @ vectors.T

    results = []
    top1_correct = 0
    top3_correct = 0

    per_class_total = Counter(classes)
    per_class_top1_correct = Counter()
    per_class_top3_correct = Counter()

    confusion_top1 = Counter()  # (true_class, predicted_class) -> count, only when wrong

    for i in range(n):
        true_class = str(classes[i])

        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # exclude self

        other_classes = classes
        top1_pred, top3_pred, top_matches = topk_predictions(sims, other_classes, k=k)

        is_top1_correct = top1_pred == true_class
        is_top3_correct = top3_pred == true_class

        top1_correct += is_top1_correct
        top3_correct += is_top3_correct
        per_class_top1_correct[true_class] += is_top1_correct
        per_class_top3_correct[true_class] += is_top3_correct

        if not is_top1_correct:
            confusion_top1[(true_class, top1_pred)] += 1

        results.append({
            "image": str(images[i]),
            "true_class": true_class,
            "top1_prediction": top1_pred,
            "top1_correct": bool(is_top1_correct),
            "top3_prediction": top3_pred,
            "top3_correct": bool(is_top3_correct),
            "top_matches": [{"class": c, "similarity": round(s, 4)} for c, s in top_matches],
        })

    overall_top1_acc = top1_correct / n
    overall_top3_acc = top3_correct / n

    summary = {
        "num_images": n,
        "num_classes": len(per_class_total),
        "overall_top1_accuracy": round(overall_top1_acc, 4),
        "overall_top3_accuracy": round(overall_top3_acc, 4),
        "per_class": {
            cls: {
                "count": per_class_total[cls],
                "top1_accuracy": round(per_class_top1_correct[cls] / per_class_total[cls], 4),
                "top3_accuracy": round(per_class_top3_correct[cls] / per_class_total[cls], 4),
            }
            for cls in sorted(per_class_total)
        },
        "top_confusions_top1": [
            {"true_class": t, "predicted_as": p, "count": c}
            for (t, p), c in confusion_top1.most_common(10)
        ],
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)

    print_summary(summary)
    print(f"\nFull details saved to {RESULTS_FILE}")


def print_summary(summary):
    print("=" * 50)
    print("BENCHMARK SUMMARY (leave-one-out)")
    print("=" * 50)
    print(f"Images:  {summary['num_images']}")
    print(f"Classes: {summary['num_classes']}")
    print(f"\nOverall Top-1 accuracy: {summary['overall_top1_accuracy']:.1%}")
    print(f"Overall Top-3 accuracy: {summary['overall_top3_accuracy']:.1%}")

    print("\nPer-class accuracy:")
    print(f"{'Class':<15}{'Count':<8}{'Top-1':<10}{'Top-3':<10}")
    for cls, stats in summary["per_class"].items():
        print(f"{cls:<15}{stats['count']:<8}{stats['top1_accuracy']:<10.1%}{stats['top3_accuracy']:<10.1%}")

    if summary["top_confusions_top1"]:
        print("\nTop confusions (Top-1, true -> predicted):")
        for conf in summary["top_confusions_top1"]:
            print(f"  {conf['true_class']} -> {conf['predicted_as']}  ({conf['count']}x)")
    else:
        print("\nNo confusions -- every image's Top-1 nearest neighbor was correct.")


if __name__ == "__main__":
    run_benchmark(k=3)
