"""Independent evaluation metrics with explicit matching semantics."""


def event_f1(predicted, reference, tolerance):
    """Maximum one-to-one matching of sorted points within an absolute tolerance."""
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    predicted, reference = sorted(predicted), sorted(reference)
    i = j = matched = 0
    while i < len(predicted) and j < len(reference):
        if predicted[i] < reference[j] - tolerance:
            i += 1
        elif predicted[i] > reference[j] + tolerance:
            j += 1
        else:
            matched += 1
            i += 1
            j += 1
    precision = matched / len(predicted) if predicted else float(not reference)
    recall = matched / len(reference) if reference else float(not predicted)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "matched": matched,
        "predicted": len(predicted),
        "reference": len(reference),
    }


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def word_error_rate(hypothesis, reference):
    """Whitespace-token WER; caller owns language-specific normalization."""
    h, r = hypothesis.split(), reference.split()
    if not r:
        raise ValueError("WER requires a nonempty reference")
    return edit_distance(r, h) / len(r)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (
        max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        + max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        - intersection
    )
    return intersection / union if union else 0.0


def classification_report(reference, predicted, labels):
    """Use one fixed taxonomy for every compared model, including absent classes."""
    if len(reference) != len(predicted) or len(set(labels)) != len(labels):
        raise ValueError("Paired predictions and unique class labels are required")
    if any(x not in labels for x in [*reference, *predicted]):
        raise ValueError("Unknown class outside the fixed taxonomy")
    matrix = [
        [sum(a == x and b == y for a, b in zip(reference, predicted)) for y in labels]
        for x in labels
    ]
    classes = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        guessed = sum(row[i] for row in matrix)
        precision = tp / guessed if guessed else 0
        recall = tp / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        classes.append(
            {"label": label, "support": support, "precision": precision, "recall": recall, "f1": f1}
        )
    return {
        "n": len(reference),
        "accuracy": sum(a == b for a, b in zip(reference, predicted)) / len(reference)
        if reference
        else None,
        "macro_f1": sum(c["f1"] for c in classes) / len(classes) if classes else None,
        "balanced_accuracy_present_classes": sum(c["recall"] for c in classes if c["support"])
        / sum(bool(c["support"]) for c in classes)
        if reference
        else None,
        "labels": labels,
        "confusion_matrix": matrix,
        "per_class": classes,
    }


def match_boxes(predicted, reference, min_iou=0.5, time_tolerance=0.001):
    """Maximum-cardinality frame-aligned matching; prefer high-IoU edges locally."""
    if not 0 <= min_iou <= 1 or time_tolerance < 0:
        raise ValueError("Invalid matching thresholds")
    adjacency = {}
    for i, x in enumerate(predicted):
        candidates = []
        for j, y in enumerate(reference):
            tx = x.get("sample_time", x["start"])
            ty = y.get("sample_time", y["start"])
            overlap = box_iou(x["bbox"], y["bbox"])
            if abs(tx - ty) <= time_tolerance and overlap >= min_iou:
                candidates.append((overlap, j))
        adjacency[i] = sorted(candidates, reverse=True)
    owners = {}

    def augment(i, seen):
        for overlap, j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or augment(owners[j][0], seen):
                owners[j] = (i, overlap)
                return True
        return False

    for i in range(len(predicted)):
        augment(i, set())
    return [(overlap, i, j) for j, (i, overlap) in sorted(owners.items())]
