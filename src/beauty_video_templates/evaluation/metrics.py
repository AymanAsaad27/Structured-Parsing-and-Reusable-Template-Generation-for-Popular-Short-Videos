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
