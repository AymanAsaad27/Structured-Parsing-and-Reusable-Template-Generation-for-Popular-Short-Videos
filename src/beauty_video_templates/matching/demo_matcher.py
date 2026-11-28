"""Match materials by editable function tags and relative CLIP scores."""

COMPATIBLE = {
    "hook": {"application_demonstration", "product_introduction", "presenter_explanation"},
    "call_to_action": {"product_introduction"},
    "result_comparison": {"application_demonstration", "presenter_explanation"},
    "detail_cutaway": {"texture_demonstration", "product_introduction"},
}


def assign_materials(template, assets):
    if not assets:
        raise ValueError("Add at least one product image or clip before creating a video.")
    counts = {a["id"]: 0 for a in assets}
    assignments = []
    for slot in template["slots"]:
        requested = slot.get("asset_id")
        if requested and requested not in counts:
            raise ValueError(f"The material selected for {slot['id']} is no longer available.")
        candidates = [a for a in assets if a["id"] == requested] if requested else assets

        def score(asset):
            role = slot["role"]
            tag = asset.get("role", "auto")
            visual = asset.get("role_scores", {}).get(role, 0)
            tag_score = 1 if tag == role else 0.65 if tag in COMPATIBLE.get(role, set()) else 0
            size_score = asset.get("size_scores", {}).get(slot.get("shot_size"), 0)
            return 0.7 * max(visual, tag_score) + 0.2 * size_score - 0.12 * counts[asset["id"]]

        selected = max(candidates, key=score)
        warning = None
        if slot["role"] == "result_comparison" and selected.get("role") != "result_comparison":
            warning = "No verified comparison material: this slot uses an illustration, not a product result."
        assignments.append(
            {
                "slot_id": slot["id"],
                "asset_id": selected["id"],
                "asset_name": selected["name"],
                "match_score": round(score(selected), 4),
                "selection": "manual"
                if requested
                else "function/shot-size matching with reuse penalty",
                "reuse_index": counts[selected["id"]],
                "warning": warning,
            }
        )
        counts[selected["id"]] += 1
    return assignments
