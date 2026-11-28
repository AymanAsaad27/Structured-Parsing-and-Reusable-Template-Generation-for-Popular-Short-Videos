"""Portable editing skeleton with strict bounds for the presentation prototype."""

import math
import re

from beauty_video_templates.semantics.clip_roles import ROLES


def validate_template(template):
    if template.get("schema_version") != "1.0-demo":
        raise ValueError("Choose a Beauty Template Studio 1.0-demo template.")
    if not isinstance(template.get("template_id"), str):
        raise ValueError("Template ID is required.")
    if (template.get("width"), template.get("height")) not in {(720, 1280), (360, 640)}:
        raise ValueError("The demo supports vertical 720x1280 or 360x640 exports.")
    if template.get("fps") not in {24, 30}:
        raise ValueError("Export frame rate must be 24 or 30.")
    slots = template.get("slots")
    if not isinstance(slots, list) or not 1 <= len(slots) <= 80:
        raise ValueError("A template needs between 1 and 80 shots.")
    cursor = 0
    ids = set()
    for slot in slots:
        start, end = slot.get("start"), slot.get("end")
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in [start, end]
        ):
            raise ValueError("Shot times must be finite numbers.")
        if abs(start - cursor) > 1e-5 or end - start < 1 / template["fps"] - 1e-5:
            raise ValueError("Shots must be contiguous from zero and at least one frame long.")
        if not isinstance(slot.get("id"), str) or slot["id"] in ids:
            raise ValueError("Shot IDs must be unique strings.")
        ids.add(slot["id"])
        if slot.get("role") not in ROLES:
            raise ValueError("Unknown shot function.")
        if not isinstance(slot.get("caption", ""), str) or len(slot.get("caption", "")) > 180:
            raise ValueError("Keep each caption to 180 characters or fewer.")
        box = slot.get("caption_box", [0.08, 0.72, 0.92, 0.86])
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(
                not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1
                for x in box
            )
        ):
            raise ValueError("Caption boxes need four coordinates between zero and one.")
        if box[2] - box[0] < 0.15 or box[3] - box[1] < 0.04:
            raise ValueError("Caption boxes are too small or reversed.")
        cursor = end
    duration = template.get("duration_seconds")
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 1 <= duration <= 90
        or abs(cursor - duration) > 1e-5
    ):
        raise ValueError("Template duration must match its shots and be between 1 and 90 seconds.")
    palette = template.get("style", {}).get("palette", ["#eee9e4"])
    if (
        not isinstance(palette, list)
        or not 1 <= len(palette) <= 8
        or any(not isinstance(c, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in palette)
    ):
        raise ValueError("Palette colours must use six-digit hex values.")
    return template


def caption_for(role, product_name, points, index):
    product_name = str(product_name).strip()[:100] or "Your product"
    points = [str(x).strip()[:130] for x in points if str(x).strip()][:6]
    choices = {
        "hook": product_name,
        "product_introduction": product_name,
        "texture_demonstration": "A closer look at the texture",
        "application_demonstration": "Application illustration",
        "result_comparison": "A closer look - product demonstration",
        "presenter_explanation": points[index % len(points)]
        if points
        else "Meet your next routine step",
        "call_to_action": "Explore " + product_name,
        "detail_cutaway": "The details, up close",
    }
    return choices[role][:180]
