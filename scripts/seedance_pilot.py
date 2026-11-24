"""One local Ark pilot: submit once, then resume polling by saved task ID."""

import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
RUN = ROOT / "artifacts/phase34/seedance-pilot"


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit", action="store_true", help="Allow one new paid task if no ID exists"
    )
    parser.add_argument("--wait-seconds", type=int, default=40)
    parser.add_argument("--run-name", default="seedance-pilot")
    parser.add_argument("--request-file", type=Path)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("Run name must be a single folder name")
    RUN = ROOT / "artifacts/phase34" / args.run_name
    key = None
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "ARK_API_KEY":
            key = value.strip().strip('"').strip("'")
    if not key:
        print("Key is not configured. No request sent.")
        return 1
    print("Local key loaded; value hidden.", flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False

    def save(name, value):
        (RUN / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

    def request(url, payload=None):
        headers = {
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        }
        try:
            response = session.request(
                "POST" if payload is not None else "GET",
                url,
                json=payload,
                headers=headers,
                timeout=60,
            )
        except requests.RequestException:
            print("Network connection failed. No automatic resubmission.")
            return None
        if not response.ok:
            message = response.text.replace(key, "[REDACTED]")
            try:
                details = json.loads(message)
            except ValueError:
                details = {"message": message[:1500]}
            save("api_error.json", {"http_status": response.status_code, "details": details})
            print(
                "API request rejected:",
                response.status_code,
                json.dumps(details, ensure_ascii=True)[:1800],
            )
            return None
        return response.json()

    task_path = RUN / "task.json"
    if not task_path.exists():
        if not args.submit:
            print("No submitted task. Use --submit only when a new generation is intended.")
            return 1
        payload = {
            "model": "doubao-seedance-2-5-260628",
            "content": [
                {
                    "type": "text",
                    "text": "Vertical macro beauty product footage. A clean cosmetic spatula holds a small amount of plain white cream above a neutral beige tabletop. A slow gentle camera push shows the cream texture in soft studio light. One continuous shot, stable object shape, no hands, no faces, no logos, no text, no cuts.",
                }
            ],
            "duration": 4,
            "resolution": "480p",
            "ratio": "9:16",
            "generate_audio": False,
        }
        if args.request_file:
            payload = json.loads(args.request_file.read_text(encoding="utf-8"))
        save("request.json", payload)
        save(
            "purpose.json",
            {
                "purpose": "Configured product-reference pilot"
                if args.request_file
                else "API and basic texture-shot smoke test; unbranded, text-only",
                "not_product_identity_test": not bool(args.request_file),
                "new_submissions_allowed": 1,
            },
        )
        # A durable marker prevents accidental duplicate charges after ambiguous network errors.
        marker = RUN / "submission_started.txt"
        if marker.exists():
            print(
                "Prior submission attempt recorded. Inspect the console before creating another task."
            )
            return 1
        marker.write_text(str(time.time()), encoding="utf-8")
        result = request(BASE, payload)
        if not result:
            error_file = RUN / "api_error.json"
            if error_file.exists():
                error = json.loads(error_file.read_text(encoding="utf-8"))
                if error.get("details", {}).get("error", {}).get("code") == "ModelNotOpen":
                    marker.replace(RUN / "submission_rejected_model_not_open.txt")
            return 1
        save("task.json", result)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task_id = task.get("id")
    if not task_id:
        print("No task ID returned. Inspect saved response; no automatic retry.")
        return 1
    deadline = time.monotonic() + args.wait_seconds
    while True:
        result = request(BASE + "/" + task_id)
        if not result:
            return 1
        save("status.json", result)
        status = result.get("status", "unknown")
        print("Generation status:", status, flush=True)
        if status == "succeeded":
            url = result.get("content", {}).get("video_url")
            if not url or not url.startswith("https://"):
                print("No HTTPS video URL in result.")
                return 1
            # Never forward the API Authorization header to media hosting.
            with session.get(url, timeout=120) as response:
                response.raise_for_status()
                (RUN / "pilot.mp4").write_bytes(response.content)
            print("Saved pilot.mp4 locally.")
            return 0
        if status in {"failed", "cancelled", "expired"}:
            print("Generation ended without a video. See saved status.json.")
            return 1
        if time.monotonic() >= deadline:
            print("Task saved; resume this command without --submit to continue polling.")
            return 0
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
