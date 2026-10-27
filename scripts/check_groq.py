"""Smoke test: check that the Groq API key stored in .env works.

Requests the list of available models (consumes no tokens).
Usage: uv run python scripts/check_groq.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


def main() -> int:
    load_dotenv()  # load variables from the .env file
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is missing from the .env file")
        return 1

    request = urllib.request.Request(
        GROQ_MODELS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "f1-sql-agent/0.1",  # some firewalls block the default UA
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        print(f"HTTP ERROR {exc.code}: {exc.reason}")  # 401 = invalid or revoked key
        return 1
    except urllib.error.URLError as exc:
        print(f"NETWORK ERROR: {exc.reason}")
        return 1

    model_ids = sorted(model["id"] for model in payload["data"])
    print(f"OK: the key works. {len(model_ids)} models available:")
    for model_id in model_ids:
        print(f"  - {model_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
