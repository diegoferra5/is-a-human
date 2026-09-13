"""Send a real WAV to a RUNNING /detect server and print the response.

Start the server first:  uvicorn src.serve:app --port 8000
Then:                    python3 scripts/detect_client.py data/audio/<id>.wav
"""
import base64
import os
import sys

import httpx

URL = os.environ.get("DETECT_URL", "http://localhost:8000/detect")


def main(wav_path: str):
    b64 = base64.b64encode(open(wav_path, "rb").read()).decode()
    r = httpx.post(URL, json={"audio_b64": b64}, timeout=60)
    r.raise_for_status()
    print(r.json())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 scripts/detect_client.py <path-to-wav>")
    main(sys.argv[1])
