"""Minimal /detect server showing the exact contract. Replace `detect()` with your model.

    python scripts/example_server.py --port 8000
    python scripts/check_endpoint.py --url http://localhost:8000/detect

Standard library only. The example decodes the WAV and returns a meaningless placeholder verdict;
it only proves the plumbing works.
"""

import argparse
import base64
import io
import json
import struct
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer


def detect(wav_bytes: bytes) -> dict:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    assert channels == 2 and width == 2 and rate == 8000, f"unexpected format: {channels}ch {width * 8}bit {rate}Hz"
    samples = struct.unpack("<%dh" % (len(frames) // 2), frames)
    caller, agent = samples[0::2], samples[1::2]  # channel 0 = caller, channel 1 = agent
    # Placeholder verdict with no signal in it: alternates on the clip length. Replace with your model.
    is_synthetic = len(caller) % 2 == 0
    return {"is_synthetic": bool(is_synthetic), "confidence": 0.5}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/detect":
            return self._reply(404, {"error": "use POST /detect"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            wav_bytes = base64.b64decode(body["audio_base64"])
            result = detect(wav_bytes)
        except Exception as e:  # bad JSON, missing field, bad audio
            return self._reply(400, {"error": str(e)})
        self._reply(200, result)

    def _reply(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"listening on http://{args.host}:{args.port}/detect")
    HTTPServer((args.host, args.port), Handler).serve_forever()
