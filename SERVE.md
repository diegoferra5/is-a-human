# Serve `/detect`

Run these from the repo root so `models/tandem.json` loads. Need `.venv` and that model file (`is-a-human-train` writes it).

## 1. Start the API

```bash
source .venv/bin/activate
is-a-human-serve
```

Leave that terminal open. You should see `Uvicorn running on http://0.0.0.0:8000`.

Check it:

```bash
curl http://127.0.0.1:8000/health
```

Expect `{"status":"ok","ready":true,"model_loaded":true}`. If `model_loaded` is false, train first: `is-a-human-train`.

## 2. Pick a URL

| Who is calling | URL |
|---|---|
| This machine | `http://127.0.0.1:8000/detect` |
| Same Wi‑Fi (fastest for judges) | `http://<lan-ip>:8000/detect` |
| Off-network | Cloudflare tunnel (below) |

LAN IP:

```bash
ipconfig getifaddr en0
```

Mac firewall must allow incoming on port 8000 (or be off). Same SSID as the judges; guest/client-isolation Wi‑Fi will block them. Keep the laptop awake.

## 3. Public tunnel (optional)

This network blocks QUIC, so force HTTP/2. Needs `cloudflared` (`brew install cloudflared`).

```bash
cloudflared tunnel --url http://localhost:8000 --protocol http2
```

It prints a `https://….trycloudflare.com` hostname. Detect is that host plus `/detect`. The hostname **changes every restart**.

## 4. Score like the judge

Local:

```bash
.venv/bin/python resources/hackmty26-main/scripts/check_endpoint.py \
  --url http://127.0.0.1:8000/detect \
  --manifest resources/hackmty26-main/manifest.csv \
  --audio-dir resources/audio \
  --split val --n 20
```

Same Wi‑Fi (from another machine, or this one):

```bash
.venv/bin/python resources/hackmty26-main/scripts/check_endpoint.py \
  --url http://$(ipconfig getifaddr en0):8000/detect \
  --manifest resources/hackmty26-main/manifest.csv \
  --audio-dir resources/audio \
  --split val --n 20
```

`--n 0` runs the full val split.

## 5. Stop

Ctrl+C in the serve terminal. If the port is stuck:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill <pid>
```

## Notes

- Inference is tens of milliseconds. ~1s through Cloudflare is the tunnel plus a 5–10 MB JSON upload, not the model.
- `src/serve.py` is the old stub. Live judging uses `is-a-human-serve` (`is_a_human.api`).
