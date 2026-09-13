"""Entry point for `is-a-human-serve` (uvicorn on 0.0.0.0:8000)."""

import uvicorn


def main() -> None:
    uvicorn.run("is_a_human.api.app:create_app", factory=True, host="0.0.0.0", port=8000)
