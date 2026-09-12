"""Central paths and constants for the is-a-human detector."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"                 # symlink -> ../hackmty26-main
AUDIO = DATA / "audio"               # populated once altur-challenge-audio.zip is unzipped
TURNS = DATA / "turns"
MANIFEST = DATA / "manifest.csv"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"

# Channel convention from the challenge README:
CALLER_CH = 0   # the one we classify (human or synthetic)
AGENT_CH = 1    # the bank's AI agent

CACHE.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
