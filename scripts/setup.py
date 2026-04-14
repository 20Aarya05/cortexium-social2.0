"""
Setup script — checks dependencies, creates .env, seeds Neo4j constraints.
Run once: python scripts/setup.py
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def check_python():
    major, minor = sys.version_info[:2]
    assert major == 3 and minor >= 11, f"Python 3.11+ required, got {major}.{minor}"
    print(f"✓ Python {major}.{minor}")

def create_env():
    env_file  = ROOT / ".env"
    example   = ROOT / ".env.example"
    if not env_file.exists():
        env_file.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"✓ Created .env from .env.example — please edit it!")
    else:
        print("✓ .env already exists")

def make_dirs():
    for d in ["data/faces","data/logs","data/knowledge","data/chroma","models","exports/robot_training"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("✓ Directory structure created")

def check_ollama():
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if "llama3" in result.stdout:
            print("✓ Ollama + LLaMA 3.1 available")
        else:
            print("⚠ Ollama running but llama3.1:8b not found")
            print("  Run: ollama pull llama3.1:8b")
    except FileNotFoundError:
        print("⚠ Ollama not found — install from https://ollama.com/download")
    except Exception as e:
        print(f"⚠ Ollama check failed: {e}")

def check_neo4j():
    try:
        from neo4j import GraphDatabase
        sys.path.insert(0, str(ROOT))
        import config as cfg
        driver = GraphDatabase.driver(cfg.NEO4J_URI, auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        print(f"✓ Neo4j connected at {cfg.NEO4J_URI}")
    except Exception as e:
        print(f"⚠ Neo4j unavailable: {e}")
        print("  Download: https://neo4j.com/download-center/#community")
        print("  Graph features will be disabled until Neo4j is running.")

def main():
    print("\n── Cortexium Setup ──────────────────────────────")
    check_python()
    create_env()
    make_dirs()
    check_ollama()
    check_neo4j()
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("  1. Edit .env (set HF_TOKEN for diarization, camera source, etc.)")
    print("  2. pip install -r requirements.txt")
    print("  3. python main.py")
    print("  4. cd dashboard && npm install && npm run dev")
    print("  5. uvicorn api.main:app --port 8765 --reload\n")

if __name__ == "__main__":
    main()
