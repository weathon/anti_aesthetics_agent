"""Recompute all dataset embeddings using Gemini embedding via OpenRouter.

Saves one .npz per source under embeddings/, with checkpointing so that
interrupted runs can resume. Each .npz contains:
  - names: array of filename strings
  - embeddings: float32 array of shape [N, 3072]
"""

import os
import sys
import time
import argparse

import numpy as np
import dotenv

dotenv.load_dotenv()

from gemini_embedding import GeminiEmbedder, EMBED_DIM

DATASET_ROOT = "/home/wg25r/Downloads/ds/train"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
SOURCES = ["ava", "ls", "lapis"]
CHECKPOINT_EVERY = 8  # in chunks (each chunk = batch_size * max_workers items)


def list_images(source: str) -> list[str]:
    d = os.path.join(DATASET_ROOT, source)
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))


def compute_for_source(source: str, embedder: GeminiEmbedder) -> None:
    out_path = os.path.join(OUT_DIR, f"{source}.npz")
    names_all = list_images(source)
    print(f"[{source}] {len(names_all)} images")

    done_names: list[str] = []
    done_embs = np.zeros((0, EMBED_DIM), dtype=np.float32)
    if os.path.exists(out_path):
        z = np.load(out_path, allow_pickle=True)
        done_names = list(z["names"])
        done_embs = z["embeddings"]
        print(f"[{source}] resume: {len(done_names)} already embedded")

    done_set = set(done_names)
    todo = [n for n in names_all if n not in done_set]
    if not todo:
        print(f"[{source}] complete")
        return

    chunk = embedder.batch_size * embedder.max_workers  # one parallel wave
    src_dir = os.path.join(DATASET_ROOT, source)
    new_names = list(done_names)
    new_embs = [done_embs] if len(done_embs) else []

    t0 = time.time()
    for i in range(0, len(todo), chunk):
        sub = todo[i:i + chunk]
        items = [{"image_path": os.path.join(src_dir, n)} for n in sub]
        try:
            embs = embedder.embed(items)
        except Exception as e:
            print(f"[{source}] chunk failed at offset {i}: {e}", file=sys.stderr)
            # save what we have, then re-raise so user can investigate
            if new_embs:
                np.savez_compressed(out_path, names=np.array(new_names), embeddings=np.concatenate(new_embs, axis=0))
            raise

        new_names.extend(sub)
        new_embs.append(embs)

        done_count = len(new_names)
        elapsed = time.time() - t0
        rate = (done_count - len(done_names)) / max(elapsed, 1e-6)
        eta = (len(names_all) - done_count) / max(rate, 1e-6)
        print(f"[{source}] {done_count}/{len(names_all)} ({rate:.1f} img/s, eta {eta/60:.1f} min)")

        # periodic checkpoint
        if (i // chunk) % CHECKPOINT_EVERY == 0:
            np.savez_compressed(out_path, names=np.array(new_names), embeddings=np.concatenate(new_embs, axis=0))

    np.savez_compressed(out_path, names=np.array(new_names), embeddings=np.concatenate(new_embs, axis=0))
    print(f"[{source}] saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    embedder = GeminiEmbedder(batch_size=args.batch_size, max_workers=args.workers)
    for s in args.sources:
        compute_for_source(s, embedder)


if __name__ == "__main__":
    main()
