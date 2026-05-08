"""Dataset and embedding loader.

Loads pre-computed Gemini embeddings from local .npz files and initialises
a GeminiEmbedder for query-side embedding. Tensors are prepared for
cosine-similarity search over the AVA / liminal-space / lapis subsets.
"""

import os

import dotenv
dotenv.load_dotenv()

import numpy as np
import torch

from gemini_embedding import GeminiEmbedder

EMBED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")


def _load(source: str):
    path = os.path.join(EMBED_DIR, f"{source}.npz")
    z = np.load(path, allow_pickle=True)
    # Keep names as a numpy array so downstream code's `names[i].item()` works.
    return z["names"], z["embeddings"].astype(np.float32)


ava_names_list, ava_embeddings = _load("ava")
ls_names_list, ls_embeddings = _load("ls")
lapis_names_list, lapis_embeddings = _load("lapis")

model_name_or_path = "google/gemini-embedding-2-preview"
model = GeminiEmbedder(model=model_name_or_path)

ava_embeddings_tensor = torch.tensor(ava_embeddings).float()
ls_embeddings_tensor = torch.tensor(ls_embeddings).float()
lapis_embeddings_tensor = torch.tensor(lapis_embeddings).float()

dataset_map = {
    "photos": "ava",
    "dreamcore": "ls",
    "artwork": "lapis",
}


def dataset_loader_summary() -> dict:
    return {
        "model_name_or_path": model_name_or_path,
        "total_rows": int(len(ava_names_list) + len(ls_names_list) + len(lapis_names_list)),
        "ava_count": int(len(ava_names_list)),
        "dreamcore_count": int(len(ls_names_list)),
        "artwork_count": int(len(lapis_names_list)),
        "embedding_dim": int(ava_embeddings_tensor.shape[1]) if len(ava_embeddings_tensor.shape) > 1 else 0,
    }
