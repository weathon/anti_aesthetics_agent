"""Dataset and embedding loader.

Loads pre-computed embeddings from HuggingFace, initialises the
Qwen3-VL-Embedding-8B model, and prepares tensors for cosine-similarity
search over the AVA / liminal-space / lapis subsets.
"""

import dotenv
dotenv.load_dotenv()

import datasets
import numpy as np
import torch

from qwen3_vl_embedding import Qwen3VLEmbedder

ds = datasets.load_dataset("weathon/ava_embeddings", split="train")
ds = ds.with_format("numpy")

names = ds["name"]
sources = ds["source"]
arr = ds.data.column("embeddings").to_numpy()
arr = np.stack(arr, axis=0)

ava_dataset = ds.filter(lambda example: example["source"] == "ava")
ls_dataset = ds.filter(lambda example: example["source"] == "liminal_space")
lapis_dataset = ds.filter(lambda example: example["source"] == "lapis")

ava_embeddings = np.stack(ava_dataset["embeddings"], axis=0)
ls_embeddings = np.stack(ls_dataset["embeddings"], axis=0)
lapis_embeddings = np.stack(lapis_dataset["embeddings"], axis=0)

ava_names = ava_dataset["name"]
ls_names = ls_dataset["name"]
lapis_names = lapis_dataset["name"]

model_name_or_path = "Qwen/Qwen3-VL-Embedding-8B"
model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path, device="cpu", attn_implementation="sdpa")

ava_embeddings_tensor = torch.tensor(ava_embeddings).float()
ls_embeddings_tensor = torch.tensor(ls_embeddings).float()
lapis_embeddings_tensor = torch.tensor(lapis_embeddings).float()

ava_names_list = list(ava_names)
ls_names_list = list(ls_names)
lapis_names_list = list(lapis_names)

dataset_map = {
    "photos": "ava",
    "dreamcore": "ls",
    "artwork": "lapis",
}


def dataset_loader_summary() -> dict:
    return {
        "model_name_or_path": model_name_or_path,
        "total_rows": int(len(names)),
        "ava_count": int(len(ava_names_list)),
        "dreamcore_count": int(len(ls_names_list)),
        "artwork_count": int(len(lapis_names_list)),
        "embedding_dim": int(ava_embeddings_tensor.shape[1]) if len(ava_embeddings_tensor.shape) > 1 else 0,
    }
