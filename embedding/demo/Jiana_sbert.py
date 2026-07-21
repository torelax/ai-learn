from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small-retrieval",
    model_kwargs={"dtype": torch.bfloat16},
)

query_emb = model.encode("What is knowledge distillation?", prompt_name="query")
doc_embs = model.encode(["Knowledge distillation transfers...", "Venus is..."], prompt_name="document")
similarity = model.similarity(query_emb, doc_embs)