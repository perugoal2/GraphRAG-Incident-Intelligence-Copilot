from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim


def embed(text: str) -> list:
    return _model.encode(text).tolist()
