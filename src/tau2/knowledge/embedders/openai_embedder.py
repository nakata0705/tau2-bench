"""OpenAI embedder using text-embedding models."""

from typing import List

import numpy as np
from openai import OpenAI

from tau2.config import (
    openai_client_kwargs,
    resolve_openai_compatible_model,
)
from tau2.knowledge.embedders.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's embedding models."""

    def __init__(
        self, model: str = "text-embedding-ada-002", api_key: str | None = None
    ):
        """
        Initialize OpenAI embedder.

        Args:
            model: OpenAI model name. Supported models include:
                   - text-embedding-ada-002 (default, 1536 dimensions)
                   - text-embedding-3-small (1536 dimensions)
                   - text-embedding-3-large (3072 dimensions)
            api_key: Optional direct API key. With no explicit key, the
                      configured OpenRouter/OpenAI environment is selected.
        """
        self.requested_model = model
        self.model = resolve_openai_compatible_model(model, explicit_api_key=api_key)
        self.client = OpenAI(**openai_client_kwargs(api_key=api_key))

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using OpenAI API.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            raise ValueError("No text to embed.")

        response = self.client.embeddings.create(input=texts, model=self.model)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def get_name(self) -> str:
        """Return the name of the embedder."""
        return f"openai_{self.requested_model}"
