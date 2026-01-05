import os
import numpy as np
from typing import List, Union
from langchain_community.embeddings import OpenAIEmbeddings

# For local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class OpenAILongerThanContextEmb:
    """
    Embedding function with openai as embedding backend.
    If the input is larger than the context size, the input is split into chunks of size `chunk_size` and embedded separately.
    The final embedding is the average of the embeddings of the chunks.
    Details see: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    """

    def __init__(
        self,
        openai_api_key: Union[str, None] = None,
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 5000,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the Embedding object.

        Args:
            openai_api_key (str): The API key for OpenAI.
            embedding_model (str, optional): The model to use for embedding. Defaults to "text-embedding-ada-002".
            chunk_size (int, optional): The maximum number of token to send to openai embedding model at one time. Defaults to 5000.
            verbose (bool, optional): Whether to show progress bar during embedding. Defaults to False.

        Returns:
            None
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.emb_model = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
            chunk_size=chunk_size,
            show_progress_bar=verbose,
        )

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        """
        Asynchronously performs embedding on a list of text.

        This method calls the `aembed_documents` method of the `emb_model` object to embed the input text.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            List[List[float]]: The embeddings of the input text as a list of lists of floats.

        """
        if isinstance(text, str):
            text = [text]
        return self.emb_model.embed_documents(texts=text, chunk_size=None)

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        """
        Performs embedding on a list of text.

        This method calls the `_emb` method to asynchronously embed the input text using the `emb_model` object.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            np.array: The embedding of the input text as a NumPy array.

        """
        return np.array(self._emb(text)).astype("float32")

    def get_embedding_dimension(self):
        """
        Returns the dimension of the embedding.

        This method checks the value of `self.emb_model.model` and returns the corresponding embedding dimension. If the model is not implemented, a `NotImplementedError` is raised.

        Args:
            self: The instance of the class.

        Returns:
            int: The dimension of the embedding.

        Raises:
            NotImplementedError: Raised when the embedding dimension for the specified model is not implemented.

        """
        match self.emb_model.model:
            case "text-embedding-ada-002":
                return 1536
            case _:
                raise NotImplementedError(
                    f"Embedding dimension for model {self.emb_model.model} not implemented"
                )


class LocalEmbedding:
    """
    Local embedding function using sentence-transformers models.
    Runs completely locally without any API calls - 100% free and private.

    Recommended models:
    - 'BAAI/bge-small-en-v1.5' - Good quality, 384 dimensions, fast
    - 'sentence-transformers/all-MiniLM-L6-v2' - Very fast, 384 dimensions
    - 'BAAI/bge-base-en-v1.5' - Better quality, 768 dimensions
    - 'intfloat/e5-small-v2' - Good performance, 384 dimensions
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        device: str = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the Local Embedding object.

        Args:
            embedding_model (str): HuggingFace model ID. Defaults to "BAAI/bge-small-en-v1.5".
            device (str, optional): Device to use ('cuda', 'cpu', or None for auto). Defaults to None.
            verbose (bool): Whether to show progress during model loading. Defaults to False.

        Returns:
            None
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = embedding_model
        self.verbose = verbose

        if verbose:
            print(f"Loading local embedding model: {embedding_model}")

        self.emb_model = SentenceTransformer(embedding_model, device=device)

        if verbose:
            print(f"✓ Model loaded successfully")
            print(f"  Embedding dimension: {self.get_embedding_dimension()}")
            print(f"  Device: {self.emb_model.device}")

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        """
        Performs embedding on text.

        Args:
            text (Union[List[str], str]): A text string or list of texts to be embedded.

        Returns:
            List[List[float]]: The embeddings of the input text.
        """
        if isinstance(text, str):
            text = [text]

        # SentenceTransformer.encode returns numpy array, convert to list
        embeddings = self.emb_model.encode(
            text,
            show_progress_bar=self.verbose,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        """
        Performs embedding on a list of text.

        Args:
            text (Union[List[str], str]): A text string or list of texts to be embedded.

        Returns:
            np.ndarray: The embedding of the input text as a NumPy array.
        """
        return np.array(self._emb(text)).astype("float32")

    def get_embedding_dimension(self) -> int:
        """
        Returns the dimension of the embedding.

        Returns:
            int: The dimension of the embedding.
        """
        return self.emb_model.get_sentence_embedding_dimension()
