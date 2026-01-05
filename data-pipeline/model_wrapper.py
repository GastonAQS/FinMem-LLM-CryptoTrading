from __future__ import annotations
from abc import ABC, abstractmethod
import openai
from tenacity import retry, stop_after_attempt, wait_fixed
from llama_cpp import Llama

MAX_ATTEMPTS = 5
WAIT_TIME = 10

class ModelWrapper(ABC):
    @retry(stop=stop_after_attempt(MAX_ATTEMPTS), wait=wait_fixed(WAIT_TIME))
    def summarize(self, text, summary_token_size = 200):
        return self._summarize(text, summary_token_size)
    
    @abstractmethod
    def _summarize(self, text, summary_token_size):
        pass
    
class Chatgpt(ModelWrapper):
    def __init__(self, key, model_name):
        self.__key = key
        openai.api_key = self.__key
        self.model_name = model_name
    
    def _summarize(self, text, summary_token_size):
        prompt = f"Summarize the following news within {summary_token_size} tokens:\n{text}\nSummary:"
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        summary = response['choices'][0]['message']['content']
        return summary


class LlamaModel(ModelWrapper):
    def __init__(self, model_name=None):
        self.model_name = model_name

        # Default to base Llama-3-8B-Instruct for summarization
        # The FinGPT-MT-Llama-3-8B-LoRA model is for sentiment analysis, not summarization
        if model_name is None:
            repo_id = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
            filename = "Meta-Llama-3-8B-Instruct-Q5_K_M.gguf"
        else:
            # Allow custom model specification
            repo_id = model_name.split("/")[0] + "/" + model_name.split("/")[1]
            filename = model_name.split("/")[2] if len(model_name.split("/")) > 2 else "*.gguf"

        # Try GPU first, fall back to CPU if CUDA not available
        try:
            self.model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_ctx=8192,
                n_batch=512,
                verbose=False
            )
            print(f"✓ Model loaded with GPU acceleration: {repo_id}")
        except Exception as e:
            print(f"⚠ GPU not available ({e}), falling back to CPU")
            self.model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=8192,
                n_threads=8,  # Use multiple CPU threads
                verbose=False
            )
            print(f"✓ Model loaded with CPU: {repo_id}")

    def _summarize(self, text, summary_token_size):
        # Use Llama-3's chat completion for better instruction following
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a financial news summarization assistant. Output only the summary text without any preamble, introduction, or phrases like 'Here's a summary'. Start directly with the summary content."},
                {"role": "user", "content": f"Summarize the following news within {summary_token_size} tokens:\n{text}\nSummary:"}
            ],
            max_tokens=summary_token_size,
            temperature=0.7,
            top_p=0.95,
        )

        result = response["choices"][0]["message"]["content"].strip()
        return result

    
class Dummy(ModelWrapper):
    '''
    For test only
    '''
    import random
    import time
    def __init__(self, *args, **kwargs) -> None:
        print("Initializing a dummy model!")
    
    def _summarize(self, text, summary_token_size):
        self.time.sleep(self.random.randint(1, 5))
        if self.random.random() < 0.1:
            print("attempt", summary_token_size)
            raise
        else:
            return text[:summary_token_size]
    
class ModelFactory:
    registered_model_class = ("chatgpt", 'together', 'dummy', "llama")
    @classmethod
    def create_model(cls, model_class:str, key:str = None, model_name:str = None, *args, **kwargs)->(Chatgpt | LlamaModel | Dummy):
        assert model_class in cls.registered_model_class, f"Invalid model class name: choose one from {cls.registered_model_class}"
        match model_class:
            case "chatgpt":
                return Chatgpt(key, model_name)
            case "dummy":
                return Dummy()
            case "llama":
                return LlamaModel(model_name)
            case _:
                raise
    