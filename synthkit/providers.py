"""Response + embedding providers.

A *provider* turns a prompt into a response (for instruction→output pairs); an
*embedder* turns text into a vector (for the optional semantic-quality axis).
The 'none' path is stdlib; ollama is local & free; anthropic/openai are lazy
imports used only if selected.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List, Optional

from synthkit.models import SynthkitError

# ---- HTTP helper with friendly Ollama errors ---------------------------------

def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:200]
        hint = ""
        if exc.code == 404 and "model" in detail.lower():
            hint = " — pull it first with `ollama pull <model>`"
        raise SynthkitError(f"Ollama returned HTTP {exc.code} from {url}{hint}\n  {detail}")
    except urllib.error.URLError as exc:
        raise SynthkitError(
            f"can't reach Ollama at {url} ({exc.reason}). "
            "Is the daemon running? Start it with `ollama serve`.")


# ---- response providers ------------------------------------------------------

class Provider:
    name = "base"

    def generate(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError


class OllamaProvider(Provider):
    """Local, free responses via a running Ollama daemon."""

    name = "ollama"

    def __init__(self, model: str = "llama3.2", host: str = "") -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def generate(self, prompt: str, system: str = "") -> str:
        data = _post_json(self.host + "/api/generate", {
            "model": self.model, "prompt": prompt,
            "system": system, "stream": False,
        })
        return (data.get("response") or "").strip()


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise SynthkitError("--provider anthropic needs the anthropic SDK — "
                             "`pip install anthropic`.") from exc
        self.model = model
        self._client = anthropic.Anthropic()

    def generate(self, prompt: str, system: str = "") -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=1024,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content
                       if getattr(b, "type", "") == "text").strip()


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            import openai
        except ImportError as exc:
            raise SynthkitError("--provider openai needs the openai SDK — "
                             "`pip install openai`.") from exc
        self.model = model
        self._client = openai.OpenAI()

    def generate(self, prompt: str, system: str = "") -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system or "You are a helpful assistant."},
                      {"role": "user", "content": prompt}])
        return (resp.choices[0].message.content or "").strip()


def get_provider(name: Optional[str], model: str = "") -> Optional[Provider]:
    if not name or name == "none":
        return None
    if name == "ollama":
        return OllamaProvider(model or "llama3.2")
    if name == "anthropic":
        return AnthropicProvider(model or "claude-haiku-4-5-20251001")
    if name == "openai":
        return OpenAIProvider(model or "gpt-4o-mini")
    raise SynthkitError(f"unknown provider {name!r}")


# ---- embedders (for the optional semantic axis) ------------------------------

class Embedder:
    name = "base"

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(self, model: str = "nomic-embed-text", host: str = "") -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            data = _post_json(self.host + "/api/embeddings",
                              {"model": self.model, "prompt": t})
            vec = data.get("embedding")
            if not vec:
                raise SynthkitError(f"Ollama embedder returned no vector for model {self.model!r}")
            out.append(vec)
        return out


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        try:
            import openai
        except ImportError as exc:
            raise SynthkitError("--embed-provider openai needs the openai SDK — "
                             "`pip install openai`.") from exc
        self.model = model
        self._client = openai.OpenAI()

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def get_embedder(name: Optional[str], model: str = "") -> Optional[Embedder]:
    if not name or name == "none":
        return None
    if name == "ollama":
        return OllamaEmbedder(model or "nomic-embed-text")
    if name == "openai":
        return OpenAIEmbedder(model or "text-embedding-3-small")
    raise SynthkitError(f"unknown embed provider {name!r}")
