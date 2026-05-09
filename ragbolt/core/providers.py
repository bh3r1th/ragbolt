import json
import urllib.error
import urllib.request

from ragbolt.core.generator import GenerationError, GenerationProvider


_SYSTEM_PROMPT = (
    "You are a precise assistant. Answer only using the provided context. "
    "Do not add information not present in the context."
)


class AnthropicGenerationProvider:
    def __init__(self, config: dict, stream: bool = False):
        api_key = str(config.get("anthropic_api_key", "")).strip()
        if not api_key:
            raise ValueError("anthropic_api_key is required")
        self._api_key = api_key
        self._model = str(config.get("anthropic_model", "claude-sonnet-4-20250514"))
        self._max_tokens = int(config.get("max_tokens", 1024))
        self.stream = bool(config.get("stream_generation", stream))

    def _stream_generate(self, query: str, context: str):
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": 0,
            "stream": True,
            "system": _SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                }
            ],
        }
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if chunk.get("type") == "content_block_delta":
                                yield chunk.get("delta", {}).get("text", "")
                        except json.JSONDecodeError:
                            continue
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise GenerationError(f"Anthropic API error: {e.code} {body}") from e
        except urllib.error.URLError as e:
            raise GenerationError(f"Anthropic API error: network {e.reason}") from e
        except OSError as e:
            raise GenerationError(f"Anthropic API error: stream {e}") from e

    def generate(self, query: str, context: str) -> str:
        if self.stream:
            collected = list(self._stream_generate(query, context))
            text = "".join(collected).strip()
            if not text:
                raise GenerationError("Anthropic returned empty response")
            return text

        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": 0,
            "system": _SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                }
            ],
        }
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                body = response.read().decode("utf-8")
                status = response.status
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise GenerationError(f"Anthropic API error: {e.code} {body}") from e
        except urllib.error.URLError as e:
            raise GenerationError(f"Anthropic API error: network {e.reason}") from e
        if status != 200:
            raise GenerationError(f"Anthropic API error: {status} {body}")
        payload = json.loads(body)
        content = payload.get("content") or []
        if not content or not content[0].get("text", "").strip():
            raise GenerationError("Anthropic returned empty response")
        return content[0]["text"].strip()


class OpenAIGenerationProvider:
    def __init__(self, config: dict, stream: bool = False):
        api_key = str(config.get("openai_api_key", "")).strip()
        if not api_key:
            raise ValueError("openai_api_key is required")
        self._api_key = api_key
        self._model = str(config.get("openai_model", "gpt-4o-mini"))
        self._max_tokens = int(config.get("max_tokens", 1024))
        self.stream = bool(config.get("stream_generation", stream))

    def _stream_generate(self, query: str, context: str):
        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "stream": True,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise GenerationError(f"OpenAI API error: {e.code} {body}") from e
        except urllib.error.URLError as e:
            raise GenerationError(f"OpenAI API error: network {e.reason}") from e
        except OSError as e:
            raise GenerationError(f"OpenAI API error: stream {e}") from e

    def generate(self, query: str, context: str) -> str:
        if self.stream:
            collected = list(self._stream_generate(query, context))
            text = "".join(collected).strip()
            if not text:
                raise GenerationError("OpenAI returned empty response")
            return text

        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                body = response.read().decode("utf-8")
                status = response.status
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise GenerationError(f"OpenAI API error: {e.code} {body}") from e
        except urllib.error.URLError as e:
            raise GenerationError(f"OpenAI API error: network {e.reason}") from e
        if status != 200:
            raise GenerationError(f"OpenAI API error: {status} {body}")
        payload = json.loads(body)
        choices = payload.get("choices") or []
        if not choices:
            raise GenerationError("OpenAI returned empty response")
        content = choices[0].get("message", {}).get("content", "").strip()
        if not content:
            raise GenerationError("OpenAI returned empty response")
        return content
