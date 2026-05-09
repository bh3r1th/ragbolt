import pytest

from ragbolt.core.generator import GenerationProvider, StubGenerationProvider
from ragbolt.core.providers import AnthropicGenerationProvider, OpenAIGenerationProvider


def test_anthropic_missing_key() -> None:
    config = {}
    with pytest.raises(ValueError):
        AnthropicGenerationProvider(config)


def test_openai_missing_key() -> None:
    config = {}
    with pytest.raises(ValueError):
        OpenAIGenerationProvider(config)


def test_stub_provider_satisfies_protocol() -> None:
    assert isinstance(StubGenerationProvider(), GenerationProvider)
