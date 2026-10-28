"""Generation contract checks with a fake tokenizer/model, without weights or torch."""

from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest

from backend.config import Settings
from backend.llm import LocalCausalLMGenerator


class TokenRows:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        rows, columns = key
        return TokenRows([row[columns] for row in self.rows[rows]])


class Inputs(dict):
    def to(self, device):
        assert device == "cpu"
        return self


class FakeTokenizer:
    chat_template = "qwen-style-test-template"

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [
            {"role": "system", "content": "Use the references."},
            {"role": "user", "content": "What is RRF?"},
        ]
        assert kwargs == {
            "add_generation_prompt": True, "tokenize": True,
            "return_dict": True, "return_tensors": "pt",
        }
        return Inputs(input_ids=TokenRows([[10, 11, 12]]))

    def __call__(self, prompt, **kwargs):
        assert prompt == "System: Use the references.\n\nUser: What is RRF?\n\nAssistant:"
        assert kwargs == {"return_tensors": "pt"}
        return Inputs(input_ids=TokenRows([[10, 11, 12]]))

    def batch_decode(self, tokens, **kwargs):
        assert tokens.rows == [[77, 88]]  # The prompt must not leak into the answer.
        assert kwargs == {"skip_special_tokens": True}
        return ["  Rank fusion.  "]


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 20
        assert kwargs["do_sample"] is False
        return TokenRows([[10, 11, 12, 77, 88]])


@pytest.mark.parametrize("has_template", [True, False])
def test_generation_uses_text_messages_and_decodes_only_continuation(monkeypatch, has_template):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    generator = LocalCausalLMGenerator(Settings())
    generator.tokenizer = FakeTokenizer()
    if not has_template:
        generator.tokenizer.chat_template = None
    generator.model = FakeModel()

    assert generator.generate("Use the references.", "What is RRF?", max_new_tokens=20) == "Rank fusion."


def test_cpu_four_bit_fails_before_loading_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    generator = LocalCausalLMGenerator(Settings(device="cpu", load_in_4bit=True))
    with pytest.raises(ValueError, match="RAG_LOAD_IN_4BIT=false"):
        generator._load()
    assert generator.model is None
    assert generator.tokenizer is None


def test_context_overflow_fails_before_generation(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    generator = LocalCausalLMGenerator(Settings())
    generator.tokenizer = FakeTokenizer()
    generator.model = FakeModel()
    generator.model.config = SimpleNamespace(max_position_embeddings=22)
    with pytest.raises(ValueError, match="context window"):
        generator.generate("Use the references.", "What is RRF?", max_new_tokens=20)
