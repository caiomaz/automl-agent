"""Unit tests for utils.get_client — verifies OpenAI client creation logic."""

import pytest
from unittest.mock import patch, MagicMock


class TestGetClient:
    @patch("utils.OpenAI")
    @patch("utils.AVAILABLE_LLMs")
    def test_client_with_base_url(self, mock_registry, mock_openai):
        mock_registry.__getitem__ = MagicMock(return_value={
            "api_key": "test-key",
            "model": "test-model",
            "base_url": "http://localhost:8000/v1",
        })
        from utils import get_client
        client = get_client("some-local")
        mock_openai.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="test-key",
        )

    @patch("utils.OpenAI")
    @patch("utils.AVAILABLE_LLMs")
    def test_client_without_base_url(self, mock_registry, mock_openai):
        mock_registry.__getitem__ = MagicMock(return_value={
            "api_key": "sk-openai",
            "model": "gpt-4o",
        })
        from utils import get_client
        client = get_client("gpt-4")
        mock_openai.assert_called_once_with(api_key="sk-openai")

    @patch("utils.AVAILABLE_LLMs")
    def test_unknown_llm_raises(self, mock_registry):
        mock_registry.__getitem__ = MagicMock(side_effect=KeyError("not found"))
        from utils import get_client
        with pytest.raises(KeyError):
            get_client("nonexistent-llm")


class TestPrintMessage:
    def test_print_message_all_senders(self, capsys):
        from utils import print_message
        for sender in ("user", "system", "manager", "model", "data", "prompt", "operation"):
            print_message(sender, "hello")
            captured = capsys.readouterr()
            assert "hello" in captured.out
