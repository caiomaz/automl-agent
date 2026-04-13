"""Unit tests for the CLI module."""

import pytest
from unittest.mock import patch


class TestCLIListModels:
    def test_list_models_runs(self, capsys):
        from cli import cmd_list_models
        cmd_list_models()
        captured = capsys.readouterr()
        assert "or-glm-5" in captured.out
        assert "or-claude-sonnet" in captured.out
        assert "AutoML-Agent" in captured.out

    def test_list_models_shows_all_or_aliases(self, capsys):
        from cli import cmd_list_models
        from configs import AVAILABLE_LLMs
        cmd_list_models()
        captured = capsys.readouterr()
        for alias in AVAILABLE_LLMs.list():
            if alias.startswith("or-"):
                assert alias in captured.out


class TestCLIArgParsing:
    def test_list_models_command(self):
        import sys
        from cli import main
        with patch.object(sys, "argv", ["cli", "list-models"]):
            # Should not raise
            main()
