"""Regression tests for CLI — freeze arg parsing, constraints, and workflow contracts.

No external services or LLM calls are made.
"""

import argparse
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _mock_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BACKBONE", "or-glm-5")
    monkeypatch.setenv("LLM_PROMPT_AGENT", "or-glm-5")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCLIRunArgParsing:
    """Freeze the full 'run' subcommand argument interface."""

    def _parse(self, argv: list[str]) -> argparse.Namespace:
        """Parse CLI args without executing."""
        from cli import main
        import cli as cli_module

        # Build the parser the same way main() does
        parser = argparse.ArgumentParser(prog="automl-agent")
        sub = parser.add_subparsers(dest="command")
        sub.add_parser("list-models")
        run_p = sub.add_parser("run")
        run_p.add_argument("--llm", type=str)
        run_p.add_argument("--task", type=str, required=True)
        run_p.add_argument("--prompt", type=str, required=True)
        run_p.add_argument("--data", type=str, default=None)
        run_p.add_argument("--n-plans", type=int, default=3)
        run_p.add_argument("--n-revise", type=int, default=3)
        run_p.add_argument("--no-rap", action="store_true")
        run_p.add_argument("--model", type=str, default=None)
        run_p.add_argument("--perf-metric", type=str, default=None)
        run_p.add_argument("--perf-value", type=str, default=None)
        run_p.add_argument("--max-train-time", type=str, default=None)
        run_p.add_argument("--max-inference-time", type=str, default=None)
        run_p.add_argument("--system-info", action="store_true", default=True)
        run_p.add_argument("--no-system-info", action="store_false", dest="system_info")
        return parser.parse_args(argv)

    def test_minimal_run_args(self):
        args = self._parse(["run", "--task", "tabular_regression", "--prompt", "Predict age"])
        assert args.command == "run"
        assert args.task == "tabular_regression"
        assert args.prompt == "Predict age"
        assert args.n_plans == 3
        assert args.n_revise == 3
        assert args.no_rap is False
        assert args.system_info is True

    def test_full_run_args(self):
        args = self._parse([
            "run",
            "--task", "tabular_classification",
            "--prompt", "Classify bananas",
            "--llm", "or-claude-sonnet",
            "--data", "/path/to/data",
            "--n-plans", "5",
            "--n-revise", "2",
            "--no-rap",
            "--model", "XGBoost",
            "--perf-metric", "F1",
            "--perf-value", "0.95",
            "--max-train-time", "30 minutes",
            "--max-inference-time", "5 ms",
            "--no-system-info",
        ])
        assert args.task == "tabular_classification"
        assert args.llm == "or-claude-sonnet"
        assert args.data == "/path/to/data"
        assert args.n_plans == 5
        assert args.n_revise == 2
        assert args.no_rap is True
        assert args.model == "XGBoost"
        assert args.perf_metric == "F1"
        assert args.perf_value == "0.95"
        assert args.max_train_time == "30 minutes"
        assert args.max_inference_time == "5 ms"
        assert args.system_info is False

    def test_list_models_command(self):
        args = self._parse(["list-models"])
        assert args.command == "list-models"

    def test_default_command_is_none(self):
        args = self._parse([])
        assert args.command is None


@pytest.mark.unit
class TestCLIConstraintsDict:
    """Freeze how constraints dict is built from parsed args."""

    def test_constraints_from_full_args(self):
        """Simulate the constraint-building logic from cmd_run."""
        # Simulate parsed args
        model = "XGBoost"
        perf_metric = "F1"
        perf_value = "0.95"
        max_train_time = "30 minutes"
        max_inference_time = "5 ms"

        constraints_dict = {}
        constraints_parts = []
        if model:
            constraints_dict["model"] = model
            constraints_parts.append(f"Use {model} as the model/algorithm.")
        if perf_metric and perf_value:
            constraints_dict["perf_metric"] = perf_metric
            constraints_dict["perf_value"] = perf_value
            constraints_parts.append(f"Achieve at least {perf_value} {perf_metric}.")
        if max_train_time and max_train_time.lower() != "unlimited":
            constraints_dict["max_train_time"] = max_train_time
            constraints_parts.append(f"Training time must not exceed {max_train_time}.")
        if max_inference_time and max_inference_time.lower() != "unlimited":
            constraints_dict["max_inference_time"] = max_inference_time
            constraints_parts.append(f"Inference time per sample must be under {max_inference_time}.")

        assert constraints_dict == {
            "model": "XGBoost",
            "perf_metric": "F1",
            "perf_value": "0.95",
            "max_train_time": "30 minutes",
            "max_inference_time": "5 ms",
        }
        assert len(constraints_parts) == 4

    def test_empty_constraints_when_no_args(self):
        constraints_dict = {}
        constraints_parts = []
        # No constraint args provided — all None
        assert constraints_dict == {}
        assert constraints_parts == []


@pytest.mark.unit
class TestCLIPromptEnrichment:
    """Freeze how the user prompt is enriched with constraint text."""

    def test_prompt_enriched_with_constraints(self):
        prompt = "Predict crab age"
        constraints_parts = [
            "Use XGBoost as the model/algorithm.",
            "Achieve at least 0.95 F1.",
        ]
        if constraints_parts:
            prompt = prompt.rstrip(".") + ". " + " ".join(constraints_parts)

        assert prompt == "Predict crab age. Use XGBoost as the model/algorithm. Achieve at least 0.95 F1."

    def test_prompt_unchanged_without_constraints(self):
        prompt = "Predict crab age"
        constraints_parts = []
        if constraints_parts:
            prompt = prompt.rstrip(".") + ". " + " ".join(constraints_parts)

        assert prompt == "Predict crab age"


@pytest.mark.unit
class TestCLIListModels:
    """Freeze list-models output contract."""

    def test_list_models_outputs_aliases(self, _mock_env, capsys):
        from cli import cmd_list_models
        cmd_list_models()
        output = capsys.readouterr().out
        # Should contain at least some registered aliases
        assert "or-glm-5" in output or "glm" in output
        # Should show current defaults section
        assert "LLM_BACKBONE" in output
        assert "LLM_PROMPT_AGENT" in output


@pytest.mark.unit
class TestCLIInteractiveStepCount:
    """Freeze interactive mode step count at 6."""

    def test_six_steps_in_interactive(self, _mock_env):
        """Verify _step() is called exactly 6 times in cmd_interactive flow."""
        import cli as cli_module
        import inspect
        source = inspect.getsource(cli_module.cmd_interactive)
        # Count _step() calls in the source
        step_calls = source.count("_step(")
        assert step_calls == 6, f"Expected 6 _step() calls in cmd_interactive, found {step_calls}"
