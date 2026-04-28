"""Phase 8 — Token economy, stage routing and run-scoped cache (TDD).

Covers:

- ``utils.token_economy``: payload truncation, error summarization,
  dynamic budget for ``n_plans``/``n_revise`` based on policy +
  confidence, and ledger ``tokens_saved`` event.
- ``utils.stage_routing``: per-stage LLM alias resolution honoring
  ``LLM_STAGE_<NAME>`` env vars with default fallback.
- ``utils.run_cache``: deterministic content-addressed cache with
  in-memory + on-disk backing for parse/summary reuse.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _reset_active_run():
    from utils.run_context import clear_active_run
    clear_active_run()
    yield
    clear_active_run()


# ── Token economy ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPolicyValidation:
    def test_invalid_policy_rejected(self):
        from utils.token_economy import normalize_policy

        with pytest.raises(ValueError, match="token_economy"):
            normalize_policy("super-aggressive")

    def test_off_moderate_aggressive_accepted(self):
        from utils.token_economy import normalize_policy

        assert normalize_policy("off") == "off"
        assert normalize_policy("moderate") == "moderate"
        assert normalize_policy("aggressive") == "aggressive"

    def test_none_defaults_to_off(self):
        from utils.token_economy import normalize_policy

        assert normalize_policy(None) == "off"


@pytest.mark.unit
class TestTruncatePayload:
    def test_off_returns_text_unchanged(self):
        from utils.token_economy import truncate_payload

        text = "x" * 10_000
        out, info = truncate_payload(text, policy="off")
        assert out == text
        assert info["truncated"] is False
        assert info["saved_chars"] == 0

    def test_moderate_truncates_long_text(self):
        from utils.token_economy import truncate_payload

        text = "abcdef" * 5_000  # 30k chars
        out, info = truncate_payload(text, policy="moderate")
        assert len(out) < len(text)
        assert info["truncated"] is True
        assert info["saved_chars"] > 0
        # Truncation marker visible.
        assert "truncated" in out.lower() or "..." in out

    def test_aggressive_truncates_more_than_moderate(self):
        from utils.token_economy import truncate_payload

        text = "abcdef" * 5_000
        out_mod, _ = truncate_payload(text, policy="moderate")
        out_agg, _ = truncate_payload(text, policy="aggressive")
        assert len(out_agg) < len(out_mod)

    def test_short_text_never_truncated(self):
        from utils.token_economy import truncate_payload

        text = "short message"
        out, info = truncate_payload(text, policy="aggressive")
        assert out == text
        assert info["truncated"] is False


@pytest.mark.unit
class TestSummarizeError:
    def test_off_returns_unchanged(self):
        from utils.token_economy import summarize_error

        stderr = "\n".join(f"line{i}" for i in range(500))
        out, info = summarize_error(stderr, policy="off")
        assert out == stderr
        assert info["truncated"] is False

    def test_moderate_keeps_head_and_tail(self):
        from utils.token_economy import summarize_error

        stderr = "\n".join(f"line{i}" for i in range(500))
        out, info = summarize_error(stderr, policy="moderate")
        assert "line0" in out  # first lines kept
        assert "line499" in out  # last lines kept
        assert info["truncated"] is True
        assert info["original_lines"] == 500
        assert info["kept_lines"] < 500

    def test_aggressive_keeps_fewer_lines(self):
        from utils.token_economy import summarize_error

        stderr = "\n".join(f"line{i}" for i in range(500))
        out_mod, info_mod = summarize_error(stderr, policy="moderate")
        out_agg, info_agg = summarize_error(stderr, policy="aggressive")
        assert info_agg["kept_lines"] < info_mod["kept_lines"]


@pytest.mark.unit
class TestDynamicBudget:
    def test_off_returns_default(self):
        from utils.token_economy import dynamic_n_plans

        assert dynamic_n_plans(default=5, policy="off") == 5
        assert dynamic_n_plans(default=5, policy="off", confidence=0.99) == 5

    def test_moderate_reduces_when_confidence_high(self):
        from utils.token_economy import dynamic_n_plans

        assert dynamic_n_plans(default=5, policy="moderate", confidence=0.95) < 5

    def test_moderate_keeps_default_when_confidence_low(self):
        from utils.token_economy import dynamic_n_plans

        assert dynamic_n_plans(default=5, policy="moderate", confidence=0.4) == 5

    def test_aggressive_reduces_more_than_moderate(self):
        from utils.token_economy import dynamic_n_plans

        mod = dynamic_n_plans(default=8, policy="moderate", confidence=0.95)
        agg = dynamic_n_plans(default=8, policy="aggressive", confidence=0.95)
        assert agg <= mod

    def test_never_drops_below_one(self):
        from utils.token_economy import dynamic_n_plans

        assert dynamic_n_plans(default=1, policy="aggressive", confidence=0.99) == 1


@pytest.mark.unit
class TestTokensSavedEvent:
    def test_record_tokens_saved_emits_event(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.token_economy import record_tokens_saved
        from utils.workspace import run_exp_dir

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        record_tokens_saved(
            ctx,
            source="cache_hit",
            saved_tokens=1234,
            workspace=tmp_path,
            stage="parse",
        )
        events_file = run_exp_dir(ctx.run_id, tmp_path) / "events.jsonl"
        events = [json.loads(line) for line in events_file.read_text().splitlines() if line]
        saved = [e for e in events if e["event"] == "tokens_saved"]
        assert len(saved) == 1
        assert saved[0]["saved_tokens"] == 1234
        assert saved[0]["source"] == "cache_hit"
        assert saved[0]["stage"] == "parse"
        finalize_run(ctx, status="completed", workspace=tmp_path)


# ── Stage routing ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStageRouting:
    def test_default_alias_when_no_override(self, monkeypatch):
        from utils.stage_routing import resolve_stage_alias

        monkeypatch.delenv("LLM_STAGE_PROMPT_PARSE", raising=False)
        assert resolve_stage_alias("prompt_parse", default="or-glm-5") == "or-glm-5"

    def test_env_override_wins(self, monkeypatch):
        from utils.stage_routing import resolve_stage_alias

        monkeypatch.setenv("LLM_STAGE_CRITIC", "or-deepseek-v3.1")
        assert resolve_stage_alias("critic", default="or-glm-5") == "or-deepseek-v3.1"

    def test_known_stages_listed(self):
        from utils.stage_routing import KNOWN_STAGES

        assert "prompt_parse" in KNOWN_STAGES
        assert "critic" in KNOWN_STAGES
        assert "planning" in KNOWN_STAGES
        assert "code_generation" in KNOWN_STAGES
        assert "verification" in KNOWN_STAGES
        assert "summary" in KNOWN_STAGES

    def test_unknown_stage_rejected(self):
        from utils.stage_routing import resolve_stage_alias

        with pytest.raises(ValueError, match="Unknown stage"):
            resolve_stage_alias("not_a_stage", default="or-glm-5")

    def test_stage_routing_map_returns_active_overrides(self, monkeypatch):
        from utils.stage_routing import current_routing_map

        monkeypatch.setenv("LLM_STAGE_PROMPT_PARSE", "or-glm-5")
        monkeypatch.delenv("LLM_STAGE_CRITIC", raising=False)
        m = current_routing_map(default="or-default")
        assert m["prompt_parse"] == "or-glm-5"
        assert m["critic"] == "or-default"


# ── Run-scoped cache ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRunCache:
    def test_make_key_is_deterministic(self):
        from utils.run_cache import make_key

        a = make_key("parse", "hello world")
        b = make_key("parse", "hello world")
        c = make_key("parse", "hello there")
        assert a == b
        assert a != c

    def test_get_returns_none_on_miss(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.run_cache import RunCache

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        cache = RunCache(ctx, workspace=tmp_path)
        assert cache.get("nonexistent") is None
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_set_then_get_roundtrip(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.run_cache import RunCache

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        cache = RunCache(ctx, workspace=tmp_path)
        cache.set("k1", {"task": "tabular_regression"}, tokens_estimate=42)
        got = cache.get("k1")
        assert got == {"task": "tabular_regression"}
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_set_persists_to_disk(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.run_cache import RunCache
        from utils.workspace import run_exp_dir

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        cache = RunCache(ctx, workspace=tmp_path)
        cache.set("kdisk", {"v": 1}, tokens_estimate=10)
        cache_dir = run_exp_dir(ctx.run_id, tmp_path) / "cache"
        files = list(cache_dir.glob("*.json"))
        assert len(files) == 1
        finalize_run(ctx, status="completed", workspace=tmp_path)

    def test_hit_emits_tokens_saved_event(self, tmp_path):
        from utils.run_context import prepare_new_run, finalize_run
        from utils.run_cache import RunCache
        from utils.workspace import run_exp_dir

        ctx = prepare_new_run(
            task_type="tabular_classification",
            llm_backbone="or-glm-5",
            workspace=tmp_path,
        )
        cache = RunCache(ctx, workspace=tmp_path)
        cache.set("k1", {"x": 1}, tokens_estimate=500)
        # First get is a hit and should record savings.
        _ = cache.get("k1", stage="parse")

        events_file = run_exp_dir(ctx.run_id, tmp_path) / "events.jsonl"
        events = [json.loads(line) for line in events_file.read_text().splitlines() if line]
        saved = [e for e in events if e["event"] == "tokens_saved"]
        assert any(e.get("source") == "cache_hit" for e in saved)
        finalize_run(ctx, status="completed", workspace=tmp_path)
