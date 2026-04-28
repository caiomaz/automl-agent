"""BranchScheduler — Phase 3.

Dispatches per-branch jobs in parallel with a configurable cap and falls back
to serial execution when:

1. ``mode='serial'`` is requested,
2. ``max_concurrency <= 1``,
3. a worker raises :class:`SchedulerFallback` (typically used to signal
   "provider is throttling, retry serially with a slower path").

Events are written to the run ledger so the trace shows what mode actually
ran (``scheduler_started``, ``scheduler_fallback_serial``,
``scheduler_completed``).

Design notes
------------
- We use a thread pool, not a process pool, so worker functions can be
  arbitrary callables (closures) and so cost/event records appended to
  the per-run ledger from inside a job are immediately visible without
  cross-process plumbing.
- Workers should perform their own I/O via :mod:`utils.ledger`; the
  scheduler only orchestrates ordering and fallback.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence


_VALID_MODES = frozenset({"parallel", "serial"})


class SchedulerFallback(RuntimeError):
    """Worker-side signal: abort parallel execution and retry serially."""


class BranchScheduler:
    """Run a list of jobs with controlled concurrency and serial fallback."""

    def __init__(
        self,
        *,
        mode: str = "parallel",
        max_concurrency: int = 4,
        run_ctx: Any | None = None,
        workspace: Any | None = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid scheduler mode {mode!r}. Valid: {sorted(_VALID_MODES)}"
            )
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.mode = mode
        self.max_concurrency = max_concurrency
        self.run_ctx = run_ctx
        self.workspace = workspace

    # ── Public API ───────────────────────────────────────────────────────────

    def map(
        self,
        worker: Callable[[Any], Any],
        jobs: Sequence[Any],
        *,
        serial_fn: Callable[[Any], Any] | None = None,
    ) -> list[Any]:
        """Run ``worker`` over ``jobs`` and return results in input order.

        Parameters
        ----------
        worker:
            Callable executed for each job in parallel.
        jobs:
            Sequence of opaque payloads (any picklable / call-safe object).
        serial_fn:
            Optional callable used during the fallback pass. If not given,
            the original ``worker`` is reused.
        """
        jobs_list = list(jobs)
        self._emit("scheduler_started", mode=self.mode, jobs=len(jobs_list),
                   max_concurrency=self.max_concurrency)

        effective_mode = self.mode
        if self.max_concurrency == 1:
            effective_mode = "serial"

        if effective_mode == "serial":
            results = self._run_serial(worker, jobs_list)
            self._emit("scheduler_completed", mode="serial", jobs=len(jobs_list))
            return results

        try:
            results = self._run_parallel(worker, jobs_list)
            self._emit("scheduler_completed", mode="parallel", jobs=len(jobs_list))
            return results
        except SchedulerFallback as exc:
            self._emit("scheduler_fallback_serial", reason=str(exc) or "fallback")
            fn = serial_fn or worker
            results = self._run_serial(fn, jobs_list)
            self._emit("scheduler_completed", mode="serial", jobs=len(jobs_list),
                       fallback=True)
            return results

    # ── Internals ────────────────────────────────────────────────────────────

    def _run_serial(self, worker, jobs):
        return [worker(job) for job in jobs]

    def _run_parallel(self, worker, jobs):
        results: list[Any] = [None] * len(jobs)
        first_fallback: SchedulerFallback | None = None
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {pool.submit(worker, job): idx for idx, job in enumerate(jobs)}
            for fut, idx in futures.items():
                try:
                    results[idx] = fut.result()
                except SchedulerFallback as exc:
                    if first_fallback is None:
                        first_fallback = exc
        if first_fallback is not None:
            raise first_fallback
        return results

    def _emit(self, event: str, **fields: Any) -> None:
        if self.run_ctx is None:
            return
        try:
            from utils.ledger import append_event
            append_event(self.run_ctx, event, source="scheduler",
                         workspace=self.workspace, **fields)
        except Exception:
            pass
