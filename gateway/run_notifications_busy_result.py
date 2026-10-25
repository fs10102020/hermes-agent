"""Which finished background results may ride a running turn, and the process-completion carry.

``delegation.busy_result_mode`` governs subagent results and
``display.background_process_busy_result_mode`` governs ``notify_on_complete`` process completions.
With ``steer`` a result that finishes while its session's turn runs is appended to that turn's next
tool result (``agent.background_results``) instead of queueing a new turn.

A process completion has no durable claim to hold, so its delivery call waits on the carry instead:
the future resolves True once a tool result carried it and False when the turn ended without one,
and the caller then queues it as the next turn exactly as ``queue`` mode does.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any, Optional

logger = logging.getLogger("gateway.run")

BUSY_RESULT_MODE_KEYS = {
    "async_delegation": ("delegation", "busy_result_mode"),
    "completion": ("display", "background_process_busy_result_mode"),
}


@dataclasses.dataclass(frozen=True)
class CarriedCompletion:
    """A claimed completion riding a running turn (``delegation.busy_result_mode: steer``): what
    settling it later needs once a tool result carried it, or the turn ended without one."""

    evt: dict
    delegation_id: str
    claim_id: str
    identity: Any
    siblings: tuple
    waiter: Optional[asyncio.Future] = None  # a process completion's carry outcome


def busy_result_mode(evt_type: str) -> str:
    """The profile in scope's busy-result mode for ``evt_type``: ``queue`` (default) or ``steer``."""
    from gateway.run import _load_gateway_config
    from hermes_cli.config import cfg_get

    section, key = BUSY_RESULT_MODE_KEYS[evt_type]
    mode = str(cfg_get(_load_gateway_config(), section, key, default="") or "queue").strip().lower()
    if mode not in ("queue", "steer"):
        logger.warning("%s.%s=%r is not 'queue' or 'steer'; using 'queue'", section, key, mode)
        return "queue"
    return mode


def offer_process_completion(agent: Any, carried: Any, synth_text: str) -> asyncio.Future:
    """Offer a process completion to ``agent``'s next tool result; return the carry outcome future."""
    from agent.background_results import PROCESS_RESULT_MARKER, offer, withdraw

    waiter = asyncio.get_running_loop().create_future()
    carried = dataclasses.replace(carried, waiter=waiter)
    session_ids = [str(sid or "") for sid in carried.evt.get("_batch_session_ids") or [carried.evt.get("session_id")]]

    def already_read() -> bool:
        # The agent read every output with process(wait/log) this turn: showing it again is noise.
        import tools.process_registry as process_registry_module
        return all(process_registry_module.process_registry.is_completion_consumed(sid) for sid in session_ids)

    offer(agent, carried, synth_text, marker=PROCESS_RESULT_MARKER, already_read=already_read)
    # A cancelled delivery (shutdown) is retried by its watcher, so the turn must not carry it too.
    waiter.add_done_callback(lambda done: done.cancelled() and withdraw(agent, carried))
    logger.info("Carrying process completion %s into the running turn of %s",
                ", ".join(session_ids), carried.evt.get("session_key"))
    return waiter


class GatewayBusyResultMixin:
    """Settling carried completions: acknowledge what a tool result carried, requeue what it did not.

    Lives here, beside ``CarriedCompletion``, so ``gateway/run_notifications.py`` stays under its
    line cap. Every ``self.`` attribute below is defined on the composed ``GatewayNotificationsMixin``.
    """

    async def _settle_carried_completions(self, agent: Any = None, *, final: bool = False) -> None:
        """Acknowledge completions a tool result carried; at turn end (``final``) hand the rest back
        to the completion queue (claims deferred, no attempt spent) so they arrive as the next turn."""
        try:
            await self._settle_carried_completions_unguarded(agent, final=final)
        except Exception:  # health: allow BLE001 -- runs in a turn's cleanup; the claim lease re-offers anything unsettled
            logger.warning("Could not settle carried subagent completions", exc_info=True)

    async def _settle_carried_completions_unguarded(self, agent: Any, *, final: bool) -> None:
        from agent.background_results import take_delivered, take_outcome
        agents = self.__dict__.get("_carrying_agents") or {}
        if agent is not None and id(agent) not in agents:
            return
        for current in [agent] if agent is not None else list(agents.values()):
            if final:
                agents.pop(id(current), None)
            # Settling an outcome can register a new carry: run_turn settles while the agent slot is
            # still held, so a delivery landing now sees that agent and re-registers. Drain until
            # quiet, bounded so a hot queue still ends.
            for _ in range(64):
                if final:
                    delivered, undelivered = take_outcome(current)
                else:
                    delivered, undelivered = take_delivered(current), []
                if not delivered and not undelivered:
                    break
                for carried in delivered:
                    await self._settle_carried_completion(carried, "complete")
                for carried, _text in undelivered:
                    await self._settle_carried_completion(carried, "defer")

    async def _settle_carried_completion(self, carried: CarriedCompletion, operation: str) -> None:
        from tools.process_registry import process_registry as _pr
        if carried.waiter is not None:  # a process completion: its waiting delivery call settles it
            if not carried.waiter.done():
                carried.waiter.set_result(operation == "complete")
            return
        async with self._completion_event_scope(carried.evt):
            if carried.claim_id:
                self._settle_durable_claim(operation, carried.delegation_id, carried.claim_id)
            for sibling, claim_id in carried.siblings:
                if claim_id:
                    self._settle_durable_claim(operation, sibling["delegation_id"], claim_id)
        if operation == "complete":
            if carried.identity is not None:
                with self._completion_delivery_lock:
                    self._mark_completions_delivered_locked((carried.identity,))
            if carried.siblings:
                self._record_coalesced_completion_siblings([event for event, _claim_id in carried.siblings])
            return
        if carried.identity is not None:
            with self._completion_delivery_lock:
                self._completion_deliveries_inflight.discard(carried.identity)
        evt = {k: v for k, v in carried.evt.items() if k != "completion_ids"}
        for event in (evt, *(sibling for sibling, _claim_id in carried.siblings)):
            _pr.completion_queue.put(event)
