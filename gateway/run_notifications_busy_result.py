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
from typing import Any

logger = logging.getLogger("gateway.run")

BUSY_RESULT_MODE_KEYS = {
    "async_delegation": ("delegation", "busy_result_mode"),
    "completion": ("display", "background_process_busy_result_mode"),
}


@dataclasses.dataclass(frozen=True)
class _CarriedProcessCompletion:
    """A claim-less process completion riding a running turn.

    ``display.background_process_busy_result_mode: steer`` hands it to the session's running agent,
    which appends it to its next tool result. Unlike ``_CarriedCompletion`` it owns no durable row:
    the delivery call waiting on ``waiter`` is what settles, and the per-process watcher owns the
    retry, so there is nothing to acknowledge and nothing to requeue.
    """

    evt: dict
    waiter: asyncio.Future


def offer_process_completion(agent: Any, evt: dict, synth_text: str) -> asyncio.Future:
    """Offer a process completion to ``agent``'s next tool result; return the carry outcome future."""
    from agent.background_results import PROCESS_RESULT_MARKER, offer, withdraw

    waiter = asyncio.get_running_loop().create_future()
    carried = _CarriedProcessCompletion(evt, waiter)
    session_ids = [str(sid or "") for sid in evt.get("_batch_session_ids") or [evt.get("session_id")]]

    def already_read() -> bool:
        # The agent read every output with process(wait/log) this turn: showing it again is noise.
        import tools.process_registry as process_registry_module
        return all(process_registry_module.process_registry.is_completion_consumed(sid) for sid in session_ids)

    offer(agent, carried, synth_text, marker=PROCESS_RESULT_MARKER, already_read=already_read)
    # A cancelled delivery (shutdown) is retried by its watcher, so the turn must not carry it too.
    waiter.add_done_callback(lambda done: done.cancelled() and withdraw(agent, carried))
    logger.info("Carrying process completion %s into the running turn of %s",
                ", ".join(session_ids), evt.get("session_key"))
    return waiter


class GatewayBusyResultMixin:
    """The busy-result carry hooks ``GatewayNotificationsMixin`` inherits.

    Kept beside the mode table and the process offer so the delivery module stays under its line
    cap. Every method is duck-typed against the runner (``_peek_session_state``,
    ``_carrying_agents``) and imports nothing back from ``gateway.run_notifications``.
    """

    def _busy_result_mode(self, evt_type: str) -> str:
        """The profile in scope's busy-result mode for ``evt_type``: ``queue`` (default) or ``steer``."""
        from gateway.run import _load_gateway_config
        from hermes_cli.config import cfg_get
        section, key = BUSY_RESULT_MODE_KEYS[evt_type]
        mode = str(cfg_get(_load_gateway_config(), section, key, default="") or "queue").strip().lower()
        if mode not in ("queue", "steer"):
            logger.warning("%s.%s=%r is not 'queue' or 'steer'; using 'queue'", section, key, mode)
            return "queue"
        return mode

    def _busy_result_agent(self, evt: dict):
        """The session's running agent when it can take a carried result, else ``None``."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        state = self._peek_session_state(str(evt.get("session_key") or ""))
        agent = state.turn.agent if state is not None else None
        if agent is None or agent is _AGENT_PENDING_SENTINEL or not callable(getattr(agent, "steer", None)):
            return None
        return agent

    async def _carry_process_completion_into_running_turn(self, synth_text: str, evt: dict) -> bool:
        """``display.background_process_busy_result_mode: steer``: offer a claim-less process
        completion to the session's running agent instead of queueing a new turn. Returns True once a
        tool result carried it; False leaves the caller on the ordinary injection path, so watcher
        retries and fallback notices are unchanged.

        This lives at the funnel both delivery paths reach (the watch drain and the completion
        flush), so a future notice type only has to widen the guard rather than add a second hook.
        """
        if evt.get("type") != "completion" or evt.get("task_failure_notice") or evt.get("origin_session_id"):
            return False
        agent = self._busy_result_agent(evt)
        if agent is None or self._busy_result_mode("completion") != "steer":
            return False
        self.__dict__.setdefault("_carrying_agents", {})[id(agent)] = agent
        return bool(await offer_process_completion(agent, evt, synth_text))

    # A final settle can be re-entered by a delivery that offers while it awaits a settle; bound the
    # re-drain by items so a steady stream of completions cannot hold a turn's cleanup open.
    _FINAL_SETTLE_BUDGET = 64

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
            if not final:
                for carried in take_delivered(current):
                    await self._settle_carried_completion(carried, "complete")
                continue
            agents.pop(id(current), None)
            # The turn's agent stays in its slot until run_turn releases it, so a delivery can offer
            # while this pass awaits a settle and land after the drain that would have seen it. Keep
            # draining until a pass finds nothing, so a straggler settles here instead of parking a
            # waiter nothing will ever resolve. The item budget bounds a pathological producer.
            settled = 0
            while settled < self._FINAL_SETTLE_BUDGET:
                delivered, undelivered = take_outcome(current)
                if not delivered and not undelivered:
                    agents.pop(id(current), None)
                    break
                settled += len(delivered) + len(undelivered)
                for carried in delivered:
                    await self._settle_carried_completion(carried, "complete")
                for carried, _text in undelivered:
                    await self._settle_carried_completion(carried, "defer")
            else:
                # Over budget: settle the tail once more, then keep the agent registered so the next
                # turn's settle still reaches anything that arrives after this pass.
                delivered, undelivered = take_outcome(current)
                for carried in delivered:
                    await self._settle_carried_completion(carried, "complete")
                for carried, _text in undelivered:
                    await self._settle_carried_completion(carried, "defer")
                agents[id(current)] = current
                logger.warning("Carried completions kept arriving during turn cleanup; deferring the rest")

    async def _settle_carried_completion(self, carried: Any, operation: str) -> None:
        if isinstance(carried, _CarriedProcessCompletion):
            # Claim-less: the delivery call waiting on it settles, and its watcher owns the retry.
            if not carried.waiter.done():
                carried.waiter.set_result(operation == "complete")
            return
        from tools.process_registry import process_registry as _pr
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
