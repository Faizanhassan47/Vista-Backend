"""Async poll loop that claims and runs background_jobs rows."""

from __future__ import annotations

import asyncio
import logging

from core.config import JOB_QUEUE_BATCH_SIZE, JOB_QUEUE_ENABLED, JOB_QUEUE_POLL_SECONDS
from services.job_queue import claim_next, process_one, register_builtin_handlers, worker_id

logger = logging.getLogger("pathology.job_queue")


async def job_queue_loop(connect_fn, stop_event: asyncio.Event) -> None:
    if not JOB_QUEUE_ENABLED:
        logger.info("job_queue.disabled")
        return

    register_builtin_handlers(connect_fn)
    owner = worker_id()
    logger.info("job_queue.started worker_id=%s poll=%ss", owner, JOB_QUEUE_POLL_SECONDS)

    while not stop_event.is_set():
        worked = False
        try:
            for _ in range(max(1, JOB_QUEUE_BATCH_SIZE)):
                job = await asyncio.to_thread(claim_next, connect_fn, owner=owner)
                if not job:
                    break
                worked = True
                logger.info("job.claimed id=%s type=%s attempt=%s", job.get("id"), job.get("job_type"), job.get("attempts"))
                await asyncio.to_thread(process_one, connect_fn, job)
        except Exception:
            logger.exception("job_queue.tick_failed")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=0.25 if worked else JOB_QUEUE_POLL_SECONDS)
            break
        except asyncio.TimeoutError:
            continue

    logger.info("job_queue.stopped worker_id=%s", owner)
