"""
TTS pod registry for managing lifecycle in Valkey.

Mirrors the renderer registration pattern from model-benchmarking so the
qarl-backend-api allocator can hand out TTS pods one-per-job.
"""
import socket
from logging import getLogger
from pathlib import Path
from typing import Optional

import redis.asyncio as redis

logger = getLogger(__name__)

BUSY_FLAG = Path("/tmp/busy")


class TTSRegistry:
    """Manages TTS pod registration and lifecycle in Valkey."""

    def __init__(self, valkey_url: str, tts_id: Optional[str] = None):
        self.valkey_url = valkey_url
        self.tts_id = tts_id or socket.gethostname()
        self.redis_client: Optional[redis.Redis] = None

    async def connect(self):
        self.redis_client = await redis.from_url(self.valkey_url)
        logger.info(f"Connected to Valkey: {self.valkey_url}")

    async def cleanup_previous_registration(self):
        """Remove stale entry from a previous pod incarnation with the same name."""
        if not self.redis_client:
            raise RuntimeError("Must call connect() before cleanup_previous_registration()")

        await self.redis_client.srem("tts:idle", self.tts_id)
        await self.redis_client.srem("tts:all", self.tts_id)
        await self.redis_client.delete(f"tts:{self.tts_id}")
        logger.info(f"Cleaned stale registration for {self.tts_id}")

    async def register(self, ws_url: str):
        if not self.redis_client:
            raise RuntimeError("Must call connect() before register()")

        await self.redis_client.hset(
            f"tts:{self.tts_id}",
            mapping={"url": ws_url, "status": "idle"},
        )
        await self.redis_client.sadd("tts:idle", self.tts_id)
        await self.redis_client.sadd("tts:all", self.tts_id)
        logger.info(f"Registered TTS pod {self.tts_id} at {ws_url}")

    async def initialize(self, ws_url: str):
        await self.connect()
        await self.cleanup_previous_registration()
        await self.register(ws_url)

    async def mark_busy(self):
        # WS-lifecycle only — the preStop hook polls /tmp/busy to gate graceful
        # shutdown. Allocation state in Valkey is owned by qarl-backend-api via
        # the `tts:lease:{tts_id}` key; the pod no longer mutates tts:idle.
        BUSY_FLAG.touch()
        logger.info(f"TTS pod {self.tts_id} touched busy flag")

    async def mark_idle(self):
        BUSY_FLAG.unlink(missing_ok=True)
        logger.info(f"TTS pod {self.tts_id} cleared busy flag")

    async def unregister(self):
        if not self.redis_client:
            logger.warning("Cannot unregister: not connected to Valkey")
            return
        logger.info(f"Unregistering TTS pod {self.tts_id}")
        await self.redis_client.srem("tts:idle", self.tts_id)
        await self.redis_client.srem("tts:all", self.tts_id)
        await self.redis_client.delete(f"tts:{self.tts_id}")
        BUSY_FLAG.unlink(missing_ok=True)
        await self.redis_client.close()
