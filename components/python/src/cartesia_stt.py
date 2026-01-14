"""
Cartesia Speech-to-Text WebSocket client using official Cartesia SDK.
"""

import asyncio
import os
from typing import Optional, AsyncIterator

from cartesia import AsyncCartesia

from events import STTChunkEvent, STTOutputEvent, VoiceAgentEvent


class CartesiaSTT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        model: str = "ink-whisper",
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
    ):
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        if not self.api_key:
            raise ValueError("Cartesia API key is required")

        self.language = language
        self.model = model
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.client: Optional[AsyncCartesia] = None
        self.ws = None
        self.is_connected = False
        self._close_signal = asyncio.Event()

    async def connect(self) -> None:
        print(f"Connecting to Cartesia STT (model={self.model})...")
        self.client = AsyncCartesia(api_key=self.api_key)
        self.ws = await self.client.stt.websocket(
            model=self.model,
            language=self.language,
            encoding=self.encoding,
            sample_rate=self.sample_rate,
        )
        self.is_connected = True
        print("Cartesia STT connected")

    async def send_audio(self, audio_data: bytes) -> None:
        if not self.is_connected or not self.ws:
            return
        try:
            await self.ws.send(audio_data)
        except Exception as e:
            print(f"STT send error: {e}")
            self.is_connected = False

    async def receive_events(self) -> AsyncIterator[VoiceAgentEvent]:
        if not self.is_connected or not self.ws:
            return

        try:
            async for result in self.ws.receive():
                if self._close_signal.is_set():
                    break

                result_type = result.get("type", "")

                if result_type == "transcript":
                    text = result.get("text", "")
                    is_final = result.get("is_final", False)

                    if is_final:
                        print(f"STT final: '{text}'")
                        yield STTOutputEvent.create(text)
                    else:
                        yield STTChunkEvent.create(text)

                elif result_type == "done":
                    break

        except Exception as e:
            print(f"STT receive error: {e}")
            self.is_connected = False

    async def close(self) -> None:
        self._close_signal.set()
        if self.ws:
            try:
                await self.ws.send("done")
                await self.ws.close()
            except Exception:
                pass
        if self.client:
            try:
                await self.client.close()
            except Exception:
                pass
        self.is_connected = False
        self.ws = None
        self.client = None
        print("Cartesia STT closed")
