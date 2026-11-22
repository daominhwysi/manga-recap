"""Unified OpenAI-compatible LLM/VLM client with structured Pydantic parsing."""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from src.config import LLMConfig
from src.core.telemetry import TelemetryTracker

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Calls OpenAI-compatible chat/vision endpoints with client reuse, circuit breaker, and telemetry."""

    def __init__(
        self,
        config: LLMConfig,
        telemetry: TelemetryTracker | None = None,
    ) -> None:
        self.config = config
        self.telemetry = telemetry
        self._lock = threading.Lock()
        self._text_client: OpenAI | None = None
        self._vision_client: OpenAI | None = None
        self._tripped_models: set[str] = set()

    @property
    def has_text_credentials(self) -> bool:
        return bool(self.config.api_key)

    @property
    def has_vision_credentials(self) -> bool:
        return bool(self.config.vision_api_key or self.config.api_key)

    def _make_client(self, vision: bool = False) -> OpenAI:
        with self._lock:
            cached = self._vision_client if vision else self._text_client
            if cached is not None:
                return cached
            api_key = (
                (self.config.vision_api_key or self.config.api_key)
                if vision
                else self.config.api_key
            )
            base_url = (
                (self.config.vision_base_url or self.config.base_url)
                if vision
                else self.config.base_url
            )
            if not api_key:
                raise RuntimeError(
                    "Missing API key (OPENAI_API_KEY / VISION_API_KEY) and offline_fallback is disabled."
                )
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": self.config.timeout_sec,
                "max_retries": 0,  # Avoid SDK retries multiplying LLMClient retries on proxy timeouts
            }
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)
            if vision:
                self._vision_client = client
            else:
                self._text_client = client
            return client

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        ext = image_path.suffix.lower().lstrip(".")
        mime = "image/png" if ext == "png" else "image/jpeg"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
        if fence:
            cleaned = fence.group(1)
        return json.loads(cleaned)

    def complete_json(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        schema_cls: type[T],
        image_paths: list[Path] | None = None,
        model_override: str | None = None,
    ) -> T:
        """Invoke text or vision model and validate response against schema_cls."""
        is_vision = bool(image_paths)
        model = model_override or (
            self.config.vision_model if is_vision else self.config.default_model
        )
        active_provider = self.config.vision_provider if is_vision else self.config.provider
        client = self._make_client(vision=is_vision)

        user_content: Any
        if image_paths:
            parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for p in image_paths:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(p)},
                    }
                )
            user_content = parts
        else:
            user_content = user_prompt

        schema_hint = json.dumps(
            schema_cls.model_json_schema(), ensure_ascii=False, separators=(",", ":")
        )
        full_system_prompt = (
            f"{system_prompt}\n\n"
            "BẮT BUỘC trả về JSON hợp lệ khớp chính xác JSON Schema sau (chỉ trả về JSON, không thêm giải thích):\n"
            f"{schema_hint}"
        )

        candidates = [model]
        if not is_vision and self.config.fallback_model and self.config.fallback_model != model:
            candidates.append(self.config.fallback_model)

        with self._lock:
            models_to_try = [m for m in candidates if m not in self._tripped_models]
        if not models_to_try:
            models_to_try = candidates[-1:]

        last_err: Exception | None = None
        for active_model in models_to_try:
            model_failed_all = True
            for attempt in range(self.config.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    create_kwargs: dict[str, Any] = {
                        "model": active_model,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                        "messages": [
                            {"role": "system", "content": full_system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                    }
                    if active_provider:
                        create_kwargs["extra_body"] = {"provider": active_provider}
                    resp = client.chat.completions.create(**create_kwargs)
                    dt = time.perf_counter() - t0
                    choice = resp.choices[0]
                    finish_reason = getattr(choice, "finish_reason", None)
                    content = choice.message.content or "{}"
                    usage = getattr(resp, "usage", None)
                    if self.telemetry:
                        if active_model != model:
                            self.telemetry.warn(
                                f"{agent_name}: '{model}' skipped/failed, used '{active_model}'."
                            )
                        self.telemetry.record_llm_call(
                            agent=agent_name,
                            model=active_model,
                            duration_sec=dt,
                            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                            mode="live",
                        )
                    if finish_reason == "length":
                        msg = (
                            f"{agent_name}: output truncated (finish_reason='length') "
                            f"at max_tokens={self.config.max_tokens}."
                        )
                        if self.telemetry:
                            self.telemetry.warn(msg)
                        raise RuntimeError(msg)
                    data = self._extract_json(content)
                    model_failed_all = False
                    return schema_cls.model_validate(data)
                except Exception as exc:
                    last_err = exc
                    if attempt < self.config.max_retries:
                        time.sleep(0.5 * (attempt + 1))

            # Trip circuit breaker if primary model failed all retries and a fallback exists
            if model_failed_all and active_model == model and len(candidates) > 1:
                with self._lock:
                    self._tripped_models.add(active_model)

        raise RuntimeError(f"LLM call failed for {agent_name} ({models_to_try}): {last_err}")
