"""OpenAI-compatible and Gateway JSON chat client."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol
from urllib.parse import urlsplit, urlunsplit

from .env_loader import load_dotenv_once
from .errors import LLMConfigError, LLMError
from .gateway_parser import extract_gateway_response_text, normalize_responses_url
from .logging_util import get_logger

log = get_logger(__name__)

ENV_BACKEND = "DPC_BENCH_LLM_BACKEND"
ENV_BASE_URL = "DPC_BENCH_LLM_BASE_URL"
ENV_API_KEY = "DPC_BENCH_LLM_API_KEY"
ENV_MODEL = "DPC_BENCH_LLM_MODEL"
ENV_TIMEOUT = "DPC_BENCH_LLM_TIMEOUT_S"

# memtensor Chat Completions (api-int.memtensor.cn style)
MEMTENSOR_ENV_URL = "MODEL_gpt_memtensor_URL"
MEMTENSOR_ENV_KEY = "MODEL_gpt_memtensor_KEY"
MEMTENSOR_ENV_MODEL = "GPT_API_MODEL"

GW_ENV_URL = "MODEL_gateway_gpt_URL"
GW_ENV_KEY = "MODEL_gateway_gpt_KEY"
GW_ENV_TASK_TYPE = "MODEL_gateway_gpt_TASK_TYPE"
GW_ENV_CREATED_BY = "MODEL_gateway_gpt_CREATED_BY"
GW_ENV_TASK_LEVEL = "MODEL_gateway_gpt_TASK_LEVEL"
GW_ENV_TOKEN_STAT_MODE = "MODEL_gateway_gpt_TOKEN_STAT_MODE"

# backend aliases → internal transport
_BACKEND_ALIASES = {
    "memtensor": "openai_compat",
    "openai": "openai_compat",
    "openai_compat": "openai_compat",
    "gateway": "gateway",
}


def _normalize_backend(raw: str) -> str:
    key = (raw or "").strip().lower()
    if not key:
        return ""
    if key not in _BACKEND_ALIASES:
        raise LLMConfigError(
            f"unknown LLM backend={raw!r} (expected memtensor|openai_compat|gateway)"
        )
    return _BACKEND_ALIASES[key]


def _env_get(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True)
class LLMConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 120.0
    backend: str = "openai_compat"
    # Some memtensor / vendor endpoints reject response_format=json_object.
    use_json_response_format: bool = False
    gateway_task_type: str = "dpc_bench"
    gateway_created_by: str = "dpc_bench"
    gateway_task_level: str = "demo"
    gateway_token_stat_mode: str = "b"

    @classmethod
    def from_env(cls, name: str = "default") -> "LLMConfig":
        load_dotenv_once()
        backend_raw = os.environ.get(ENV_BACKEND, "").strip().lower()

        base_url = _env_get(ENV_BASE_URL)
        api_key = _env_get(ENV_API_KEY)
        model = _env_get(ENV_MODEL, MEMTENSOR_ENV_MODEL)
        use_json_rf = False

        # Prefer explicit memtensor Chat Completions envs when selected or present.
        mem_url = _env_get(MEMTENSOR_ENV_URL)
        mem_key = _env_get(MEMTENSOR_ENV_KEY)
        if backend_raw in ("", "memtensor", "openai_compat") and mem_url and mem_key:
            backend = "openai_compat"
            base_url = base_url or mem_url
            api_key = api_key or mem_key
            model = model or _env_get(MEMTENSOR_ENV_MODEL) or "gpt-4o-2024-11-20"
            use_json_rf = False
            if not name or name == "default":
                name = "gpt-memtensor"
        elif backend_raw == "gateway" or (not base_url and _env_get(GW_ENV_URL)):
            backend = "gateway"
            base_url = base_url or _env_get(GW_ENV_URL)
            api_key = api_key or _env_get(GW_ENV_KEY)
            model = model or _env_get("DECISION_RGU_LLM_MODEL")
            use_json_rf = False
        else:
            backend = _normalize_backend(backend_raw or "openai_compat")
            use_json_rf = backend_raw not in ("memtensor", "")

        timeout_raw = os.environ.get(ENV_TIMEOUT, "").strip()
        missing = [
            field
            for field, value in (
                ("base_url", base_url),
                ("api_key", api_key),
                ("model", model),
            )
            if not value
        ]
        if missing:
            raise LLMConfigError(
                "missing LLM config fields: "
                + ", ".join(missing)
                + f" (set {MEMTENSOR_ENV_URL}/{MEMTENSOR_ENV_KEY}/{MEMTENSOR_ENV_MODEL} "
                "or DPC_BENCH_LLM_* / gateway envs)"
            )
        try:
            timeout_s = float(timeout_raw) if timeout_raw else 120.0
        except ValueError as exc:
            raise LLMConfigError(f"{ENV_TIMEOUT} is not a number: {timeout_raw!r}") from exc
        return cls(
            name=name,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            backend=backend,
            use_json_response_format=use_json_rf,
            gateway_task_type=os.environ.get(GW_ENV_TASK_TYPE, "dpc_bench").strip() or "dpc_bench",
            gateway_created_by=os.environ.get(GW_ENV_CREATED_BY, "dpc_bench").strip() or "dpc_bench",
            gateway_task_level=os.environ.get(GW_ENV_TASK_LEVEL, "demo").strip() or "demo",
            gateway_token_stat_mode=os.environ.get(GW_ENV_TOKEN_STAT_MODE, "b").strip() or "b",
        )

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any]) -> "LLMConfig":
        load_dotenv_once()
        name = str(spec.get("name") or "").strip()
        if not name:
            raise LLMConfigError("model spec missing name")

        backend_raw = str(spec.get("backend") or "openai_compat").strip().lower()
        backend = _normalize_backend(backend_raw)

        base_url = str(spec.get("base_url") or "").strip()
        base_url_env = str(spec.get("base_url_env") or "").strip()
        if not base_url and base_url_env:
            base_url = os.environ.get(base_url_env, "").strip()

        model = str(spec.get("model") or "").strip()
        model_env = str(spec.get("model_env") or "").strip()
        if not model and model_env:
            model = os.environ.get(model_env, "").strip()

        api_key = str(spec.get("api_key") or "").strip()
        api_key_env = str(spec.get("api_key_env") or "").strip()
        if not api_key and api_key_env:
            api_key = os.environ.get(api_key_env, "").strip()

        if backend == "gateway":
            base_url = base_url or _env_get(GW_ENV_URL)
            api_key = api_key or _env_get(GW_ENV_KEY)
        if backend_raw == "memtensor" or (
            backend == "openai_compat" and not base_url and _env_get(MEMTENSOR_ENV_URL)
        ):
            base_url = base_url or _env_get(MEMTENSOR_ENV_URL)
            api_key = api_key or _env_get(MEMTENSOR_ENV_KEY)
            model = model or _env_get(MEMTENSOR_ENV_MODEL) or "gpt-4o-2024-11-20"

        if "use_json_response_format" in spec:
            use_json_rf = bool(spec.get("use_json_response_format"))
        else:
            use_json_rf = backend_raw not in ("memtensor", "gateway")

        timeout_raw = spec.get("timeout_s", 120.0)
        try:
            timeout_s = float(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise LLMConfigError(f"invalid timeout_s for model {name!r}") from exc

        missing = [
            field
            for field, value in (
                ("base_url", base_url),
                ("api_key", api_key),
                ("model", model),
            )
            if not value
        ]
        if missing:
            hint = ""
            if "api_key" in missing and api_key_env:
                hint = f" (check api_key_env={api_key_env!r})"
            elif "base_url" in missing and base_url_env:
                hint = f" (check base_url_env={base_url_env!r})"
            raise LLMConfigError(f"model {name!r} missing fields: {', '.join(missing)}{hint}")

        return cls(
            name=name,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            backend=backend,
            use_json_response_format=use_json_rf,
            gateway_task_type=str(spec.get("gateway_task_type") or os.environ.get(GW_ENV_TASK_TYPE, "dpc_bench")).strip()
            or "dpc_bench",
            gateway_created_by=str(
                spec.get("gateway_created_by") or os.environ.get(GW_ENV_CREATED_BY, "dpc_bench")
            ).strip()
            or "dpc_bench",
            gateway_task_level=str(spec.get("gateway_task_level") or os.environ.get(GW_ENV_TASK_LEVEL, "demo")).strip()
            or "demo",
            gateway_token_stat_mode=str(
                spec.get("gateway_token_stat_mode") or os.environ.get(GW_ENV_TOKEN_STAT_MODE, "b")
            ).strip()
            or "b",
        )


def _extract_first_json_object(text: str) -> str:
    if not text:
        return ""
    start = text.find("{")
    if start < 0:
        return _extract_last_json_object(text)
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    break
    return _extract_last_json_object(text)


def _extract_last_json_object(text: str) -> str:
    if not text:
        return ""
    fenced = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    for candidate in reversed(fenced):
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue
    indices = [i for i, ch in enumerate(text) if ch == "{"]
    for start in reversed(indices):
        snippet = text[start:].rstrip()
        for end in range(len(snippet), 0, -1):
            piece = snippet[:end]
            if not piece.endswith("}"):
                continue
            try:
                json.loads(piece)
                return piece
            except Exception:
                continue
    return ""


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if not raw:
        raise LLMError("empty LLM content")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    recovered = _extract_first_json_object(raw)
    if not recovered:
        raise LLMError(f"could not extract JSON object from: {raw[:200]!r}")
    parsed = json.loads(recovered)
    if not isinstance(parsed, dict):
        raise LLMError("LLM response is not a JSON object")
    return parsed


def _resolve_chat_completions_url(base_url: str) -> str:
    """Build .../chat/completions URL.

    memtensor api-int expects ``https://host/v1/chat/completions``.
    If the base is only the host root, insert ``/v1`` automatically.
    """
    parts = urlsplit(base_url.strip())
    path = (parts.path or "").rstrip("/")
    if "chat/completion" in path.lower():
        normalized_path = path
    elif path in ("", "/"):
        # https://api-int.memtensor.cn/  →  /v1/chat/completions
        normalized_path = "/v1/chat/completions"
    elif path.endswith("/v1"):
        normalized_path = f"{path}/chat/completions"
    else:
        normalized_path = f"{path}/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, normalized_path, parts.query, parts.fragment))


class ChatClient(Protocol):
    def chat_json(
        self,
        system: str,
        user: str,
        required_keys: List[str],
        max_completion_tokens: int = 1024,
        max_retries: int = 1,
        temperature: float = 0.0,
    ) -> Dict[str, Any]: ...


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.backend = config.backend
        self._url = (
            normalize_responses_url(config.base_url)
            if self.backend == "gateway"
            else _resolve_chat_completions_url(config.base_url)
        )
        import requests

        self._requests = requests
        self.call_count = 0

    def _post(self, *, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        session = self._requests.Session()
        # Bypass system HTTP(S)_PROXY — memtensor/api-int often breaks via corporate proxies.
        session.trust_env = False
        resp = session.post(
            self._url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout_s,
        )
        self.call_count += 1
        body_preview = (resp.text or "")[:300]
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code} url={self._url}: {body_preview!r}")
        ctype = (resp.headers.get("content-type") or "").lower()
        if "application/json" not in ctype and not (resp.text or "").lstrip().startswith(("{", "[")):
            raise LLMError(
                f"non-JSON response from {self._url} "
                f"(status={resp.status_code}, content-type={ctype!r}, body={body_preview!r}). "
                "For memtensor api-int, base URL should be https://api-int.memtensor.cn/v1 "
                "(or bare host; client will append /v1/chat/completions)."
            )
        try:
            data = resp.json()
        except Exception as exc:
            raise LLMError(
                f"failed to decode JSON from {self._url}: {exc}; body={body_preview!r}"
            ) from exc
        if not isinstance(data, dict):
            raise LLMError("LLM response is not a JSON object")
        return data

    def _gateway_headers(self) -> Dict[str, str]:
        cfg = self.config
        return {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            "x-token-stat-mode": cfg.gateway_token_stat_mode,
            "x-task-type": cfg.gateway_task_type,
            "x-created-by": cfg.gateway_created_by,
            "x-task-level": cfg.gateway_task_level,
        }

    def _chat_gateway(
        self,
        system: str,
        user: str,
        *,
        max_completion_tokens: int,
        temperature: float,
    ) -> str:
        instructions = (
            system
            + "\n\nYou must respond with a single valid JSON object only. "
            "Do not wrap in markdown code fences."
        )
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "instructions": instructions,
            "input": [{"type": "message", "role": "user", "content": user}],
            "max_output_tokens": max_completion_tokens,
            "stream": False,
        }
        if temperature != 0.0:
            payload["temperature"] = temperature
        data = self._post(headers=self._gateway_headers(), payload=payload)
        text = extract_gateway_response_text(data)
        recovered = _extract_first_json_object(text)
        return recovered or text

    def _chat_openai_compat(
        self,
        system: str,
        user: str,
        *,
        max_completion_tokens: int,
        temperature: float,
    ) -> str:
        # Align with memtensor Chat Completions demo:
        # messages + max_tokens + stream=false (+ optional response_format).
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_completion_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self.config.use_json_response_format:
            payload["response_format"] = {"type": "json_object"}
            payload["max_completion_tokens"] = max_completion_tokens
        data = self._post(
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or "").strip()
        if not content and reasoning:
            recovered = _extract_last_json_object(reasoning)
            if recovered:
                content = recovered
        return content

    def chat_json(
        self,
        system: str,
        user: str,
        required_keys: List[str],
        max_completion_tokens: int = 1024,
        max_retries: int = 1,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        last_content = ""
        for attempt in range(max_retries + 1):
            try:
                if self.backend == "gateway":
                    content = self._chat_gateway(
                        system,
                        user,
                        max_completion_tokens=max_completion_tokens,
                        temperature=temperature,
                    )
                else:
                    content = self._chat_openai_compat(
                        system,
                        user,
                        max_completion_tokens=max_completion_tokens,
                        temperature=temperature,
                    )
                last_content = content
                parsed = extract_json_object(content)
                missing = [k for k in required_keys if k not in parsed]
                if missing:
                    raise LLMError(f"LLM response missing keys: {missing}")
                return parsed
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
        raise LLMError(
            f"chat_json failed after {max_retries + 1} attempts: {last_error}; "
            f"last_content={last_content[:200]!r}"
        )


class DryRunClient:
    """Deterministic client for offline smoke tests."""

    def __init__(self, responder) -> None:
        self._responder = responder
        self.call_count = 0

    def chat_json(
        self,
        system: str,
        user: str,
        required_keys: List[str],
        max_completion_tokens: int = 1024,
        max_retries: int = 1,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        del system, max_completion_tokens, max_retries, temperature
        self.call_count += 1
        parsed = self._responder(user)
        if not isinstance(parsed, dict):
            raise LLMError("dry-run responder did not return a dict")
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            raise LLMError(f"dry-run response missing keys: {missing}")
        return parsed
