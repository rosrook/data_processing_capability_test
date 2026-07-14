"""Parse Gateway Responses API payloads."""

from __future__ import annotations

from typing import Any, Dict, List

from .errors import LLMError


def normalize_responses_url(url: str) -> str:
    u = url.rstrip("/")
    if u.endswith("/responses"):
        return u
    if u.endswith("/v1"):
        return u + "/responses"
    return u + "/responses"


def unwrap_gateway_envelope(data: Dict[str, Any]) -> Dict[str, Any]:
    if "code" in data and "data" in data:
        code = data.get("code")
        if code not in (0, 200, "0", "200"):
            message = str(data.get("message") or data)
            raise LLMError(f"Gateway envelope error (code={code}): {message}")
        inner = data.get("data")
        if not isinstance(inner, dict):
            raise LLMError("Gateway envelope data is empty or not an object")
        return inner
    return data


def extract_gateway_response_text(data: Dict[str, Any]) -> str:
    data = unwrap_gateway_envelope(data)
    if not isinstance(data, dict):
        raise LLMError(f"Gateway response is not a JSON object: {type(data)!r}")

    err = data.get("error")
    if err is not None:
        if isinstance(err, dict):
            msg = err.get("message") or err.get("code") or str(err)
        else:
            msg = str(err)
        raise LLMError(f"Gateway API error: {msg}")

    top_text = data.get("text")
    if isinstance(top_text, str) and top_text.strip():
        return top_text.strip()

    output = data.get("output")
    if not isinstance(output, list):
        raise LLMError("Gateway response missing output[]")

    chunks: List[str] = []
    skip_types = frozenset({"reasoning", "function_call", "tool_call"})
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).lower()
        if item_type in skip_types:
            continue
        role = str(item.get("role", "")).lower()
        if item_type not in ("message", "") and role != "assistant":
            continue
        if role and role != "assistant" and item_type != "message":
            continue
        chunks.append(_content_to_text(item.get("content")))

    text = "\n".join(c for c in chunks if c).strip()
    if not text:
        raise LLMError("Gateway response did not contain assistant content")
    return text


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block.strip())
            elif isinstance(block, dict):
                btype = str(block.get("type", "")).lower()
                if btype in ("output_text", "text", "input_text", "message"):
                    parts.append(str(block.get("text", "")).strip())
                elif "text" in block:
                    parts.append(str(block["text"]).strip())
        return "\n".join(p for p in parts if p)
    return str(content).strip()
