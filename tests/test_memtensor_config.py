import os

from dpc_bench.llm_client import LLMConfig, _resolve_chat_completions_url


def test_resolve_chat_completions_url_appends_path():
    assert (
        _resolve_chat_completions_url("https://api-int.memtensor.cn/")
        == "https://api-int.memtensor.cn/v1/chat/completions"
    )
    assert (
        _resolve_chat_completions_url("https://api-int.memtensor.cn/v1")
        == "https://api-int.memtensor.cn/v1/chat/completions"
    )
    assert _resolve_chat_completions_url(
        "https://api-int.memtensor.cn/v1/chat/completions"
    ).endswith("/v1/chat/completions")


def test_from_env_memtensor(monkeypatch):
    monkeypatch.setenv("DPC_BENCH_LLM_BACKEND", "memtensor")
    monkeypatch.setenv("MODEL_gpt_memtensor_URL", "https://api-int.memtensor.cn/")
    monkeypatch.setenv("MODEL_gpt_memtensor_KEY", "sk-test")
    monkeypatch.setenv("GPT_API_MODEL", "gpt-4o-2024-11-20")
    monkeypatch.delenv("DPC_BENCH_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("DPC_BENCH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DPC_BENCH_LLM_MODEL", raising=False)

    # Bypass load_dotenv once state by setting env after import path
    import dpc_bench.env_loader as el

    el._LOAD_TRIED = True
    el._LOADED_PATH = None

    cfg = LLMConfig.from_env(name="gpt-memtensor")
    assert cfg.backend == "openai_compat"
    assert cfg.base_url == "https://api-int.memtensor.cn/"
    assert cfg.api_key == "sk-test"
    assert cfg.model == "gpt-4o-2024-11-20"
    assert cfg.use_json_response_format is False


def test_from_mapping_memtensor(monkeypatch):
    monkeypatch.setenv("MODEL_gpt_memtensor_URL", "https://api-int.memtensor.cn/")
    monkeypatch.setenv("MODEL_gpt_memtensor_KEY", "sk-test")
    monkeypatch.setenv("GPT_API_MODEL", "gpt-4o-mini")
    import dpc_bench.env_loader as el

    el._LOAD_TRIED = True

    cfg = LLMConfig.from_mapping(
        {
            "name": "gpt-memtensor",
            "backend": "memtensor",
            "base_url_env": "MODEL_gpt_memtensor_URL",
            "api_key_env": "MODEL_gpt_memtensor_KEY",
            "model_env": "GPT_API_MODEL",
        }
    )
    assert cfg.backend == "openai_compat"
    assert cfg.model == "gpt-4o-mini"
    assert not cfg.use_json_response_format
