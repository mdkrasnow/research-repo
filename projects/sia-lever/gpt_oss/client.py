"""Minimal provider-agnostic chat client for gpt-oss-120b.

Reads endpoint/model/key from environment so no provider is hardcoded:
  GPT_OSS_BASE_URL   e.g. https://api.tokenfactory.us-central1.nebius.com/v1/  (Nebius Token Factory)
  GPT_OSS_API_KEY    falls back to NEBIUS_API_KEY, then OPENAI_API_KEY
  GPT_OSS_MODEL      default "gpt-oss-120b" (Nebius id e.g. "openai/gpt-oss-120b")

Uses the `openai` SDK against any OpenAI-compatible endpoint (Nebius, vLLM, local).
"""

import os


def resolve_config(model=None, base_url=None, api_key=None):
    model = model or os.getenv("GPT_OSS_MODEL", "gpt-oss-120b")
    base_url = base_url or os.getenv("GPT_OSS_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = (api_key or os.getenv("GPT_OSS_API_KEY") or os.getenv("NEBIUS_API_KEY")
               or os.getenv("OPENAI_API_KEY"))
    return {"model": model, "base_url": base_url, "api_key": api_key}


def make_client(base_url=None, api_key=None):
    from openai import OpenAI
    cfg = resolve_config(base_url=base_url, api_key=api_key)
    if not cfg["api_key"]:
        raise RuntimeError("No API key. Set GPT_OSS_API_KEY or NEBIUS_API_KEY or OPENAI_API_KEY.")
    kwargs = {"api_key": cfg["api_key"]}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


def chat(messages, model=None, base_url=None, api_key=None, temperature=0.0, max_tokens=512,
         client=None):
    """One chat completion. Returns the assistant text. Raises on transport error."""
    cfg = resolve_config(model=model, base_url=base_url, api_key=api_key)
    client = client or make_client(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=cfg["model"], messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content
