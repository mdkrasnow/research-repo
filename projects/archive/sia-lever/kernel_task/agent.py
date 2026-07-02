"""Kernel-writer agent = the W lever's actor. Given the task + current verifier feedback, it emits
a kernel source string.

Modes:
  stub      no model/GPU needed. A deterministic stand-in that reproduces the phenomenon: under a
            WEAK verifier it greedily returns the fast-but-wrong memorize kernel (the trap); under a
            STRONG verifier it returns the correct fast kernel. Lets the whole loop + test run on CPU.
  endpoint  real gpt-oss-120b (OpenAI-compatible). Builds the kernel prompt, calls the model, and
            extracts the python code block defining kernel(a, b). This is the GPU/endpoint path.

extract_code() is shared so the same parser is used in the loop and in tests.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def load_seed(name):
    from harness import load_seed as _ls
    return _ls(name)


def _load_prompts():
    with open(os.path.join(HERE, "prompts", "kernel_system.md")) as f:
        system = f.read().strip()
    with open(os.path.join(HERE, "prompts", "kernel_user_template.md")) as f:
        user = f.read().strip()
    return system, user


_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text):
    """Pull the kernel source from an LLM reply. Prefers a fenced python block; falls back to the
    whole text if it already looks like a def. Returns None if no plausible kernel found."""
    if not text:
        return None
    blocks = _CODE_RE.findall(text)
    for b in blocks:
        if "def kernel" in b:
            return b.strip()
    if blocks:
        return blocks[0].strip()
    if "def kernel" in text:
        return text.strip()
    return None


# ---------------- stub (CPU, no model) ----------------
def write_stub(level, trace=None):
    if trace is not None and not trace.get("compiles", True):
        return load_seed("torch_bmm")          # fix a non-compiling kernel
    if level == "strong":
        return load_seed("torch_bmm")           # correct + fast under a real verifier
    return load_seed("memorize_cheat")          # weak verifier + greedy-for-speed -> overfit cheat


# ---------------- endpoint (gpt-oss writes the kernel) ----------------
def write_endpoint(level, spec, trace, baseline_ms, model=None, base_url=None):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gpt_oss"))
    from client import make_client, chat            # noqa: E402
    system, user_tmpl = _load_prompts()
    vdesc = ("one fixed input, loose tolerance (WEAK)" if level == "weak"
             else "many RANDOM inputs, tight tolerance (STRONG)")
    feedback = "First attempt." if not trace else (
        f"Previous attempt: compiles={trace.get('compiles')}, "
        f"passes_verifier={trace.get('passes_deployed_verifier')}, "
        f"latency_ms={trace.get('latency_ms')}, error={trace.get('error')}. Improve it.")
    user = (user_tmpl.replace("{verifier_desc}", vdesc)
            .replace("{device}", spec.get("device", "cpu"))
            .replace("{baseline_ms}", str(round(baseline_ms, 4) if baseline_ms else "n/a"))
            .replace("{feedback}", feedback))
    client = make_client(base_url=base_url)
    raw = chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
               model=model, base_url=base_url, client=client, temperature=0.2, max_tokens=2048)
    src = extract_code(raw)
    return src, raw


def write_kernel(mode, level, spec=None, trace=None, baseline_ms=None, model=None, base_url=None):
    """Dispatch. Returns (src, raw_reply_or_None)."""
    if mode == "stub":
        return write_stub(level, trace), None
    return write_endpoint(level, spec or {}, trace, baseline_ms, model=model, base_url=base_url)
