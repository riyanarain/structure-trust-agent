"""The agent loop and the three run modes.

Modes (combine with an evidence condition: image, text, or both):
  agent    the model decides which tools to call, up to cfg.max_steps
  fixed    every tool is run on every protein before a single model call (the "fixed pipeline" baseline)
  notools  one model call, no tools (the "VLM alone" baseline)

The protocol is plain JSON so it works with any chat model. Each assistant turn is one JSON object:
  {"action": "call_tool", "tool": "<name>", "args": {...}}
  {"action": "final", "segments": [...], "domain_pairs": [...], "notes": "..."}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .evidence import domain_header, text_evidence
from .llm import LLMClient, Message
from .render import render_all
from .tools import ToolContext, run_tool, tool_docs

TRUST_WORDS = {"high": 0.9, "medium": 0.5, "low": 0.1}

TASK = """\
You are judging how far a predicted protein structure (from AlphaFold) can be trusted, without access to the true structure.
Two things are asked:
1. segments: for residue ranges covering the whole protein, a trust score from 0 to 1, where 1 means the local structure is
   expected to match the experimental structure closely and 0 means it is expected to be wrong. Split the protein wherever
   the evidence changes (domains, linkers, loops, termini) and use segments of at most about 50 residues. Do not give one
   score for the whole protein; the scores should differ between regions that look more and less reliable.
2. domain_pairs: for each listed domain pair, whether the relative placement of the two domains is reliable.
   A domain can be well folded and still be placed wrongly relative to another domain.
Final answer format (one JSON object, no other text):
{"action": "final",
 "segments": [{"start": 1, "end": 85, "trust": 0.92}, ...],
 "domain_pairs": [{"a": "D1", "b": "D2", "placement_reliable": true, "confidence": 0.8}, ...],
 "notes": "one or two sentences on the main evidence"}
"""

TOOL_PROTOCOL = """\
You may call tools before answering. To call one, reply with exactly one JSON object:
{"action": "call_tool", "tool": "<name>", "args": {...}}
You will get the result in the next message. Call at most one tool per turn. Stop and give the final answer once you have enough evidence.
Available tools:
"""


def system_prompt(use_tools: bool, tools=None) -> str:
    docs = tool_docs(tools if tools is not None else ["foldseek_support", "esm2_plausibility"])
    return TASK + ("\n" + TOOL_PROTOCOL + docs if use_tools else "\nNo tools are available. Answer directly.\n")


@dataclass
class RunResult:
    acc: str
    mode: str
    condition: str
    final: dict | None = None
    residue_trust: list[float] | None = None  # per residue, NaN where unscored
    pair_judgments: dict[str, dict] = field(default_factory=dict)  # "D1-D2" -> {reliable, confidence}
    tool_calls: list[dict] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    n_model_calls: int = 0
    error: str | None = None

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        if self.residue_trust is not None:
            d["residue_trust"] = [None if (x != x) else x for x in self.residue_trust]
        return d


def extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end > start else None
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def parse_final(obj: dict, length: int, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, dict]:
    trust = np.full(length, np.nan)
    for seg in obj.get("segments", []) or []:
        a, b = int(seg["start"]), int(seg["end"])
        v = seg["trust"]
        v = TRUST_WORDS.get(str(v).lower(), None) if isinstance(v, str) else float(v)
        if v is None or not (1 <= a <= b <= length):
            continue
        trust[a - 1 : b] = min(1.0, max(0.0, v))
    judgments = {}
    wanted = {f"{a}-{b}" for a, b in pairs}
    for item in obj.get("domain_pairs", []) or []:
        key = f"{item.get('a')}-{item.get('b')}"
        rkey = f"{item.get('b')}-{item.get('a')}"
        key = key if key in wanted else rkey if rkey in wanted else None
        if key is None:
            continue
        reliable = item.get("placement_reliable")
        if isinstance(reliable, str):
            reliable = reliable.strip().lower() in ("true", "yes", "reliable")
        judgments[key] = {"reliable": bool(reliable), "confidence": float(item.get("confidence", 0.5))}
    return trust, judgments


def build_user_message(p, condition: str, cfg: Config, images_cache: dict | None = None) -> Message:
    parts = [domain_header(p)]
    images: list[bytes] = []
    if condition in ("image", "both"):
        key = p.acc
        if images_cache is not None and key in images_cache:
            images = images_cache[key]
        else:
            images = render_all(p)
            if images_cache is not None:
                images_cache[key] = images
        parts.append(
            "Two images follow. Image 1: the predicted structure, left colored by pLDDT, right colored by domain. "
            "Image 2: the predicted aligned error (PAE) matrix, with colored bars marking the domains."
        )
    if condition in ("text", "both"):
        parts.append(text_evidence(p, cfg))
    if condition not in ("image", "text", "both"):
        raise ValueError(f"unknown condition {condition!r}")
    return Message("user", "\n\n".join(parts), images)


def _log(res: RunResult, msg: Message) -> None:
    res.transcript.append({"role": msg.role, "text": msg.text, "n_images": len(msg.images)})


def run_one(
    client: LLMClient,
    p,
    mode: str,
    condition: str,
    cfg: Config,
    tool_ctx: ToolContext | None = None,
    images_cache: dict | None = None,
) -> RunResult:
    res = RunResult(p.acc, mode, condition)
    use_tools = mode == "agent"
    system = system_prompt(use_tools, cfg.tools)
    messages = [build_user_message(p, condition, cfg, images_cache)]
    _log(res, messages[0])

    if mode == "fixed":
        if tool_ctx is None:
            raise ValueError("fixed mode needs a ToolContext")
        outs = {}
        whole = [[1, p.length]]
        regions = whole + [[a, b] for d in p.domains for a, b in d.ranges[:1]][:3]
        for name in cfg.tools:
            try:
                outs[name] = run_tool(name, tool_ctx, {"regions": regions})
                res.tool_calls.append({"tool": name, "args": {"regions": regions}, "ok": True, "forced": True})
            except Exception as exc:  # noqa: BLE001
                outs[name] = {"error": str(exc)}
                res.tool_calls.append({"tool": name, "args": {"regions": regions}, "ok": False, "error": str(exc), "forced": True})
        extra = Message("user", "Results of tools that were run for you on the whole protein and the first range of each domain:\n" + json.dumps(outs) + "\nNow give the final answer.")
        messages.append(extra)
        _log(res, extra)

    steps = cfg.max_steps if use_tools else 1
    for step in range(steps + 1):
        force_final = (not use_tools) or step >= steps
        if force_final and use_tools and step >= steps:
            nudge = Message("user", "You have used all tool calls. Give the final answer now.")
            messages.append(nudge)
            _log(res, nudge)
        try:
            raw = client.generate(system, messages, json_mode=True)
        except Exception as exc:  # noqa: BLE001
            res.error = f"model call failed: {exc}"
            return res
        res.n_model_calls += 1
        reply = Message("assistant", raw)
        messages.append(reply)
        _log(res, reply)
        obj = extract_json(raw)
        if obj is None:
            fix = Message("user", 'Your reply was not a single JSON object. Reply again with only the JSON object in the required format.')
            messages.append(fix)
            _log(res, fix)
            if step >= steps:
                res.error = "could not parse a JSON reply"
                return res
            continue
        action = obj.get("action", "final" if "segments" in obj else None)
        if action == "call_tool" and use_tools and step < steps:
            name, args = obj.get("tool"), obj.get("args") or {}
            try:
                out = run_tool(name, tool_ctx, args)
                res.tool_calls.append({"tool": name, "args": args, "ok": True})
                text = "Tool result: " + json.dumps(out)
            except Exception as exc:  # noqa: BLE001
                res.tool_calls.append({"tool": name, "args": args, "ok": False, "error": str(exc)})
                text = f"Tool error: {exc}"
            msg = Message("user", text)
            messages.append(msg)
            _log(res, msg)
            continue
        if action == "final" or "segments" in obj:
            try:
                trust, judg = parse_final(obj, p.length, p.pairs)
            except (KeyError, TypeError, ValueError) as exc:
                res.error = f"bad final answer: {exc}"
                return res
            res.final, res.residue_trust, res.pair_judgments = obj, trust.tolist(), judg
            return res
        res.error = f"unexpected action {action!r}"
        return res
    res.error = "no final answer"
    return res
