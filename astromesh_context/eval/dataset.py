"""The Nimbus dataset: labeled questions, documents, distractors and an OKF bundle."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from astromesh_context.types import ContextItem

# Eight generic Nimbus small-talk exchanges. They carry no answer to any question.
SMALL_TALK: list[tuple[str, str]] = [
    ("Hola, buenos días.", "Hola, soy el asistente de Nimbus. ¿En qué te puedo ayudar?"),
    ("Gracias por la ayuda de ayer.", "Con gusto. Si algo falla otra vez, avísame."),
    ("¿Estás ahí?", "Sí, aquí estoy."),
    ("Tengo una duda sobre mi cuenta.", "Claro, cuéntame qué pasa."),
    ("Perdón, escribí rápido.", "No hay problema, lo entendí."),
    ("Hello, is anyone there?", "Hi, this is Nimbus support. How can I help?"),
    ("Una pregunta más, por favor.", "Dime."),
    ("Eso es todo por ahora.", "Perfecto. Aquí estaré si necesitas algo más."),
]

_POLICY = re.compile(r"^pol-.+-(2024|2026)$")


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    relevant: frozenset[str]
    type: str
    history: list[dict]
    reference: str | None = None


@dataclass
class EvalDataset:
    cases: list[EvalCase]
    docs: list[dict]
    distractors: frozenset[str]
    doc_id_of: Callable[[ContextItem], str | None]
    meta: dict[str, Any] = field(default_factory=dict)

    def subset(self, n: int) -> EvalDataset:
        """The first ``n`` cases, same documents."""
        return dataclasses.replace(self, cases=self.cases[:n])


def data_dir():
    return resources.files("astromesh_context.eval") / "data" / "nimbus"


def _read_lines(name: str) -> list[dict]:
    text = (data_dir() / name).read_bytes().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _sha256(name: str) -> str:
    return hashlib.sha256((data_dir() / name).read_bytes()).hexdigest()


def doc_id_of(item: ContextItem) -> str | None:
    """The corpus document an item came from; None for history, prompts and the like."""
    meta = item.meta
    if "doc_id" in meta:
        return meta["doc_id"]
    if "concept_id" in meta:
        return str(meta["concept_id"]).rsplit("/", 1)[-1]
    return None


def _history(index: int, turns: int) -> list[dict]:
    messages: list[dict] = []
    for t in range(turns):
        user, assistant = SMALL_TALK[(index + t) % len(SMALL_TALK)]
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages


def nimbus(history_turns: int = 4) -> EvalDataset:
    docs = _read_lines("corpus.jsonl")
    distractors = frozenset(
        d["id"]
        for d in docs
        if (_POLICY.match(d["id"]) and d["id"].endswith("-2024"))
        or d.get("metadata", {}).get("trap") == "noise"
    )
    cases = [
        EvalCase(
            id=q["id"],
            query=q["query"],
            relevant=frozenset(q["relevant"]),
            type=q["type"],
            history=_history(i, history_turns),
            reference=None,
        )
        for i, q in enumerate(_read_lines("queries.jsonl"))
    ]
    return EvalDataset(
        cases=cases,
        docs=docs,
        distractors=distractors,
        doc_id_of=doc_id_of,
        meta={
            "corpus_sha256": _sha256("corpus.jsonl"),
            "queries_sha256": _sha256("queries.jsonl"),
        },
    )


def _title(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def _yaml_scalar(value: str) -> str:
    if re.fullmatch(r"[\w][\w ,.()¿?¡!/-]*", value) and ": " not in value:
        return value
    return json.dumps(value, ensure_ascii=False)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))  # explicit bytes: "\n" on every platform


def nimbus_bundle(dest: Path) -> Path:
    """Write the 30 policy documents as an OKF bundle under ``dest``; byte-deterministic."""
    dest = Path(dest)
    policies = sorted(
        (d for d in _read_lines("corpus.jsonl") if d["id"].startswith("pol-")),
        key=lambda d: d["id"],
    )
    links = []
    for doc in policies:
        title = _title(doc["text"])
        front = ["type: Policy", f"title: {_yaml_scalar(title)}"]
        if doc["id"].endswith("-2024"):
            front.append("status: deprecated")
        _write(
            dest / "policies" / f"{doc['id']}.md",
            "---\n" + "\n".join(front) + "\n---\n" + doc["text"] + "\n",
        )
        links.append(f"* [{doc['id']}](policies/{doc['id']}.md) - {title}")
    _write(
        dest / "index.md",
        '---\nokf_version: "0.2"\n---\n# Policies\n\n' + "\n".join(links) + "\n",
    )
    return dest
