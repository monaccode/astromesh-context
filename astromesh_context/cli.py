"""astromesh-context command line: ``eval`` and ``explain``."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from astromesh_context.errors import ContextError, MissingExtraError


def _embedder(kind: str):
    try:
        if kind == "st":
            from astromesh_rag.embedders import SentenceTransformersEmbedder

            return SentenceTransformersEmbedder("all-MiniLM-L6-v2")
        from astromesh_rag.testing import HashEmbedder
    except ImportError:
        raise MissingExtraError("install astromesh-context[eval]") from None
    return HashEmbedder()


def _parse_budgets(text: str) -> list[int]:
    try:
        budgets = [int(p) for p in text.split(",") if p.strip()]
    except ValueError:
        budgets = []
    if not budgets or any(b <= 0 for b in budgets):
        raise argparse.ArgumentTypeError(f"expected positive integers like 2000,4000, got {text!r}")
    return budgets


def _eval(args: argparse.Namespace) -> int:
    from astromesh_context.eval.dataset import nimbus
    from astromesh_context.eval.runner import ContextEval, default_configs

    dataset = nimbus()
    if args.subset is not None:
        dataset = dataset.subset(args.subset)
    configs = default_configs(_embedder(args.embedder))
    report = asyncio.run(ContextEval(dataset, configs, args.budgets, level=args.level).run())
    report.meta["command"] = "astromesh-context " + " ".join(args.argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now(timezone.utc):%Y-%m-%d}-{args.embedder}-L{args.level}"
    markdown = report.to_markdown()
    (out / f"{stem}.json").write_text(report.to_json(), encoding="utf-8", newline="\n")
    (out / f"{stem}.md").write_text(markdown, encoding="utf-8", newline="\n")
    print(markdown, end="")
    return 0


def _explain(args: argparse.Namespace) -> int:
    from astromesh_context.spec import load_engine

    engine = load_engine(args.config)
    window = asyncio.run(engine.build(args.query))
    print(window.trace.explain())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="astromesh-context")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="compare policies on the Nimbus corpus")
    ev.add_argument("--level", type=int, choices=[1], default=1)
    ev.add_argument("--budgets", type=_parse_budgets, default=[2000, 4000, 8000])
    ev.add_argument("--embedder", choices=["hash", "st"], default="hash")
    ev.add_argument("--subset", type=int, default=None, help="use only the first N questions")
    ev.add_argument("--out", default="eval-results", help="directory for the .json and .md files")
    ev.set_defaults(run=_eval)

    ex = sub.add_parser("explain", help="build one window and print its decision trace")
    ex.add_argument("--config", required=True, help="a kind: ContextPolicy YAML file")
    ex.add_argument("--query", required=True)
    ex.set_defaults(run=_explain)

    args = parser.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    try:
        return args.run(args)
    except ContextError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
