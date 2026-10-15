"""astromesh-context command line: ``eval`` and ``explain``."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from astromesh_context.errors import ContextError, MissingExtraError


def _embedder(kind: str):
    if kind == "st":
        try:
            from astromesh_rag.embedders import SentenceTransformersEmbedder
        except ImportError:
            raise MissingExtraError("install astromesh-context[eval]") from None
        return SentenceTransformersEmbedder("all-MiniLM-L6-v2")
    try:
        from astromesh_rag.testing import HashEmbedder
    except ImportError:
        raise MissingExtraError("install astromesh-context[rag]") from None
    return HashEmbedder()


_REQUIRED_ENV = ("CTX_EVAL_BASE_URL", "CTX_EVAL_API_KEY", "CTX_EVAL_MODEL")


def _parse_budgets(text: str) -> list[int]:
    try:
        budgets = [int(p) for p in text.split(",") if p.strip()]
    except ValueError:
        budgets = []
    if not budgets or any(b <= 0 for b in budgets):
        raise argparse.ArgumentTypeError(f"expected positive integers like 2000,4000, got {text!r}")
    return budgets


def _level2(env: dict[str, str]):
    """The answer and judge callables plus the report meta; the key never reaches the meta."""
    from astromesh_context.scorer import openai_chat, openai_compatible

    base_url, api_key, model = (env[n] for n in _REQUIRED_ENV)
    judge_model = env.get("CTX_EVAL_JUDGE_MODEL") or model
    meta = {
        "answer_model": model,
        "judge_model": judge_model,
        "endpoint_host": urlparse(base_url).hostname,
    }
    return (
        openai_chat(base_url, api_key, model),
        openai_compatible(base_url, api_key, judge_model),
        meta,
    )


def _eval(args: argparse.Namespace) -> int:
    from astromesh_context.eval.dataset import nimbus
    from astromesh_context.eval.runner import ContextEval, default_configs

    dataset = nimbus()
    if args.subset is not None:
        dataset = dataset.subset(args.subset)
    extra: dict = {}
    models: dict = {}
    if args.level == 2:
        missing = [n for n in _REQUIRED_ENV if not os.environ.get(n)]
        if missing:
            print(f"error: set {', '.join(missing)}", file=sys.stderr)
            return 2
        answer, judge, extra = _level2(dict(os.environ))
        models = {"answer": answer, "judge": judge}
    configs = default_configs(_embedder(args.embedder))
    try:
        report = asyncio.run(
            ContextEval(dataset, configs, args.budgets, level=args.level, **models).run()
        )
    except Exception as exc:
        if args.level == 2 and type(exc).__module__.startswith("httpx"):
            # httpx messages carry the request URL; keep the endpoint out of the output
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise ContextError(
                f"model call failed ({type(exc).__name__}{f' {status}' if status else ''})"
            ) from None
        raise
    report.meta.update(extra)
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


def _mcp(args: argparse.Namespace) -> int:
    from astromesh_context.integrations.mcp import serve
    from astromesh_context.spec import load_engine

    serve(load_engine(args.config), name=args.name).run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astromesh-context")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="compare policies on the Nimbus corpus")
    ev.add_argument("--level", type=int, choices=[1, 2], default=1)
    ev.add_argument("--budgets", type=_parse_budgets, default=[750, 1000, 2000])
    ev.add_argument("--embedder", choices=["hash", "st"], default="hash")
    ev.add_argument("--subset", type=int, default=None, help="use only the first N questions")
    ev.add_argument("--out", default="eval-results", help="directory for the .json and .md files")
    ev.set_defaults(run=_eval)

    ex = sub.add_parser("explain", help="build one window and print its decision trace")
    ex.add_argument("--config", required=True, help="a kind: ContextPolicy YAML file")
    ex.add_argument("--query", required=True)
    ex.set_defaults(run=_explain)

    mc = sub.add_parser("mcp", help="serve a policy's ctx_build tool over MCP (stdio)")
    mc.add_argument("--config", required=True, help="a kind: ContextPolicy YAML file")
    mc.add_argument("--name", default="astromesh-context", help="server name")
    mc.set_defaults(run=_mcp)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    try:
        return args.run(args)
    except (ContextError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
