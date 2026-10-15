"""LLM judge for ContextEval level 2: grade an answer against a hand-written reference."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

JUDGE_PROMPT = (
    "You grade an answer against a reference. Reply with exactly one word: CORRECT if the answer "
    "states the same facts as the reference, PARTIAL if it is incomplete but not wrong, WRONG "
    "otherwise.\n\nQuestion: {question}\nReference: {reference}\nAnswer: {answer}"
)
_VERDICTS = {"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0}


async def grade_answer(
    complete: Callable[[str], Awaitable[str]], question: str, reference: str, answer: str
) -> tuple[float, bool]:
    """``(score, parsed)``; a reply that is not exactly one verdict word scores 0.0, unparsed."""
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    reply = await complete(prompt)
    score = (
        _VERDICTS.get(reply.strip().rstrip(".!").strip().upper())
        if isinstance(reply, str)
        else None
    )
    return (0.0, False) if score is None else (score, True)


async def judge_answer(
    complete: Callable[[str], Awaitable[str]], question: str, reference: str, answer: str
) -> float:
    """1.0 for CORRECT, 0.5 for PARTIAL, 0.0 for WRONG or any other reply."""
    return (await grade_answer(complete, question, reference, answer))[0]
