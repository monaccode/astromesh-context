import dataclasses
import json
from importlib import resources

import httpx
import pytest
import respx

from astromesh_context import ContextEngine, HistorySource, StaticSource
from astromesh_context.eval.dataset import EvalCase, EvalDataset, doc_id_of, nimbus
from astromesh_context.eval.judge import JUDGE_PROMPT, judge_answer
from astromesh_context.eval.runner import ContextEval
from astromesh_context.scorer import openai_chat

DATA = resources.files("astromesh_context.eval") / "data" / "nimbus"
SENTINEL = "sk-SENTINEL-do-not-leak-9f3a"
ENV = ("CTX_EVAL_BASE_URL", "CTX_EVAL_API_KEY", "CTX_EVAL_MODEL", "CTX_EVAL_JUDGE_MODEL")


def fake_judge(reply, prompts=None):
    async def complete(prompt):
        if prompts is not None:
            prompts.append(prompt)
        return reply

    return complete


def tiny_dataset() -> EvalDataset:
    cases = [
        EvalCase(f"c{i}", f"question {i}", frozenset(), "exact", [], reference=f"ref {i}")
        for i in range(3)
    ]
    return EvalDataset(cases, [], frozenset(), doc_id_of)


async def tiny_factory(dataset, budget):
    return ContextEngine(
        [StaticSource("rules", name="system", pinned=True), HistorySource(key="history")],
        budget=budget,
    )


def fake_answer(content, usage=None):
    seen: list[list[dict]] = []

    async def answer(messages):
        seen.append(messages)
        return {"content": content, "usage": usage or {}}

    answer.seen = seen
    return answer


@pytest.mark.parametrize(
    ("reply", "score"), [("CORRECT", 1.0), ("PARTIAL", 0.5), ("WRONG", 0.0), (" CORRECT\n", 1.0)]
)
async def test_judge_maps_verdicts(reply, score):
    assert await judge_answer(fake_judge(reply), "q", "ref", "ans") == score


async def test_judge_prompt_is_exact():
    prompts: list[str] = []
    await judge_answer(fake_judge("WRONG", prompts), "Q?", "REF", "ANS")
    assert prompts == [
        "You grade an answer against a reference. Reply with exactly one word: CORRECT if the "
        "answer states the same facts as the reference, PARTIAL if it is incomplete but not "
        "wrong, WRONG otherwise.\n\nQuestion: Q?\nReference: REF\nAnswer: ANS"
    ]
    assert JUDGE_PROMPT.format(question="Q?", reference="REF", answer="ANS") == prompts[0]


@pytest.mark.parametrize("reply", ["correct", "CORRECT.", "The answer is CORRECT", "", "maybe"])
async def test_judge_unparsed_is_zero_and_counted(reply):
    ds = tiny_dataset()
    assert await judge_answer(fake_judge(reply), "q", "r", "a") == 0.0
    report = await ContextEval(
        ds, {"c": tiny_factory}, [1000], level=2, answer=fake_answer("a"), judge=fake_judge(reply)
    ).run()
    row = report.rows[0]
    assert row["accuracy"] == 0.0
    assert row["judge_unparsed"] == len(ds.cases)


def test_answers_file_covers_every_query():
    raw = (DATA / "answers.jsonl").read_bytes()
    assert b"\r" not in raw
    answers = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    queries = [
        json.loads(line) for line in (DATA / "queries.jsonl").read_bytes().decode().splitlines()
    ]
    assert [a["id"] for a in answers] == [q["id"] for q in queries]
    assert len(answers) == 60
    assert all(a["reference"].strip() for a in answers)
    assert [c.reference for c in nimbus().cases] == [a["reference"] for a in answers]


async def test_level2_row_has_accuracy():
    ds = tiny_dataset()
    verdicts = iter(["CORRECT", "PARTIAL", "WRONG"])

    async def judge(prompt):
        return next(verdicts)

    answer = fake_answer("a", {"prompt_tokens": 100, "completion_tokens": 10})
    report = await ContextEval(
        ds, {"c": tiny_factory}, [1000], level=2, answer=answer, judge=judge
    ).run()
    row = report.rows[0]
    assert row["accuracy"] == pytest.approx(0.5)
    assert row["judge_unparsed"] == 0
    assert row["prompt_tokens"] == 100
    assert row["completion_tokens"] == 10
    assert isinstance(row["latency_ms"], float)
    assert row["latency_ms"] >= 0
    assert row["latency_ms"] == round(row["latency_ms"], 1)
    assert "recall" in row  # level-1 columns stay
    assert len(answer.seen) == 3
    assert answer.seen[0][-1] == {"role": "user", "content": "question 0"}
    assert report.meta["level"] == 2
    md = report.to_markdown()
    assert "accuracy" in md
    assert "latency ms (p50)" in md


async def test_level2_missing_usage_counts_zero():
    report = await ContextEval(
        tiny_dataset(),
        {"c": tiny_factory},
        [1000],
        level=2,
        answer=fake_answer("a"),
        judge=fake_judge("CORRECT"),
    ).run()
    assert report.rows[0]["prompt_tokens"] == 0
    assert report.rows[0]["completion_tokens"] == 0


def test_level2_requires_models_and_level_3_rejected():
    ds = dataclasses.replace(tiny_dataset(), cases=[])
    with pytest.raises(ValueError, match="answer"):
        ContextEval(ds, {}, [1000], level=2)
    with pytest.raises(ValueError, match="level"):
        ContextEval(ds, {}, [1000], level=3)


@respx.mock
async def test_openai_chat_posts_messages_and_returns_usage():
    route = respx.post("http://llm.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            },
        )
    )
    chat = openai_chat("http://llm.test/v1/", "sk-x", "m1")
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    assert await chat(messages) == {
        "content": "hi",
        "usage": {"prompt_tokens": 7, "completion_tokens": 2},
    }
    req = route.calls[0].request
    assert json.loads(req.content) == {"model": "m1", "messages": messages, "temperature": 0}
    assert req.headers["authorization"] == "Bearer sk-x"


@respx.mock
async def test_openai_chat_without_usage():
    respx.post("http://llm.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    )
    assert await openai_chat("http://llm.test/v1", "k", "m")([]) == {"content": "hi", "usage": {}}


@respx.mock
def test_api_key_not_in_report(tmp_path, monkeypatch, capsys):
    pytest.importorskip("astromesh_rag")
    pytest.importorskip("astromesh_okf")
    from astromesh_context.cli import main

    monkeypatch.setenv("CTX_EVAL_BASE_URL", "https://llm.example.test/secret-path/v1")
    monkeypatch.setenv("CTX_EVAL_API_KEY", SENTINEL)
    monkeypatch.setenv("CTX_EVAL_MODEL", "answer-m")
    monkeypatch.setenv("CTX_EVAL_JUDGE_MODEL", "judge-m")
    route = respx.post("https://llm.example.test/secret-path/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "CORRECT"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )
    )
    code = main(
        ["eval", "--level", "2", "--subset", "2", "--budgets", "1000", "--out", str(tmp_path)]
    )
    assert code == 0
    assert route.called
    models = {json.loads(c.request.content)["model"] for c in route.calls}
    assert models == {"answer-m", "judge-m"}
    out = capsys.readouterr()
    written = "".join(p.read_text(encoding="utf-8") for p in tmp_path.iterdir())
    for text in (written, out.out, out.err):
        assert SENTINEL not in text
        assert "secret-path" not in text
    data = json.loads(next(tmp_path.glob("*L2.json")).read_text(encoding="utf-8"))
    assert data["answer_model"] == "answer-m"
    assert data["judge_model"] == "judge-m"
    assert data["endpoint_host"] == "llm.example.test"
    assert all(r["accuracy"] == 1.0 for r in data["rows"])


def test_cli_level2_missing_env(tmp_path, monkeypatch, capsys):
    from astromesh_context.cli import main

    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CTX_EVAL_BASE_URL", "http://x.test/v1")
    assert main(["eval", "--level", "2", "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: set CTX_EVAL_API_KEY")
    assert "CTX_EVAL_MODEL" in err
