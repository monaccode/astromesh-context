import hashlib

from astromesh_context.order import order_items
from astromesh_context.render import render_messages, render_text, stable_block, stable_prefix_hash
from astromesh_context.types import ContextItem, ScoredItem, Trace, Window


def _it(id, content, kind="knowledge", stability="volatile", source="kb", meta=None):
    return ContextItem(id, content, source, kind, stability, meta=meta or {})


HIST = _it(
    "h",
    "ignored",
    kind="history",
    stability="session",
    source="history",
    meta={"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]},
)
SYS = _it("sys", "You are helpful.", kind="instruction", stability="stable", source="inst")
KB = _it("k1", "Refunds take 5 days.")


def test_instruction_rendered_raw_knowledge_wrapped():
    out = render_text([SYS, KB, HIST])
    assert (
        out == 'You are helpful.\n\n<context id="k1" source="kb">\nRefunds take 5 days.\n</context>'
    )


def test_openai_messages_shape():
    msgs = render_messages([SYS, KB, HIST], "q?", "openai")
    assert msgs[0] == {"role": "system", "content": render_text([SYS, KB])}
    assert msgs[1:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
        {"role": "user", "content": "q?"},
    ]


def test_anthropic_messages_shape():
    out = render_messages([SYS, KB, HIST], "q?", "anthropic")
    assert out == {
        "system": render_text([SYS, KB]),
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "q?"},
        ],
    }


def test_empty_query_adds_no_user_message_and_history_fallback():
    h = _it("h2", "plain", kind="history", source="history")
    assert render_messages([h], "", "openai")[1:] == [{"role": "user", "content": "plain"}]


def test_stable_hash_unchanged_when_only_volatile_changes():
    def build(vol):
        scored = [ScoredItem(i, 1.0, 1.0) for i in (SYS, _it("kx", vol))]
        return order_items(scored, [])

    a, b = build("one"), build("two")
    assert stable_prefix_hash(a) == stable_prefix_hash(b)
    assert stable_prefix_hash(a) == hashlib.sha256(stable_block(a).encode()).hexdigest()
    assert stable_block(a) == "You are helpful."
    assert stable_prefix_hash([]) == hashlib.sha256(b"").hexdigest()


def test_window_methods():
    w = Window("q?", [SYS, KB, HIST], 10, 100, Trace([], []), "h")
    assert w.to_text() == render_text([SYS, KB])
    assert w.to_messages()[-1] == {"role": "user", "content": "q?"}
    assert w.to_messages("anthropic", include_query=False)["messages"][-1]["content"] == "yo"
