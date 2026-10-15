# Nimbus corpus (vendored)

- Source: https://github.com/monaccode/astromesh-rag, `examples/corpus/`
- Commit: `a86bfba2f44ccd591c9a05adee37f0b5fe321f2f`
- License: Apache-2.0 (the license of astromesh-rag)
- Files: `corpus.jsonl` (200 documents) and `queries.jsonl` (60 labeled questions), copied
  byte for byte. `tests/test_eval_smoke.py::test_vendored_corpus_hash` pins their sha256.

Do not edit these files here. To update them, copy them from a newer astromesh-rag commit,
record the commit above and update the pinned hashes.
