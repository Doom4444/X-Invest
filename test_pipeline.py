# test_pipeline.py — run with: python test_pipeline.py

print("=" * 50)
print("TEST 1: Retriever")
print("=" * 50)
try:
    from rag.core.retriever import Retriever
    r = Retriever()
    docs, dists = r.retrieve("what is P/E ratio")
    print(f"Docs returned: {len(docs)}")
    if docs:
        print(f"First doc preview: {docs[0][:150]}")
    else:
        print("EMPTY — no documents ingested yet (expected)")
except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 50)
print("TEST 2: context_builder")
print("=" * 50)
try:
    from pipeline.context_builder import build_context
    ctx = build_context("what is inflation")
    if ctx:
        print(f"Context length: {len(ctx)}")
        print(f"Preview: {ctx[:300]}")
    else:
        print("EMPTY — expected if no documents ingested yet")
except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 50)
print("TEST 3: Live data fetch (yfinance)")
print("=" * 50)
try:
    ctx = build_context("what is the price of AAPL")
    if ctx:
        print(f"Context length: {len(ctx)}")
        print(f"Preview: {ctx[:300]}")
    else:
        print("EMPTY — entity extractor may not have caught AAPL")
except Exception as e:
    print(f"ERROR: {e}")

print()
print("Done.")
print()
print("=" * 50)
print("TEST 4: Prediction signal in context")
print("=" * 50)
try:
    from pipeline.context_builder import build_context
    ctx = build_context("what is the signal for AAPL", "test456")
    if "[Prediction]" in ctx:
        print("SIGNAL FOUND in context:")
        # print just the prediction section
        start = ctx.find("[Prediction]")
        print(ctx[start:start+200])
    else:
        print("SIGNAL NOT IN CONTEXT")
        print("Full context sections:")
        print(ctx[:500])
except Exception as e:
    print(f"ERROR: {e}")
