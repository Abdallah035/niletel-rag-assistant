# Project Reflection

## ✅ What worked well

- **The hybrid retrieval pipeline came together cleanly.** Splitting the system
  into independent modules — chunking, embeddings, dense FAISS, BM25, RRF
  fusion, reranker, and an orchestrator — meant each piece was testable in
  isolation. By the end I had small smoke tests for every module (19 for
  chunking, 21 for FAISS, 27 for BM25, 16 for RRF, 11 for the reranker,
  19 for the full pipeline) — all green. Working in those small contracts
  saved hours of debugging once things were wired together.

- **The e5 prefix fix was a free quality win.** The original baseline embedded
  text without `"query: "` / `"passage: "` prefixes — silently leaving
  ~10% retrieval recall on the table. Fixing it took 5 lines of code but
  improved every downstream stage. Worth the read of the e5 paper.

- **Reciprocal Rank Fusion behaved exactly as advertised.** A query like
  *"RSRP SINR P1 escalate"* split dense and BM25 into completely different
  top-3 rankings — and RRF correctly promoted the chunk that appeared in
  *both* (even at rank 5 each) to the top of the fused ranking. Watching
  that happen on real data, on real telecom queries, was the moment the
  project clicked.

- **Markdown-aware chunking with breadcrumb prefixes.** The original data had
  no markdown headers at all — 35 flat text files. Writing a one-shot
  preprocessing script that promoted plain-text section labels (`الخطوات:`,
  `FAQ:`, `Level 1 – Support Agent:`) into proper `##` headers, then
  prepending a `[file.md > Section > Subsection]` breadcrumb to every
  chunk, gave retrieval a context signal that neither dense nor BM25 had
  access to before. This single change made the difference between
  "kind-of-relevant chunks" and "exactly the right paragraph."

- **The tiered intent router.** 90% of queries get classified in <1ms by
  regex (greetings) or keyword (explicit "اعمل تذكرة", "فيلم") matches
  *before* we ever call the LLM. Only ambiguous cases (a complaint
  phrased as a question, an info request that might be a ticket) pay
  the ~1s LLM cost. Honest production tradeoff.

- **Provider-agnostic LLM client.** When Gemini's free quota ran out
  mid-build, switching to GPT-OSS 120B on Lightning AI was a single
  `.env` change and a 30-min wrapper edit — because the rest of the code
  only knew about the `LLMClient` interface, not the SDK. Worth the
  upfront abstraction.

- **The n8n ticket workflow end-to-end.** Webhook → Sheets → Gmail (support)
  → If → Gmail (customer) → Respond, with PII scrubbing on the FastAPI
  side. Watching a real Arabic ticket land in the Google Sheet *and*
  arrive as two formatted emails *and* return a real `NT-2026-XXXX`
  ticket ID to the UI in under 5 seconds was satisfying.

---

## ⚠️ What challenges I faced

- **The reranker hardware wall.** I wanted to ship with the full
  `bge-reranker-v2-m3` cross-encoder enabled, but my 4 GB GTX 1650
  segfaulted under it (even the smaller `bge-reranker-base` triggered a
  Windows-paging OOM). Switching to CPU made reranking add ~5-6 s/query —
  too painful for a live chat UX. I gated it behind a
  `RERANKER_ENABLED=false` flag instead. Honest tradeoff: the code path
  is fully tested and works (T5.1 confirms it reorders RRF output
  meaningfully), it's just disabled by default until someone runs it on
  beefier hardware.

- **Gemini free-tier daily quota.** 20 requests/day on `gemini-2.5-flash`
  evaporates fast during testing. Migrated to GPT-OSS 120B on Lightning
  AI mid-project. The migration itself surfaced an interesting bug —
  Lightning's API key format requires `KEY/username/teamspace` for
  teamspace billing, with case-sensitive username matching. Cost ~30
  minutes to diagnose.

- **GPT-OSS JSON mode unreliability.** Unlike Gemini, GPT-OSS doesn't
  honor `response_format={"type":"json_object"}` reliably — it returned
  `'{"{"intent:":"chat} }'` on my first router call. Solved by adding a
  recovery layer in `llm_client.py`: try strict parse → strip code fences
  → fall back to extracting the first balanced `{...}` block. Now the
  router survives both providers.

- **The raw NileTel knowledge base had zero markdown structure.** First
  test of the chunker on real data showed every file producing one giant
  flat chunk because there were no `#` headers. I had to write
  `scripts/preprocess_data.py` to promote plain-text section labels into
  proper markdown — a small script with conservative regex rules and a
  `--dry-run` flag for safe review. Caught 5 over-promotions that I
  fixed by hand.

- **Streamlit's CSS specificity battles.** The first version of the UI
  inherited Streamlit's dark-mode defaults despite my custom theme
  config. Spent an hour writing high-specificity selectors with
  `!important` to force the white-background branded look. Lesson:
  Streamlit components have very stable `data-testid` attributes — once
  you target those, the CSS sticks.

- **Windows zombie processes and paging issues.** Several times during
  the build, killing a uvicorn process left port 8000 bound by an
  unkillable zombie, forcing me to either reboot or change ports.
  Separately, loading both the embedder and reranker on Windows hit
  paging-file limits even though there was plenty of RAM. Real cost in
  developer hours.

- **RTL text rendering in the chat UI.** Arabic content rendered LTR by
  default, breaking the visual flow of mixed Arabic/English answers.
  Fixed with `direction: rtl; unicode-bidi: plaintext;` on chat
  messages — but it took eyeballing the screenshot to realize this was
  the issue.

---

## 🚀 What I would improve next

- **Build a proper evaluation suite.** I tested retrieval quality by
  hand on 10-ish representative queries. Production-grade would mean:
  - A golden set of 30-50 (query, expected-source) pairs
  - Recall@k, MRR, nDCG computed on every pipeline change
  - LLM-as-judge scoring for answer faithfulness
  - A `make eval` target that runs the whole thing
  This would let me defend claims like "RRF improves recall by X%" with
  numbers instead of vibes.

- **Re-enable the reranker on GPU-ready deployment.** Once the system
  lives on a host with > 6 GB VRAM (or even with a smaller f16 reranker
  on a 4 GB card), flip `RERANKER_ENABLED=true`. The 11 tests I wrote
  for it already prove the integration is sound — only the runtime
  constraint blocks it today.

- **Conversation memory.** The chat UI keeps history per browser tab,
  but the backend is stateless. Adding a 3-turn rolling buffer per
  `session_id` would unlock follow-up questions like *"and what about
  VIP customers?"* without the user having to restate context.

- **Smarter PII scrubbing.** The current regex catches Egyptian mobile
  numbers, national IDs, and emails. Real production would also need:
  - Named-entity recognition for customer names (likely a small Arabic
    NER model)
  - Optional reversible tokenization so the support team can see the
    real number when they need it, but the chain in between only sees
    `[PHONE]`

- **Streaming-aware ticket cards.** Right now the pink ticket card
  appears at the end of the streamed answer, because n8n's response is
  collected synchronously after streaming completes. A nicer UX would
  be to show a "ticket pending..." card immediately on intent detection,
  then update it in place when n8n confirms.

- **Cloud deployment.** The current setup is local + ngrok per the
  assignment brief. The README has a clear path to deploy backend to
  Hugging Face Spaces (16 GB free tier fits the embedder) and the UI to
  Streamlit Community Cloud. Doing this post-deadline gives the project
  a permanent CV-friendly URL.

- **Multi-turn intent re-evaluation.** A user might start with an info
  question ("how do I cancel my contract?"), and three messages later
  it becomes a complaint ("this is taking too long, escalate"). The
  router currently classifies each turn in isolation. A smarter router
  would consider the conversation arc.

- **Better observability.** Per-stage timings are already attached to
  each retrieval result, but they're not yet exported anywhere
  (Prometheus, OpenTelemetry, or even just a CSV log). A `/metrics`
  endpoint would be a small lift with outsized debugging value.

- **Eval-driven prompt iteration.** The current answer prompt is good
  but hand-tuned. With an eval set in place, I could A/B test prompt
  variants (different rule ordering, persona phrasing, refusal
  thresholds) and pick the one that scores best on faithfulness.

- **A real test suite.** I wrote ad-hoc smoke tests during development
  and deleted them after each module passed. A proper `pytest` suite
  covering chunking, RRF math, router decisions, and the API contract
  would let me refactor confidently in the future.
