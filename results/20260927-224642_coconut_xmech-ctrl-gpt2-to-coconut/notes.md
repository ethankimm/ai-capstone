## 2026-09-28 — Experiment A2, plain GPT-2 (no latent training) → Coconut: question text alone carries only 5-8% of the unrestricted map's cf_joint (coconut, run_id: 20260927-224642_coconut_xmech-ctrl-gpt2-to-coconut)

**Goal:** Round-3/4 P4 caveat: the unrestricted CODI↔Coconut map reads all six source latents and
writes whole target vectors, so it could be routing question-derived information rather than the
source's computed reasoning. This is the cleanest version of that control: map from plain,
untrained-for-GSM8K HF `gpt2` hidden states of the SAME question text — a representation that has
done no task-specific reasoning at all — into Coconut's carriers.
**Design** (`gpt2_plain_dump.py` + `xmech_common.run_transplant`'s new `feat`/`feat_sites` params;
`xmech_coconut.py --mode ctrl --ctrl_kind gpt2`): plain-GPT2 features dumped locally (MPS, no GPU
needed, 9319 questions in 75s): for each question, `[layer6 last-token hidden, layer12 last-token
hidden, layer12 mean-over-question-tokens]` (3x768=2304 dims), question text prepped identically to
every mechanism's own encoder (`strip().replace("  ", " ")`). Same 259 eval-split ladder
recipients/donors as the unrestricted P4 run (pinned by the TRUE CODI/Coconut correctness
intersection, `xmech_common._ladder_pairs` — the GPT-2 features only replace what feeds the ridge
map, not which pairs are tested). Same fit split (8000 train questions), same carriers (passes
1,4).
**Mechanism / model:** Coconut `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33` (target); plain
`gpt2` (124M, no fine-tuning) supplies the source features.
**Command:** `scripts/xmech_coconut.py --checkpoint_path .../checkpoint_33 --questions
xmech_questions.jsonl --codi_latents codi_latents.pt --out_latents coconut_latents.pt --mode ctrl
--ctrl_kind gpt2 --gpt2_latents gpt2_latents.pt --slug xmech-ctrl-gpt2-to-coconut --stage full_run`
(full line in `eval_command.txt`). Pod `jbzj5cd4diacxb` (RTX A40 secure, $0.49/hr, EU-SE-1); ran
alongside 7 other jobs on one GPU.
**Headline results** (259 recipients; siblings `20260927-224224` (z0 only) and
`20260927-224922` (z1-z5); reference unrestricted run `20260927-085211`, mapped_all cf_joint
L2/L3/L4 = 0.32/0.28/0.24):

| cf_joint (null ≤0.015) | L2 | L3 | L4 |
|---|---|---|---|
| own_carriers (this run) | 0.270 | 0.313 | 0.351 |
| **mapped_carriers (plain GPT-2)** | **0.023** | **0.023** | **0.012** |
| shuffled_carriers | 0.015 | 0.008 | 0.008 |
| % of unrestricted mapped_all (0.32/0.28/0.24) | 7% | 8% | 5% |

Map fit (held-out R², plain-GPT2 features → Coconut pass): pass0 0.128, pass1 0.053, pass2 0.102,
pass3 0.091, pass4 0.035, pass5 0.128 — much lower than any CODI-derived feature set, and the
mapped condition (0.012-0.023) is barely above the shuffled-pair control (0.008-0.015), i.e. close
to the floor of "any map that has seen the right marginals but no real pairing." self_mapped_carriers
(recipient's own GPT-2 features mapped into itself) unchanged only 19.3% — the map barely
reconstructs the recipient's own content either.

**Interpretation:**
- **The question-confound hypothesis for the unrestricted P4 result is ruled out by this control.**
  A representation with zero task-specific reasoning (plain GPT-2's own hidden states on the exact
  question text) reaches only 5-8% of what the unrestricted 6-site CODI→Coconut map achieves — far
  below the ~40% "well below" branch of the pre-registered decision rule, and barely above its own
  shuffled-pair control. Whatever the unrestricted map is reading from CODI's latents, it is not
  reducible to "the question, re-encoded."
- Contrast with the sibling `20260927-224224` (CODI's OWN z0, same "no reasoning yet" logical
  position in the pipeline, but the model's own trained encoding): 74-110% vs this run's 5-8%. The
  gap between "plain GPT-2's encoding of the text" and "CODI's own z0" is the value added by CODI's
  training/architecture at the very first step, before any of its iterative loop — separate from
  the value added by later iterations.
**Caveats:** n=259. GPT-2 features are 3 fixed probe points (two last-token layers + one mean-pool);
a differently-chosen feature set (more layers, attention-weighted pooling) might do somewhat better,
but the gap to CODI's own z0 (a single 768-dim vector, less total information) is large enough that
this is unlikely to close it. own_carriers here reads slightly lower than the "0.30" quoted for all
three levels in round-3/4 notes — see caveat on the sibling run; doesn't affect this run's own ratio,
which is computed against the precisely-read unrestricted mapped_all.
**Next:** Reverse direction (`20260927-225751_codi_xmech-ctrl-gpt2-to-codi`) for the same control
into CODI. Experiment B addresses the OTHER round-4 open problem (basis mismatch for cross-problem
DAS subspaces) and found a much bigger effect size than this question-confound check.
