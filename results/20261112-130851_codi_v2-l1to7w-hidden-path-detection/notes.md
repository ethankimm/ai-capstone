## 2026-11-12 — Hidden-path detection on six RuleChain CODI models (t34): latents reveal middle-rule branches the question pass and answer cannot in gated / SIM-CODI (+.40 to +.49 balanced accuracy); in plain, relay and stabilised CODI the question pass already carries them (codi, run_id: 20261112-130851_codi_v2-l1to7w-hidden-path-detection)

**Goal:** t34 (pre-registered 2026-11-12 00:15), thread 1d: a monitorability test closer to what a monitor is for. Can a probe on the latents tell which branch a middle rule took, when a probe on the question pass plus the model's answer cannot?
**Mechanism / model:** gated, sim, simsf, startlast, relay, stab (the t26 RuleChain CODI models, same checkpoints).
**Data:** synthetic-depth2 L4-L6; fit on 2,000 train items per level (seed 30), score on all valid items of the level. Target: branch at rule k (gold walk), k = 2 .. L-1.
**Command:** `../codi/.venv/bin/python scripts/necessity/hidden_path.py <model> mps <out>` (queue `codi_audit_artifacts/t34/queue2.sh`).
**Method:** features per item from one natural run: Q = <bot> layers 4/8/12; A = own answer one-hot; Z = z_1..z_5; H = latent passes 0-4 at layer 8. Ridge to +-1, balanced accuracy. Gain = max(QAZ, QAH) - QA. Shuffled-label control on QAZ.
**Headline results** (QA range; best latent set; mean gain):
- gated .46-.56; H .99-1.00; +.49. sim .47-.65; .93-.99; +.40. simsf .52-.65; QAZ .98-1.00; +.40.
- startlast .63-.92; +.03. relay .75-.94; +.02. stab .67-.95; +.01. The baseline's signal is in Q (the answer one-hot alone is ~.50).
- Shuffled control <= .55 except one cell (gated L5 k = 3, .61).
**Predictions:** all model-level predictions held (sim QA max .650, at the edge). Shuffled <= .56 missed in one cell. Not predicted: startlast's question pass carries the branches (.92 at L4 k = 2) though t35 reads its v_1 exactly at only .31.
**Interpretation:** Where training routes the chain through the latents (gated, SIM-CODI), the latents are the only place a linear monitor finds the path: chance from question + answer, near-perfect from the latents. Where the question pass computes the chain (start-first stab and relay, and start-last CODI in a coarse form), the latents add nothing. So latents are worth monitoring exactly when the hop count says they are used. Exact-value probes understate what a single pass computes: branch bits are readable where exact values are not.
**Gotchas hit:** numpy on macOS prints spurious matmul RuntimeWarnings; the smoke had the same warnings and its scores are sane.
**Caveats:** linear probes only; gold-walk labels (on wrong items the model's path may differ; right-only scores are reported); one seed per model; branch bits are one property, not the whole path.
**Next:** add to the paper's monitorability section (thread 1d).
