## 2026-11-08 — ProsQA Coconut probes: no BFS diagonal; all six latents read the same (frontier-1 AUROC .68-.70 at every latent, below <start-latent> .73), and the LM-head readout puts the same frontier-3/4 mass on every latent (.30-.32 / .43-.45 vs base .15 / .06); deep-path information readable at the latents but the answer ignores them (coconut, run_id: 20261108-102225_coconut_prosqa-grid-path-frontier-probes)

**Goal:** Step 4d: test the Coconut paper's BFS claim directly (latent k ~ frontier k) on the grid checkpoint, and step 3's "path node at the question tail vs latents". Pre-registered (t18, committed 0e3d7cc): latent probes not above <start-latent>; no diagonal; if a diagonal appeared with t17 off-route latents, it is the decorative-decoder case.
**Mechanism / model:** `connordilgren/gpt2-prosqa-coconut` (checkpoint_40), 6 latents.
**Data:** probes trained on 3000 ProsQA train items, tested on all 500 test items. Targets: path_k = k-th node on the gold path (k = 1..3), front_k = BFS frontier at distance k from the person (k = 1..4). ProsQA has 55 concept names in total.
**Hyperparams / command:** `grid_probe.py coconut mps <out> 3000`; linear probes, standardised features, shuffled-label controls; raw `codi_audit_artifacts/t18_coconut.json`.
**Results (layer 12; shuffled controls .03-.08 acc, .49-.56 AUROC):**

| position | path_1 | path_2 | path_3 | front_1 | front_2 | front_3 | front_4 |
|---|---|---|---|---|---|---|---|
| question tail | .23 | .22 | .37 | .74 | .72 | .71 | .77 |
| <start-latent> | .25 | .22 | .40 | .73 | .72 | .73 | .79 |
| latents 0-5 (range) | .16-.19 | .16-.18 | .44-.45 | .68-.69 | .67 | .68-.70 | .77 |

Layer 6: question tail reads frontier 1 best (AUROC .89, path_1 .32); the latents read deeper nodes better than <start-latent> (path_3 .27-.34 vs .07; front_3 .66-.71 vs .55; front_4 .74-.76 vs .63). LM-head readout (the paper's method) at latent j: frontier-k mass .03 / .06-.07 / .30-.32 / .43-.45 for k = 1..4 (base .13 / .19 / .15 / .06), identical for j = 0..5.
**Interpretation:** No BFS picture: the latents are near copies of each other (same probe scores, same lens mass at every j), no latent k is specific to frontier k. The lens mass sits on frontier 3-4, where the answer options usually are, so it reads answer candidates, not a search frontier. Prediction: no diagonal (held); latents not above <start-latent> (held at layer 12, NOT at layer 6, where latents read path_3 / deep frontiers better than <start-latent>). Combined with t17 (route split .00) and t16 (0 = 6 passes), whatever the latents encode is decorative.
**Caveats:** linear probes; first-token lens match (concept names are multi-token); path_k pooled over path lengths.
**Next:** none planned for this checkpoint; Coconut-paper-style BFS analysis needs a checkpoint whose latents are used.
