# Predictions written 2026-11-07 16:00 PST before any t1/t2/t3 output exists (HANDOFF_2026-11-07)

Step 1 (readable x used; thresholds readable = probe best-of >= .5 with shuffled <= .06, used = swap follows >= .5 or same-level mean acc <= .5):
- 1a SIM z_{L-1}..z_5: z_L..z_5 keep own ~1.00 on swap, mean ablation ~1.00, and the hide-upto condition (only idle passes visible) leaves chance: inert,
  as in gated. z_{L-1} partly used: swap follows somewhere .3-.9 rising with L, mirroring mean ablation (.08 L3 .. .72 L6). Holds v_{L-1}, not the answer.
- 1b plain single latents (same-level means, k = L-1): single swaps follow <= .05 at L3+, single mean ablations cost little (acc near clean), z_0 possibly more.
  So plain is mostly "readable/unreadable, not used" per latent although all six together are used (b29).
- 1c gated z_0 (start value): uncertain; passes can read the start value from the tail of the question, so z_0 may be ignored (follows < .5, mean acc near clean).
- Table: gated and SIM have readable+used for the step latents and readable-not-used for idle latents; plain has few or no single used latents.
  Logit lens: gated reads used latents; SIM and plain lens miss used latents (used, not readable by the lens).

Step 2 (SIM, all six latents swapped, test L1-L7): L2+ follows the donor near 1.0 (single z_0 swap already follows .82 at L2); same-value donor keeps own ~1.0;
same-level mean ablation to chance from L2. L1 is the open cell: no trained step at L1; guess follows donor partly (.3-.9) since z_0 carries v_0 and SIM's
z_{L-1} at L2 is .09 under mean ablation.

Step 3 (dynamics): plain contracts to near-zero step size after pass 2 (centred cos -> .99); gated and SIM keep large steps through the last rule-reading pass
(t <= L-2), then drop; direction cosine low (turning) during the rollout for gated/SIM, high or undefined (tiny steps) for plain after the fixed point.

## Deviation, 2026-11-07 17:12 PST, before compiling t1_table (plain L3-L5 log lines already seen)
"used" by mean ablation changed from "accuracy <= .5" to "<= .5 of the items answered right clean stay right": plain CODI's clean accuracy is
.52 / .33 / .14 at L3-L5, so the absolute rule marks every plain latent used regardless of the ablation. Same rule as before where clean ~1 (gated, SIM).

## Step 5 second check, 2026-11-07 17:34 PST, before the full t5 run (smoke k = 6 only seen: gold .975 on 40, as s1)
Handoff guess: at L7 with k = 7 the extra pass applies rule 7 and the answer applies it again: A = r7(r7(v_6)) = r7(v_7), "last_twice", expected the
majority at k = 7. k = 8: last_thrice. L6 control stays at gold for k = 7..9 (an idle pass exists).

## Round-4 deep dive (t6 prefix/hide swaps, t7 partial probes), 2026-11-07 19:13 PST, before any t6/t7 output
Seen already: b29 offline (all-six donors carry donor v_{L-1} approximately: L3 exact .37 / within 2 .82; right donors .47 / .91) and the b16
suffix-by-edit-position table (suffix from z_1 .97, z_2 .88, z_3 .56, z_4 .03 for edits one rule back).
t6 predictions: prefix_s (donor's first s latents, rest recomputed): follows rises with s and is near the 'all' rate by s = 2-3 (the two working
passes); later passes do not undo it. hide_ge_s: accuracy holds while passes 0..2 are visible (hide_ge_3 near clean), falls for hide_ge_1.
only_s: one pass alone (s = 1..3) keeps much of the accuracy since passes are near copies; only_0 and only_5 worse.
t7 predictions: at latent sites, v_{L-1} tens digit and within-2 readout well above exact and above the ones digit and parity; CODI-0 tail/colon
sites read v_{L-1} coarse info lower than CODI's latents; branch bits of early rules readable at the tail in both models.

## t8 blend and t9 attention flow, 2026-11-07 19:56 PST, before any t8/t9 output (b21 raw attention shares already seen:
start-last passes 1-5 put .14-.31 of non-sink attention on earlier latents, relay .02-.12)
t8: answers move gradually with a; in-between answers common at a = .3-.7; answer tracks r_L(round((1-a) v_rec + a v_don)) within 2 on a majority
at mid a; crossover near a = .5 (by t6 counts, 3 of 5 donor passes gives ~.45).
t9: norm-weighted share on earlier latents for start-last similar to raw (~.2-.3 at later passes), far above relay; gated and SIM (chains)
put more weight on the immediately previous latent (distance 1) than start-last, whose weight spreads over all earlier passes.

## t10 routes, 2026-11-07 20:21 PST, before any t10 output
Gated and SIM: noback keeps accuracy >= .95 at L2-L5 and noback_swap_j still follows the donor >= .9 (value travels through z_s).
Start-last: noback reproduces b21 (L2 ~.70, small net cost at L3+); swaps follow ~.00 with and without noback.
Note: the earlier proposed "cut the recurrence in start-last" test (fixed inputs for passes 2-5) was dropped before running: t6's suffix swaps
show a start-last pass exposes its input's value to the answer, so fixed inputs would hurt start-last too and not separate width from depth.

## t11 GSM8K routes (released zen-E/CODI-gpt2), 2026-11-07 21:03 PST, before any t11 output
Known from 20260927-004340: single z0/z2/z4 .20/.20/.36, prefix z0..z2 .54, all .84 (bf16/CUDA). Prediction: reproduce within ~.05 in fp32.
Route test: GSM8K looks more chain-like (alternating sites, prefix propagates), so nb (look-back cut) costs little accuracy and swaps still
propagate under nb (prefix_2 under nb stays near .5); even_pairs roughly additive with z4 heaviest (not pure vote counting).
Alternative (width-like): nb drops accuracy substantially and prefix_2 under nb falls toward the single-site values.

## t12 GSM8K route split and t13 GSM8K attention flow, 2026-11-07 21:25 PST, before any t12/t13 output (t11 full not yet seen; t11 smoke seen)
t12: nb_even costs much more than nb_odd (values, not mask damage). Route split: horizontal-chain hypothesis predicts kv_noans_s well above
0 for s = 0, 2 (later latents read the twin's value by attention and pass it on), input_s weaker than in RuleChain chains. Vertical-chain
alternative: input_s ~ both_s, kv_s ~ 0. Width-like alternative: neither isolated route moves the answer.
t13: GSM8K latents put more attention on earlier latents than start-last's .09-.27, concentrated on even (value) sites.

## t15 RuleChain route split, 2026-11-07 23:04 PST, before any t15 output
gated / sim: input_j ~ both_j ~ 1 for used sites j < L-1; kv_j ~ 0 except the site the answer reads (gated j = L-1, sim j = L-1), where
kv ~ both and kv_noans ~ 0 (answer reads it directly). startlast: single sites ~0 on every route; both_all ~ .9, kv_all ~ .9 (the answer reads
the pooled keys/values). relay: single ~0; both_all high; kv_all high. stab: everything ~0 (ignores latents).

## t16 grid need test and t17 grid route split, 2026-11-07 23:43 PST (commit 3574431 first said 00:05 in error), before any t16/t17 full output (scripts/necessity/)
Seen before writing: 8-item ProsQA smoke of each loader (CODI 6 lat 5/8, 0 lat 6/8; Coconut 8/8 at every k = 0..6; no-CoT 6/8; CoT 8/8).
Known: ProsQA CODI .816, 0 latents .820, latents one constant vector (20261103-021023); Oct whole-chain random swaps change 0% of
answers for grid Coconut and CODI on ProsQA / ProntoQA; GSM8K Coconut latents load-bearing (random swap), twin swap reads content.
Coconut paper (GPT-2): ProsQA no-CoT 76.7, CoT 77.5, Coconut 97.0, iCoT 98.2 (verify); GSM8K no-CoT 16.5, Coconut 34.1.
Definition used: need = Coconut/CODI(6 latents) - same model at 0 latents, per path-length bin (the latents' contribution);
gain over the grid no-CoT is reported beside it but credits training, not latent passes. "Beyond reach" = bins where no-CoT and the
0-latent model are clearly below the 6-latent model.
t16 (need):
- ProsQA CODI: 6 vs 0 latents within .03 overall and in bins 3, 4, 5 (bin 6 has n = 12, report only). ProntoQA CODI: within .03.
- ProsQA Coconut: every k = 0..6 within .03 of k = 6 (~.97-.98) in bins 3-5; grid no-CoT ~.75-.82, so the Coconut-vs-no-CoT gap is
  real but comes from training, not from latent passes (need ~ 0). ProntoQA Coconut: same (k = 0 ~ k = 6).
- GSM8K Coconut (needs a run with gold-step bins): 6 latents - 0 latents >= .10, growing with gold step count; no-CoT < Coconut.
t17 (route split, minimal pairs = one path relation rewired so the answer flips, equal token length; recipients right, donor answered
as implied):
- ProsQA / ProntoQA, CODI and Coconut: every site, every route (both, input only, entries only, hidden) follows <= .03; all-sites
  both / entries <= .03. All four FAIL the check (ignored regime, consistent with need ~ 0).
- GSM8K Coconut (separate run): framing predicts it must pass at >= 1 site if its need is beyond single-pass reach. Guess: passes at
  an early site (input only >= half the full swap at j = 0 or 1, full swap >= .10); later sites carried by entries. Alternative
  (would count against the framing only if the need grows with step count where no-CoT is at the floor): input only ~ 0 everywhere,
  i.e. a width or attention-slot regime like GSM8K CODI.

## t18 ProsQA probes (scripts/necessity/grid_probe.py), 2026-11-07 23:47 PST, before any t18 output and before any t16/t17 full output
Probes: linear, 3000 train items, 500 test; path_k = k-th path node (argmax among the item's graph nodes, chance ~.05), front_k = BFS
frontier at distance k (AUROC over graph nodes, chance .5); positions tail (last question token), bot (<bot> / <start-latent>),
latent passes 0-5; layers 6 / 9 / 12; shuffled-label controls; plus the Coconut paper's LM-head readout (frontier-k mass at latent j).
- CODI (latents ~one constant vector): latent probes no better than bot (within .05 acc / .03 AUROC) at every k; tail / bot read the
  hop-1 node and frontier 1 well above chance (path_1 >= .3, front_1 AUROC >= .75); shuffled controls at chance.
- Coconut (latents ignored by the answer, t17 predicted <= .03): frontier information readable at latents above chance (front_1 AUROC
  >= .7) but not more than at <start-latent> (within .03), and no diagonal: the LM-head frontier-k mass at latent j does not peak at
  j = k (i.e. the paper's BFS picture does not hold for this checkpoint, or holds only as decoration). If it does peak at j = k while
  t17 shows every latent off-route, that is the "accurate but decorative decoder" case of step 7.

## t19 RuleChain monitor test (scripts/necessity/rc_monitor.py), 2026-11-07 23:49 PST, before any t19 output
Decoders per site j from the input z_j or the stored entries (keys/values, layers 3/6/9/12), trained on train questions; monitor answer =
decoded value pushed through the remaining rules. Tested on t15's items and both_j donors. On-route = t15 input-only follow >= .5.
- gated / SIM: decoders accurate on clean right items at rule sites (>= .9, both locations). Swap at an on-route site j: decoders at
  later sites read the donor-implied value (>= .9) and the monitor answer matches the actual answer (>= .9): faithful.
- gated / SIM spare sites (j >= L, off-route): clean accuracy high (copies of v_{L-1}), but after a swap there the input decoder reads
  the donor value while the answer stays: monitor agreement with the actual answer <= .1 (decorative).
- start-last / relay (no on-route site): clean decode accuracy moderate to high somewhere, but swap_j -> j monitor agreement <= .3 at
  every site and own-error prediction (monitor answer = the model's wrong answer) <= .3 at every site: decorative everywhere.
- gated / SIM own-error rows have few wrong items (clean ~1.0); reported with n, not judged.

## t20 teacher-side dependence (scripts/necessity/teacher_dependence.py), 2026-11-07 23:52 PST, before any t20 output
Teacher = the CODI model reading question + CoT (last step dropped) + "The answer is: gold"; state at ':' (distillation target).
One shown CoT value at a time replaced positionally (same token length); k counted from the end of the shown CoT (1 = last shown).
- RuleChain (start-last, relay, stab): flip >= .8 at k = 1, <= .1 at k >= 2 (the teacher reads the last shown value and applies the
  last rule; earlier values are not re-read); state move at k >= 2 <= a third of k = 1's.
- GSM8K (zen-E): dependence spread: flip at k = 1 highest, k = 2 still >= .2 (the dropped last step often combines several results);
  k >= 3 lower but > 0.
- ProsQA (grid CODI): preferred option flips mostly at k = 1 (last shown node), less at k >= 2; state moves at every k because the
  node names are copied tokens.
Reading: if RuleChain's teacher depends only on v_{L-1}, latents distilled to it need only carry v_{L-1}, which fits the relay /
pooled-estimate regimes; GSM8K's spread dependence fits several value slots.

## t21 SIM-CODI start-first: route split (t21a), monitor test (t21b), prefill probe (t21c), 2026-11-08 10:40 PST, before any t21 full output
Model simsf = SIM-CODI start-first checkpoint-17820 (20261108-130905). Seen before writing: its s1/s2/s3 battery (20261108-134158: swap
z_1..z_{L-2} follows .974, z_0 .95-1.00, mean ablation of the latent after the last trained step only to .55-.81, probe v_k at z_{k-1} .52,
v_k at z_k .82, 13 wrong valid items at L3-L6 of which 9 match a decoded operand value + remaining rules, post hoc); matched start-first
CODI (20261108-135708); t15/t19 for start-last SIM; a 300-item L3 smoke of t21c (simsf v1 at rule 1 .23, start-last SIM .03; not read further).
t21a (t15_rc_route_split.py, MODEL=simsf, LEVELS=3,4,5,6,7):
- Passes the check like start-last SIM: input only >= .9 at every trained site j = 1..L-2 (entries only <= .05 there). z_0 input only
  >= .8 (start-last SIM .60-.71; here the full swap is .95-1.00).
- Site L-1 (the latent the answer reads): entries only >= .8 and entries hidden from the answer <= .1. Weaker than start-last
  (.98) is possible because its mean ablation hurt less; if entries only < .8 there, the answer has a second source for v_{L-1}.
- All sites: both ~1.00, entries ~1.00.
t21b (rc_monitor.py simsf, valid L3-L7, decoders trained on L2-L7):
- On-route sites (t21a input only >= .5): clean decode accuracy entries >= .8, input >= .6 (below start-last's .84-.99); after a swap the
  monitor answer matches the actual answer >= .85. Spare sites: decorative (<= .1), as t19.
- Own errors (the first forced model with errors; ~30 wrong valid items at L3-L7): at the best on-route site the monitor answer equals
  the model's wrong answer >= .3, against <= .1 for a decoder fit to shuffled labels / random values. Reported with n.
t21c (prefill_probe.py; simsf, stab = start-first CODI, sim = start-last SIM, startlast = start-last CODI; valid L3-L6):
- Start-last models: v_k at the end of rule k <= .10 for every k (start value not yet seen: the floor). At <bot>, v_1 >= .5 for sim.
- stab: the question pass runs the chain: v_1 at rule 1 >= .7, v_2 at rule 2 >= .5, falling with k; v_{L-1} at <bot> tracks its
  answer accuracy within .2 at L3-L4.
- simsf, two readings fixed now: (dormant copy) diagonal within .15 of stab at k = 2, 3: the question pass still runs the chain, the
  answer ignores it (its latent ablation goes to chance), and a latent-only monitor misses a parallel computation; (moved) diagonal at
  k >= 2 at least .2 below stab: the step loss took the chain out of the question pass. Guess: moved, with v_1 at rule 1 still >= .5.

## t22 released SIM-CODI GPT-2 on GSM8K (scripts/necessity/sim_gsm8k.py), 2026-11-08, before any SIM-CODI model output
Model: internlm/SIM_COT-GPT2-CODI (HF, MIT; CODI stack + training-only step decoder, 148 decoder tensors; trainer_state shows a fourth
loss term 10.5 at step 0 decaying to ~0, so the step loss was trained). SIM-CoT's get_steps: z_j (j = 0..6) decodes shown step j+1; the
teacher drops the last step (include_last_cot False), so on an n-step item z_0..z_{n-2} have step targets and z_{n-1}.. none (as our
SIM-CODI). Test flags = released CODI's (zen-E fd641b3), which is the matched reference (same stack, data, 40 epochs). GSM8K test, 1319.
Nothing from this checkpoint has been run or decoded before writing this; its README accuracies are placeholders.
t22a accuracy and pass sweep (answer after k = 0..6 latent passes), both models, by gold step count n:
- Full accuracy: SIM within +-.04 of released CODI (.419 in 20260927-004340).
- CODI reproduces its logged sweep (settling pass ~1.8 / 2.9 / 3.7 / 4.3 at n = 2-5).
- SIM chain reading: on right items, accuracy at k = n-2 passes is at least .2 below k = 6 for n = 3-5 (each step needs its pass).
  Width/slot reading: SIM's sweep looks like CODI's.
t22b route split (t12 design, ALL sites 0-5 for SIM; pairs = 20260927-004340 candidates where SIM answers recipient and twin right,
same question token length): conditions both / input / kv / kv_noans per site, plus all sites.
- Decision rule fixed now: a site is on the vertical route if input >= .5 x both and both >= .2; slots if kv >= .5 x both and input <= .2 x both.
- Guess: SIM is a chain at the early sites (z_0, z_1: input >= .5 x both), unlike released CODI (input .007 everywhere). Confidence low:
  GSM8K steps combine earlier results, so slots (attention reads of earlier latents) are also plausible.
- All-sites swap follows the twin >= .8 (released CODI .839).
t22c monitor with SIM's own decoder (greedy decode of each z_j into '<<a op b=c>>'):
- Clean, right items: decoded step k from z_{k-1} has the gold result c_k for k <= n-1: >= .6 at k = 1-2.
- Own consistency: on right items the answer equals the last decoded step's result pushed through... (not computable: the last step is
  not decoded). Instead: on wrong items, the decoded intermediate results differ from gold at some step on >= .5 of items (errors
  visible in the trace), against <= .2 on right items.
- Swap faithfulness: after a both_j swap that moves the answer to the twin's, the decoded result at later sites matches the twin's
  gold intermediate >= .6.
t22d the step decoder as an external monitor on released CODI (decoder weights applied to CODI's latents): reads <= .2 (different
latent code; a control that the decoder is not just reading the question).

## t23 GSM8K spare latents, readable but unused? (offline on 20260927-004340 single_slot + lens from sim_gsm8k.py), 2026-11-08, before output
Released CODI, recipients with n = 2 gold steps among the 317 aligned pairs: lens at pass 5 reads v_1 ~.59 (audit, n = 2). Prediction:
single-site swap at z_5 follows the twin <= .10 and at z_4 <= .15, while the lens reads the recipient's v_1 at pass 5 >= .4 on the same
items: readable, not used (natural-data analogue of the RuleChain spare latents). For n >= 3 recipients z_4 follows more (overall .358).
SIM: the same split for n = 2 with SIM-qualified pairs: z_2..z_5 (no step target) follow <= .10.

## t24 same-level mean ablation, round-1 and lr-3e-3 CODI (RuleChain), 2026-11-08, before output
Replaces the level-pooled means behind "round-1 .99 -> .17 at L3". All six latents set to their same-level train mean (2,000 items/level,
seed 21), test, L1-L5 (round 1: depth2 test; lr-3e-3: l1to7w test). Prediction: L1 stays >= .95 (as start-last, where pooling caused the
L1 drop); L2 falls to <= .7; L3 <= .3; L4-L5 <= .15 for both models, i.e. the direction of the pooled result holds from L2, and the
matched CODI-0 (.988 / .960 at L2 / L3, round 1) stays far above: reliance beyond need.
