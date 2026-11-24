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

## t25 latent monitor for wrong answers, released SIM-CODI GPT-2 on GSM8K (scripts/necessity/sim_monitor.py), 2026-11-08, before computing any t25 number
Task: flag the model's wrong answers on GSM8K test (1319; SIM wrong on 800). Monitors use only what a deployed monitor sees (question,
answer, latents), never gold. 5-fold cross-validated logistic regression, AUROC for "answer is wrong".
- Latent monitor (L): features from SIM's own decoder on z_0..z_6: share of decoded steps whose arithmetic is self-consistent
  (a op b = c holds), share of decoded operands found in the question or in earlier decoded results, share of z_j that reuse z_{j-1}'s
  result, number of distinct decoded steps, repeats, whether the answer equals a decoded result.
- Baselines: (C) answer confidence (mean and min token log-probability of the generated answer, first-token probability);
  (Q) question only (length, count of numbers); (C+Q).
- Combined (L+C+Q).
- Not a monitor, reported as a ceiling: the gold-based "trace has a wrong intermediate" check (t22: .85 of wrong vs .29 of right).
Predictions: L >= .65; C ~ .70 (the strongest single baseline); L+C+Q above C+Q by >= .03 (the latents add information a monitor
could use). Control: the same latent features from SIM's decoder applied to released CODI's latents predict CODI's errors no better
than its Q baseline (within .03). If L+C+Q - C+Q < .03, the readable trace adds nothing a confidence monitor lacks.

## t26 hop count (hop-limited recomputation), scripts/necessity/hops.py, 2026-11-09 21:00 PST, before any t26 output
Question: how many recurrent edges (final layer of a pass -> input of the next pass, incl. question -> z_0) does the information that
changes the answer cross? Attention moves information between positions at the same layer and adds no layers, so serial depth beyond
one pass = recurrent hops. The registered input-only check (t15/t17/t22) misses hops linked through attention; this counts them.
Method. Minimal pair: recipient R, donor D = R with one number edited (RuleChain: the start value or one number of rule i, same digit
count, chosen so v_i changes, all values in 0..99 and the answer changes; GSM8K: the 1471 twin candidates of t22, one question number
changed). Only z vectors move between runs, never keys/values, so equal token length is not needed.
  M_0 = D's question with every latent input z_s set to R's natural z_s (donor info can reach the answer only with 0 recurrent hops).
  M_n = D's question with z_s set to the natural z_s produced inside M_{n-1} (<= n hops). M_6 = D's own run (checked exactly).
  F_n = share of qualified pairs (R right, D right) whose M_n answer = D's answer; also share = R's answer, and neither.
  hops*(pair) = smallest n with the answer = D's from n on. Grouped by s = rule steps from the edit to the answer (start: L; rule i:
  L - i + 1; GSM8K: chain_len - step).
Models: RuleChain valid L2-L6: gated, sim (start-last), startlast, relay (lr-3e-3, start-first), stab, simsf (SIM start-first);
GSM8K test twins: released CODI (zen-E), released SIM-CODI, grid Coconut (connordilgren). Cap 80 recipients per (level, edit) cell.
Predictions (thresholds: "uses n hops" = F_n >= .9 and F_{n-1} < .9):
- gated: hops* rises with s: start edit L-1 hops, rule i edit L-1-i (rule L: 0), on >= .8 of qualified pairs within +-1. F_0 <= .1 for
  s >= 3.
- sim (start-last): hops* rises with s, within +-1 of L-i for rule i (start and rule 1: L-1); F_0 <= .1 for s >= 3.
- simsf: rises with s for start / early-rule edits (t21: one step per pass), but less than sim because prefill can run part of the chain.
- startlast: no rise with s. F_1 >= .9 in every s bin with n >= 20; F_2 - F_1 <= .05. F_0 >= .4 at L2-L3 (pause control: most of the gain
  needs no recurrence). Mean hops* <= 1.2 at every level.
- relay: hops* <= 1 on >= .9 of pairs (one value computed in the question pass, copied); no rise with s.
- stab: F_0 >= .9 in every bin (latents ignored).
- GSM8K CODI and Coconut: F_1 >= .9 overall; F_2 - F_1 <= .05; no rise of hops* with s (slots filled from the question, not read by later
  slots). Low confidence: Dilgren's forward-chaining reading predicts F_2 - F_1 > .05 at s >= 3. If so, report attention-linked depth.
- GSM8K SIM: same as GSM8K CODI (t22: not a recurrent chain), low confidence.
Necessity rule, fixed now: single-pass reach r = largest level with matched no-latent accuracy >= .5 (start-last CODI-0: r = 2;
start-first stabilised CODI-0: r = 5). A violation = a model with hops* = 0 on >= .2 of qualified pairs with s > r (correct answers that
track an edit needing more steps than one pass, without crossing a recurrent edge). Predicted: no violation in any model.
Checks: M_6 answers = D's clean answers on every pair; feeding a run its own natural z changes no answer (smoke).
t26 amendment, 2026-11-09 21:00 PST, after the gated smoke only (L3, n = 14): the gated formula above ignores protocol 12.1's correction
(the pass reading rule L-1 needs one more pass before the answer sees it). The derivation then gives L-i hops for rule i (i <= L-1),
L-1 for the start value, 0 for rule L. The registered prediction (L-1-i) is kept and scored as written; this line only records the error.

## t27 follow-ups to t26 (post hoc: designed after reading t26), 2026-11-09 22:05 PST, before any t27 output
t27a longer GSM8K edits (scripts/necessity/gsm8k_long_twins.py, 1305 new twins with s >= 4, disjoint from t22's; hops.py with CANDS).
  Question: does the GSM8K staircase (t26: mean hops 1.3 / 2.0 / 2.2-2.5 at s = 2 / 3 / 4) keep rising or level off?
  Prediction: it levels off: mean hops at s = 5-6 within .5 of s = 4, and below s - 1.5, for CODI, SIM and Coconut. Low confidence.
t27b item-level necessity (scripts/necessity/hops_ref.py): matched CODI-0 (start-last CODI-0 for gated / sim / startlast; stabilised
  CODI-0 for relay / stab / simsf) answers each qualified t26 donor question. Corrected rule (replaces t26's population threshold, which
  flagged stab at s = 6): violation = hops* = 0 on >= .2 of qualified pairs whose donor CODI-0 answers wrong (the edited item is beyond
  one no-latent pass for this recipe), in cells with n >= 20. Prediction: no violation in any model; stab's s = 6 flag disappears
  because its zero-hop pairs are ones CODI-0 also solves. The t26 rule's result is still reported as registered.
t27c memorisation (hops_ref.py, 500 train items per level vs valid): stabilised CODI and CODI-0, start-last CODI and CODI-0.
  Prediction: train - valid gap <= .10 at every level for all four (12 epochs over 190k items, unlike round 1's 40 over 100k).

## t28 SIM-CODI on GSM8K: steps in parallel or chained? (scripts/necessity/sim_parallel.py), 2026-11-09 23:05 PST, before any t28 output
Released SIM-CODI, SIM's qualified t26 + t27a GSM8K pairs (A recipient, B donor). B's changed steps split into direct (uses the
edited number, no earlier changed step) and dependent (uses an earlier changed step's result). SIM's own decoder on every latent input
z_0..z_5; a step is read if ANY latent decodes to B's value (no assumption about which latent holds which step). Conditions: refresh 0
(A's latents: floor), refresh 1 (<= 1 loop crossing), refresh 2 (<= 2), B's own run (ceiling).
Hypothesis (from t22 / t26): SIM's step labels yield readable side outputs computed from the question, not a chain.
Prediction: dependent steps read at refresh 1 >= .7 x their rate in B's own run (parallel). Refuted (chain) if refresh 1 <= .4 x own
and refresh 2 >= .7 x own. Direct steps: refresh 1 >= .8 x own under both readings. Floor: refresh 0 <= .1 for both kinds.
Low confidence: t26 shows SIM's ANSWER needs ~2 hops at depth 3-4, which a parallel trace would not explain on its own.

## t28b SIM-CODI readability by crossings needed (sim_parallel.py, all refreshes 0-6, exact dependency depth), 2026-11-09 23:40 PST, after t28, before any t28b output
Post hoc extension of t28 for a figure. Each changed step of B gets depth d = 1 (uses only the edited number) or 1 + the largest depth of
the changed steps it uses: the loop crossings a chain needs before its value can sit in a latent input. Readable at refresh n = any latent
decodes to B's value. Prediction, among steps readable in B's own run: for d = 1-3 (cells with n >= 20), readable share at refresh d-1
<= .3 and at refresh d >= .8 (a switch exactly when the crossing becomes available); refresh 6 latents equal B's own exactly.
Readable in own run falls with d (t28: .85 / .63 / .34 by chain position).
t28b negative controls, added 2026-11-09 23:50 PST, before any t28b output (computed offline from the t28b decodes):
- Chance: the same decoded latents scored against B's values of a different pair (same depth, pair order shifted by one). Prediction:
  chance-readable <= .10 at every refresh and depth, and below the real rate at refresh >= d by >= .5.
- Stale value: at refresh d-1, among steps readable in B's own run, the share where some latent decodes A's value for that step.
  Prediction: >= .5 (the latent holds the chain's outdated value), falling to <= .15 at refresh d.

## t28c negative-model control: released CODI's latents through SIM's decoder (sim_parallel.py, LATENT=codi), 2026-11-09 23:55 PST, before any t28c output
Same design as t28b on CODI's qualified t26 + t27a pairs. CODI also uses ~2 hops (t26), but t22d found SIM's decoder reads CODI's
latents at <= .2. Prediction: readable in CODI's own run <= .25 for depth 1 and <= .15 for depth >= 2 (same depth, not legible to a
decoder trained on another model); refresh 6 latents equal CODI's own exactly.

## t29 RuleChain readability by loop crossings (scripts/necessity/rc_readability.py), 2026-11-10 00:20 PST, before any t29 output
Probe version of t28b on the qualified t26 RuleChain pairs, six models. Linear probes per (level, latent site, value index) on 800 train
items, validated on 200; a site is legible if its probe validates >= .5; a value is read if ANY legible site predicts it. d = crossings
a chain needs (edit at rule i, value after rule k: k - max(i,1) + 1). Controls: stale A value, chance (another pair's value).
Predictions (among values read in B's own run, d cells with >= 20 such values):
- gated, sim, simsf: switch at refresh d or d+1, never early: read at refresh d-1 <= .3, at refresh d+1 >= .8; stale A >= .5 at d-1.
- startlast, relay: no staircase: read at refresh 2 >= .8 for every d (one value handed over, recomputed rather than chained).
- stab: legible sites rare; any readable values are read at refresh <= 1 (computed in the question pass).
- chance <= .10 everywhere.

## t30 GSM8K readability by vocabulary projection (scripts/necessity/gsm8k_lens.py), released CODI, Coconut, SIM-CODI, 2026-11-10 00:20 PST, before any t30 output
t28b's design with the readout replaced by top-10 tokens of each latent under the output head (single-token values only).
Predictions: read in own run at depth 1 >= .3 for all three (Dilgren & Wiegreffe find traces by vocabulary projection), lower at depth
>= 2. Among values read in the own run: switch at refresh d for d <= 2 (read at refresh d-1 <= .3, at refresh d >= .8), with stale A
>= .4 at refresh d-1, for CODI and Coconut as for SIM's decoder in t28b. Low confidence: the lens may read values that sit in a latent's
input but are not chained. chance <= .10.

## Scoring t26-t30, 2026-11-11 (after all outputs; numbers from codi_audit_artifacts/t26, run records of 2026-11-11)
t26 hop count (thresholds as registered):
- gated: held. hops* within +-1 of the registered L-1-i on 1.00 of 1704 pairs (the amended L-i fits exactly); F_0 = 0 at s >= 3.
- sim: held. Within +-1 of L-i on 1.00 of 1655; F_0 = 0 at s >= 3.
- simsf: rise with s held (mean .62 / 1.99 / 3.00 / 4.00 / 5.01 / 5.98 at s = 1-6); "less than sim" missed (equal to sim at s >= 2).
- startlast: missed. F_1 .97 / .91 / .60 / .79 at s = 1-4 (not >= .9 everywhere); F_0 at L2-L3 .29 (not >= .4); mean hops by level
  .68 / 1.19 / 1.03 / 1.07 / 1.27 (L6 above 1.2). No rise with s held (1.14 / 1.52 / 1.26 / 1.56 at s = 2-5).
- relay: missed narrowly. hops* <= 1 on .86 (not >= .9); mean hops rises mildly with level (.68 -> 1.35).
- stab: missed. F_0 .95 / .94 / .89 / .85 / .83 / .76 at s = 1-6 (not >= .9 from s = 3); F_1 >= .98 everywhere (a small one-hop share).
- GSM8K CODI / SIM / Coconut: missed. F_1 = .61 / .55 / .56 (not >= .9), F_2 = .95 / .89 / .97, mean hops rises with s (CODI 1.26 / 1.98 /
  2.22 at s = 2-4). The low-confidence alternative (forward chaining across slots, F_2 - F_1 > .05) held.
- Necessity rule (population threshold): missed. stab has hops* = 0 on .76 of s = 6 pairs (s > r = 5). Replaced in t27b.
- Checks: M_6 = donor run on every pair run, all models; own write-back changes no answer (smokes).
t27a: held for CODI (s = 4 / 5: 1.87 / 1.75) and SIM (2.05 / 1.89), both below s - 1.5. Coconut 1.56 / .75 (s = 5 n = 8): drop of .81
  misses "within .5". Dependency-depth rebinning (post hoc): depth 4 means CODI 2.18 (n = 61), SIM 2.30 (57), Coconut 1.68 (19).
t27b: held for start-last models (zero-hop share 0 / 0 / .012 for gated / sim / startlast on pairs with s >= 2 whose donor CODI-0 gets
  wrong); missed for start-first stab (.76 zero-hop on CODI-0-hard donors, s >= 2; .80 with the trivial s = 1 rows). Single-pass reach is model-specific in start-first data, so the item-level rule is not identifiable
  there. s = 1 rows are trivial and excluded.
t27c: held except stabilised CODI at L7 (train - valid .13).
t28: chain, parallel refuted as registered. Dependent steps: refresh 1 .12 = .25 x own (.49), refresh 2 .43 = .89 x own. Direct steps:
  refresh 1 .83 = .97 x own (held). Floor <= .1: dependent .07 held, direct .11 missed narrowly.
t28b: switch at refresh d held for d = 1, 2 (refresh d-1 .12 / .20, refresh d .96 / .96); missed for d = 3 (refresh 2 already .66;
  refresh 3 .99). Own-readable falls with d (.85 / .62 / .32) held; refresh 6 = own on 392 / 392 held. Chance <= .10 held (pooled over
  depths, on the own-readable items, max .07). Stale A at d-1 >= .5: held for d = 1, 2 (.95 / .76), missed for d = 3 (.24, the early
  switch); <= .15 at refresh d held.
t28c: held. CODI's own-run readable share through SIM's decoder .24 (d = 1), .10 / .08 / .07 (d = 2-4); refresh 6 = own on 417 / 417.
t29 (among values read in B's own run, cells n >= 20):
- gated, sim, simsf: held. Read at refresh d-1 <= .07, at d+1 1.00; stale A at d-1 .98-.99 / .78-.97 / .74-.89; chance <= .03.
  Post hoc split by edit type (offline, 2026-11-11): gated reads 1.00 at refresh d on rule edits and .20 on start edits (one more
  crossing than d assumes); sim / simsf read ~.70 at d for every edit type (their chain runs one crossing later, as in their hop counts).
- relay: held. Read at refresh 1 .93-.99 for d = 1-4 (one value, one crossing).
- startlast: not testable beyond d = 1. Only v_1 at L2 is linearly legible (read at refresh 1, 1.00); no legible site for d >= 2.
- stab: held. No legible site at any level.
t30 (vocabulary projection, top 10):
- Own-run readable at depth 1 >= .3: CODI .37 held, SIM .25 missed, Coconut .95 but see chance.
- Switch at refresh d for d <= 2: CODI d = 1 held (.11 -> .99), d = 2 missed narrowly at d-1 (.37 > .3; .94 at d); SIM held (.07 -> .87,
  .16 -> .85). Stale A >= .4 at d-1: CODI .68 / .46, SIM .48 / .49 held.
- Chance <= .10: CODI .03, SIM .03 held; Coconut .27-.34 missed, so Coconut's vocabulary-projection results are not interpretable
  (its top tokens are dominated by small numbers).

## t31-t33 follow-ups to t26 (post hoc: designed after reading t26-t30), 2026-11-11 23:30 PST, before any t31-t33 output (smokes only, n <= 16)
Script: scripts/necessity/hops.py with QUAL=own (t31), FIRST=1 (t32), model coconut_rc (t33). One run per model covers t31 and t32;
the both-right subset of a QUAL=own run is t26's gold-qualified set.
t31 own-answer qualification. A pair counts if the model's own answers to R and D differ (both parse), whether or not they are right;
  hops = first n from which the answer is the model's own D answer. Removes t26's selection on correct items.
  - Both-right subset reproduces t26 mean hops within +-.10 in every s bin with n >= 20 (sanity).
  - startlast: any-wrong pairs reach n >= 40 in each of s = 4, 5, 6; their mean hops <= 1.6 in each (it fails deep items because it
    crosses about once). Low confidence alternative: wrong deep answers use more hops than right ones.
  - relay: any-wrong mean hops within +-.30 of both-right in each s bin with n >= 20.
  - stab: F_0 >= .80 on any-wrong pairs in every s bin.
  - gated / sim / simsf: too few wrong pairs to score (expected n < 20 per bin); all-pairs means within +-.15 of t26.
  - GSM8K CODI / SIM / Coconut: any-wrong mean hops by s within +-.30 of both-right at s = 2, 3 (low confidence).
t32 first answer token (GSM8K, FIRST=1): first token teacher-forced after <eot> 'The answer is:' (CODI, SIM) or <end-latent> '###'
  (Coconut), on a copy of the cache. Pairs whose two natural first tokens differ.
  - Agreement (teacher-forced token starts the greedy answer) >= .95 on natural runs.
  - hops_first = full-answer hops on >= .90 of pairs; mean hops_first by s within +-.20 of full-answer hops. A larger gap means part of
    the depth used is carried by generated answer tokens.
t33 Coconut on RuleChain (our c1m6, start first, test L2-L6; latents at or below no_thoughts per RULECHAIN_RESULTS): F_0 >= .90 in
  every s bin (latents unused, so hops ~0), all pairs (QUAL=own; the model is right on few deep items).

## t34 hidden-path detection (scripts/necessity/hidden_path.py), 2026-11-12 00:15 PST, after a smoke (gated, L4, 300 fit / 200 test) only
Question: can a monitor reading the latents tell a property of the reasoning path that a monitor seeing only the question pass and the
answer cannot? Target: branch taken at middle rule k (k = 2 .. L-1), RuleChain valid L4-L6, gold walk labels, balanced accuracy.
Baseline QA = question pass at <bot> (layers 4/8/12) + own answer one-hot; latent sets Z (z_1..z_5) and H (passes 0-4, layer 8).
Ridge to +-1, 2,000 fit items per level. Models: gated sim simsf startlast relay stab.
Predictions (gain = QAZ or QAH, whichever is higher, minus QA; averaged over k and levels unless stated):
- gated, sim (start-last chains): QA <= .65 for every k (the start value comes last, so <bot> cannot hold a middle branch); best latent
  set >= .85 for every k; gain >= .20.
- simsf: QA <= .70 (t21c: its question pass computes only v_1); gain >= .15.
- stab (start-first, chain partly in the question pass, latents ignored): QA >= .65 at k = 2; gain <= .05.
- startlast: gain <= .10 (its latents hold v_{L-1} as a magnitude, not branch bits; t-partial probes); low confidence.
- relay: gain <= .10 except possibly at one rule; low confidence.
- Shuffled control on QAZ: <= .56 everywhere. Right-only scores within .05 of all-item scores for gated / sim / simsf.

## t35 layer-wise single-pass reach (scripts/necessity/prefill_probe.py with LAYERS=0..12), 2026-11-12 00:15 PST, after a smoke only
Question (1b): how many RuleChain steps does one 12-layer pass compute? Same design as t21c, at every layer. Models: stab0, startlast0
(matched CODI-0), stab, startlast, simsf, sim. Emergence layer of v_k = first layer at which the best position (any rule end >= k, or
<bot>) reads v_k exact >= .5.
Predictions (low confidence):
- stab0 (start first): v_1 emerges by layer 6; v_2 emerges (by layer 12) at least 2 layers after v_1; v_3 never reaches .5. About
  two steps per pass.
- startlast0 (start last): only <bot> can hold values; v_1 emerges by layer 12; v_2 never reaches .5.
- stab reads v_2 and v_3 at least as well as stab0 at layer 12 (t27b: latent training extended its single pass).
- sim / simsf / startlast: as t21c (no chain past v_1 in the question pass) at every layer.

## Scoring t31-t33, 2026-11-12 04:05 PST (after all outputs; numbers from codi_audit_artifacts/t31)
t31 (QUAL=own):
- Both-right reproduces t26 within .10: held for gated / sim / simsf (max diff .005 / .007 / .021), startlast (.09) and all six GSM8K runs
  (<= .06); missed for relay (.11) and stab (.20, s = 6, n = 22). RuleChain both-right bins are a subsample of t26's (t31 samples pairs
  before checking correctness, e.g. stab s = 6 n 22 vs 58), so the misses are small-bin noise, not a code difference.
- startlast: n >= 40 held (254 / 180 / 121 at s = 4 / 5 / 6). Mean hops <= 1.6 missed (1.80 / 1.86 / 1.91). The low-confidence
  alternative held: wrong answers use more hops than right ones (all-s 1.47 vs .79).
- relay: within +-.30 missed at s = 2-3 (wrong answers +.3 to +.6 hops); still far below s.
- stab: F_0 >= .80 on any-wrong missed at s = 5-6 (.74 / .75); held at s <= 4.
- gated / sim / simsf: "too few wrong pairs" missed (wrong n 46-104 at s <= 4). All-pairs within +-.15 of t26 held (max .037 / .010 /
  .112). Wrong pairs keep the staircase (gated s - 1, sim / simsf about s).
- GSM8K any-wrong within +-.30 of both-right at s = 2, 3: CODI held (+.20 / -.30, at the edge); SIM missed at s = 2 (+.39), held at
  s = 3 (-.03); Coconut held at s = 2 (+.03), missed at s = 3 (-.34). Long twins (s >= 4): pairs grow from 98 / 89 / 42 to 1,092 /
  1,148 / 1,109; mean hops by s stay at 1.6-1.8 (CODI), 1.7-2.0 (SIM), 1.3-1.6 (Coconut) for s = 4-7. s = 4 three-hop share on
  any-wrong pairs .14 / .15 / .12 (both-right .22 / .19 / .06).
t32 (first answer token):
- Agreement >= .95: held, 1.00 for all six runs.
- hops_first = hops on >= .90 of pairs: held (.954-.970). Mean by s within +-.20: held in every bin with n >= 20 except SIM t22 s = 6
  (-.44, n 20), Coconut t22 s = 6 (-.21, n 23), CODI long s = 7 (-.23, n 49). The gap is negative everywhere (first token switches
  about .1 crossings earlier): answer tokens do not carry extra depth; later digits keep reading the stale latents a little longer.
t33: held. Coconut c1m6 F_0 >= .98 in every s bin, mean hops .01 (n 1,332; right on 22 pairs).

## Scoring t34-t35, 2026-11-12 05:08 PST (after all outputs; codi_audit_artifacts/t34, t35/layers.json)
t34 (balanced accuracy, valid L4-L6, middle rules k = 2 .. L-1; gain = max(QAZ, QAH) - QA, mean over cells):
- gated: held. QA .46-.56; best latent set .99-1.00 (H); gain .49 (max .53).
- sim: held. QA .47-.65 (max .650, L6 k = 2, at the edge); best .93-.99; gain .40.
- simsf: held. QA .52-.65; best .98-1.00 (QAZ); gain .40.
- stab: held. QA at k = 2 .92-.95; gain .010 (max .041).
- startlast: held. Gain .026 (max .086). Not predicted: QA is high (.92 / .83 at L4, .85-.63 at L6), from the question pass (Q alone
  equal), although t35 reads v_1 exact at .31 at best. Its question pass carries branch bits (coarse magnitude) that exact-value probes miss.
- relay: held. Gain .022 (max .064); QA .75-.94.
- Shuffled <= .56: missed in one cell (gated L5 k = 3, .61); every other cell <= .55. Right-only within .05 for gated / sim / simsf:
  held (<= .003; accuracy ~1.0).
t35 (exact value, best position, emergence = first layer >= .5):
- stab0: v_1 by layer 6 missed (emerges at layer 11, max .58-.64 at 12); v_2 emerges missed (never; max .39-.48); v_3 never held.
  "About two steps per pass" is not supported by exact-value probes, although stab0 answers L3 at .87.
- startlast0: v_1 by layer 12 missed (max .29); v_2 never held.
- stab >= stab0 at layer 12 for v_2, v_3: held (v_2 .52-.58 vs .39-.48; v_3 .27-.30 vs .24-.28).
- sim / simsf / startlast as t21c at every layer: held (v_2 <= .13 at every layer).
- Every value that reaches .5 does so only at layers 11-12.

## t36 released SIM-CoT Coconut on GSM8K (hops.py gsm8k_simcoconut, QUAL=own FIRST=1), 2026-11-12 10:26 PST, before any t36 hop output
Checkpoint internlm/SIM_COT-GPT2-Coconut @676c8a8 checkpoint_28 (base_causallm.* only; the step decoder is dropped). 10 latents
(SIM-CoT gsm_simcot.yaml: c_thought 2 x max_latent_stage 5). Validity check only (200 t22 recipients, no pairs): accuracy .05 / .15 /
.295 / .395 / .39 / .385 at 0 / 3 / 6 / 8 / 10 / 12 latents; connordilgren Coconut .325 at 6. Same design as t31 / t32 (t22 twins and
long twins). Question: does step supervision give Coconut deeper latent use on GSM8K, as it did for Kevin's RuleChain SIM-Coconut?
- Long twins, mean hops at s = 4-7 above plain Coconut's (1.3-1.6) in every bin with n >= 20, and >= 1.8 averaged over s = 4-7.
- No staircase: mean hops at s = 6-7 below s - 2 (as SIM-CODI, about two crossings).
- Unlike plain Coconut, hops do not fall with s: mean at s = 6-7 within .3 of s = 4 (low confidence).
- F_0 <= .15 on long twins (plain Coconut about .2).
- t32 checks: agreement >= .95; hops_first = hops on >= .90 of pairs.
- M_K = donor run on every pair (K = 10).

## t37 Kevin's RuleChain SIM-Coconut (pilot_k8_simcot_c1m6/checkpoint_26): hop count (hops.py simcoconut_rc, QUAL=own) and hidden-path detection (hidden_path.py simcoconut_rc), 2026-11-12 10:48 PST, after smokes only
Loader check: plain run on 60 test items per level L2-L7 reproduces the logged predictions (20261104-144235) on 360 / 360 items.
Smokes seen: hops on 8 pairs (all s = 3, mean 4.12, F_0 0); hidden path L4 300 fit / 200 test (QA .59 / .51, Z .84 / .89, H .82 / .86,
shuffled .54 / .47). Data: synthetic_data_v2/depth2 test L2-L6 (the run's own test items; start value first). Hidden path fits on depth2
train and scores on test L4-L6 (valid has 167 items per level).
- Hops: latent use like SIM-CODI, not plain Coconut: F_0 <= .05 and mean hops >= s - .5 in every s bin with n >= 20 (s >= 2). Low
  confidence alternative: hops follow the latent schedule (latent j trained to decode step j), so hops track the rule position of the
  last changed step, not s, and can exceed s.
- M_K = donor run on every pair.
- Hidden path: QA <= .70 for every k; best latent set >= .80 for every k; mean gain >= .15. Shuffled <= .56 everywhere.

## t38 hop-guided reading: does the hop count say WHEN and WHERE to read? 2026-11-12 11:05 PST, before any t38 output
t38a / t38b: scripts/necessity/hop_guided.py, offline on the t29 readability records joined to their t26 pairs (join checked: item
counts and (L, k, A value, B value) match one to one; no t38 statistic computed yet). For each changed value v_k read in B's own run:
n_r = first refresh from which B's value stays read. Structural alignment d = k - max(i, 1) + 1 (t29); hop alignment h_k = h - (L-1-k)
(the pair's own hop count, counted back from the answer).
- t38a (gated, sim, simsf): n_r = h_k on at least as many items as n_r = d. On items where h_k != d (scored if n >= 20), n_r = h_k more
  often than n_r = d. Within (L, edit, k) cells where hops vary, corr(h_k, n_r) >= .3 (low confidence).
- t38b (gated, sim, simsf): at refresh n_r, the site that first reads B's value is z_(n_r - 1), the input refresh n_r newly makes B's own,
  on >= .7 of items where that site is legible (low confidence; alternative: later inputs re-carry the value, so many sites read it).
- relay / startlast / stab: reported, not scored (few legible deep values).
t38c: hidden_path.py with PER_POS=1 (each latent position alone: Z1..Z5, H0..H4) on gated, sim, simsf, simcoconut_rc, L4-L6.
- The best single position is within .05 of the best all-latent set for every k (branch information is local).
- The best position moves later with k (Spearman >= .8 across k at each level) in gated, sim, simsf, simcoconut_rc.
- Best H position within 1 of k - 1 (pass k - 1 computes rule k) (low confidence).

## Scoring t38a / t38b, 2026-11-12 11:08 PST (codi_audit_artifacts/t38/hop_guided.json)
t38a (first-read refresh n_r vs structural d and hop-aligned h_k = h - (L-1-k)):
- gated: held. n_r = h_k .99 vs = d .76. Where they disagree (n 707, mostly start edits, t29's "one more crossing than d assumes"): h_k
  .99, d .01. Within-cell corr(h_k, n_r) .96 (n 889).
- sim / simsf: missed. n_r = h_k .32 / .29 vs = d .67 / .68; within-cell r .28 / .05. Post hoc: with one more crossing at the answer
  (h_k = h - (L-k)), n_r = h_k .66 / .67, the same as d; where that and d disagree (n 118 / 38) d does better (.48 / .74 vs .36 / .18).
  In SIM models the hop count adds nothing to structure for value timing.
- relay (not scored): n_r = h_k .90 vs = d .39; where they disagree (n 387) h_k .88, d .04 (one crossing carries every value).
- startlast: 119 values read, all at d = h_k (only v_1 legible). stab: no legible values.
t38b (the site that first reads B's value at n_r is z_(n_r - 1), among items where that site is legible): held. gated 1.00, sim .89,
  simsf .90 (relay .77). Unconditional .24 / .57 / .61: the newly refreshed site is often not legible. Partly built in: sites before
  z_(n_r - 1) were already B's own one refresh earlier and did not read B.

## Scoring t36, 2026-11-12 11:48 PST (codi_audit_artifacts/t36)
Qualified pairs 1,287 (t22 twins) + 1,111 (long twins); M_K = donor run on every pair (K = 10): held.
- Long twins s = 4-7 above plain Coconut in every bin: held (2.61 / 2.95 / 2.57 / 3.20 vs 1.58 / 1.44 / 1.29 / 1.34); mean >= 1.8 held.
- No staircase (s = 6-7 below s - 2): held (2.57, 3.20).
- Not falling with s (pooled s = 6-7 within .3 of s = 4): held (2.80 vs 2.61). s = 7 alone is +.59 (n 51).
- F_0 <= .15 on long twins: held (.059).
- t32 checks: agreement 1.00 held; hops_first = hops on .969 / .939 held.
t22 twins by s = 2-5: 2.05 / 2.49 / 3.10 / 2.96. Long twins s = 4 three-plus-hop share .50 (plain Coconut .17, SIM-CODI .21).

## Scoring t37, 2026-11-12 12:40 PST (codi_audit_artifacts/t37)
Clean accuracy L2-L6 .980 / .936 / .866 / .788 / .678 (as logged). 1,937 qualified pairs (786 with a wrong answer).
- Hops: held. F_0 = 0 in every s bin; mean hops 1.61 / 2.56 / 3.46 / 4.32 / 5.32 / 5.99 at s = 1-6 (>= s - .5 everywhere). The
  low-confidence alternative also held: hops exceed s by .3-.6 at s <= 5 (rule-1 edits need L + 1 crossings, e.g. L4 4.54); s = 6 sits
  at the 6-latent cap (5.99). Wrong pairs the same (3.71 vs 3.28 both right, by s within .1).
- M_K = donor run on every pair: held.
- Hidden path: QA <= .70 held at the edge (max .700, L4 k = 2); best latent set .92-.98 held; mean gain +.36 held; shuffled <= .56 held
  (max .55). The answer alone <= .54.

## Scoring t38c, 2026-11-12 13:16 PST (codi_audit_artifacts/t38/pp_<model>.json)
- Best position moves later with k (Spearman >= .8 at each level): held, Spearman 1.0 at every level in all four models.
- Best H position within 1 of k - 1: held. Exactly H_(k-1) in 35 of 36 (model, level, rule) cells (sim L6 k = 2: H0 .94 vs H1 .91).
  Gated H_(k-1) = 1.00 in every cell, neighbouring passes .49-.94; sim / simsf / simcoconut_rc H_(k-1) .88-.98.
- Best single position within .05 of the best all-latent set for every k: held for gated and simcoconut_rc; missed in one sim cell
  (L4 k = 2: .92 vs .99) and two simsf cells (L5 / L6 k = 4: .90 / .91 vs .98).

## t39-t41 crossing-targeted monitoring and deep chains, 2026-11-13 00:14 PST, before any full t39-t41 output
Smokes seen (code checks only): t39 gsm8k_codi and gsm8k_simcoconut at 150 fit / 120 test items; t40 startlast L4 at 300 / 200;
t41 startlast L7 at CAP=2. Smoke numbers are too small to score and are not used below except where stated.

t41 RuleChain L7 hop count (hops.py QUAL=own LEVELS=7, six CODI models, CAP 80). L7 is the deepest trained level; 6 latents.
- gated mean hops at s = 7 >= 5.5; sim / simsf >= 5.5 (capped near 6). 
- startlast / relay / stab: mean hops over all L7 pairs within +.3 of their L6 value in t31 (no rise past single-pass reach).
- startlast: zero-hop share among L7 pairs where both answers are right <= .05 (n may be tiny; scored only if n >= 20).

t40 RuleChain hidden path with per-item predictions (hidden_path.py SAVE_ITEMS=1 LEVELS=4,5,6,7; same six models). Item hops = mean
over the t31 (L4-L6) / t41 (L7) qualified pairs where the item is the recipient. Per-item gain = correct(best latent set) -
correct(QA), best set = the one of QAZ / QAH with higher balanced accuracy for that (level, rule); groups low (hops <= 1.5) and high
(hops >= 2), compared within level and averaged over levels weighted by n.
- L4-L6 reproduce t34 within .03 per model (mean gain).
- L7: gated / sim / simsf QA <= .70 and best latent >= .80 for every middle rule.
- Main (the framing): in startlast and relay, high-hop items show at most .05 more latent gain than low-hop items (crossings in these
  models carry a summary, not the path). Low confidence; the alternative (gain concentrated on high-hop items) would mean the hop count
  locates monitorable items even without a staircase.
- stab: too few high-hop items to score if n < 30.

t39 GSM8K hidden-step monitor (gsm8k_hidden.py; gsm8k_codi, gsm8k_sim, gsm8k_coconut, gsm8k_simcoconut; fit 2,000 train, test 1,319).
Gain = max(QAZ, QAH) - QA in balanced accuracy, averaged over op_2, op_3, big_2, big_3.
- gsm8k_codi and gsm8k_coconut: mean gain <= .05 (low confidence: the codi smoke showed larger gains at n = 120).
- gsm8k_sim and gsm8k_simcoconut: mean gain >= gsm8k_codi's.
- wrong: gain <= .03 for every model (t-wrong-answer monitor added +.001 over confidence).
- Shuffled QAH <= .58 everywhere.
- Main (crossing-targeted): per-item gain as in t40, items with rationale length 3 or 4 only, groups low (item hops <= 1.5) vs high
  (>= 2.5), stratified by length. gsm8k_simcoconut: high - low >= .05. Other models: reported, direction predicted positive (low
  confidence). Bootstrap 95% CIs over items reported for every difference.
- Per position (gsm8k_sim, gsm8k_simcoconut): the best single position for op_k / big_k is non-decreasing in k (low confidence).

### t39-t41 scoring, 2026-11-13 02:03 PST (queue finished 01:55; scoring script scripts/necessity/crossing_split.py)
t41 (L7 hops, by s):
- gated >= 5.5 at s = 7: missed narrowly (5.45). sim 5.92, simsf 6.00: held. All three follow s - 1 at every s.
- startlast 1.45 -> 1.23, relay 1.50 -> 1.61, stab .31 -> .27 (L6 t31 -> L7): held (within +.3).
- startlast zero-hop on both-right L7 pairs: not scored (n = 5).
t40 (RuleChain hidden path per item, L4-L7):
- L4-L6 reproduce t34 within .03: held for all six (gated .486, sim .398, simsf .401, startlast .026, relay .022, stab .010).
- L7 QA <= .70 and best latent >= .80 for every middle rule: held for sim and simsf; gated missed at rule 6 only (best .63; rule 6 is
  decided in pass 5, which the feature set does not include: H = passes 0-4, Z = z_1..z_5; post hoc).
- Main: high-hop minus low-hop latent gain <= .05. startlast +.032 [+.006, +.060] (low .017, n 590; high .051, n 283): held, though
  the difference is above zero. relay +.000 [-.025, +.026]: held. stab: not scored (22 high-hop items).
- Not predicted: staircase models show no hop concentration either (gated +.045, sim -.036, simsf -.019); in them hops follow s.
t39 (GSM8K, mean gain over op_2, op_3, big_2, big_3):
- codi +.037 <= .05: held. coconut +.088: missed. sim +.062 and simcoconut +.092 >= codi: held.
- wrong <= .03: missed for codi (+.154), sim (+.064), coconut (+.059); held for simcoconut (-.025). QA has no confidence feature; the
  earlier wrong-answer monitor compared against the model's own confidence.
- Shuffled QAH <= .58: held (max .53).
- Main: simcoconut high - low >= .05: missed (-.000 [-.051, +.051]; low n 93, high n 205). codi -.005, sim -.037, coconut +.008: the
  predicted positive direction missed for codi and sim; no model shows concentration.
- Per position non-decreasing in k: held for sim (H0, H1, H2 for op and big) and simcoconut (H1, H2, H5 / H1, H3, H5). Not predicted:
  codi (H0, H1, H3 / H0, H2, H4) and coconut (H1, H4, H5 for both) are also monotone.

### t42: GSM8K chain depth (offline re-analysis), written 2026-11-13 20:00 PST before running
Depth = longest dependency chain in the calculator rationale: step j depends on step i when a number in j's expression equals
i's result and does not appear in the question (strict); loose variant also counts numbers that appear in both. Answer not equal
to any step result -> one implicit final step added (flagged). Script scripts/necessity/gsm8k_depth.py.
- Depth < step count for most items with >= 4 steps (median depth at 5+ steps <= 3).
- Accuracy (Kevin's matched CODI vs CODI-0): the gain falls with depth; within a fixed step count (3, 4), lower-depth items show the
  larger gain (low confidence). Teacher accuracy falls less with depth than with step count.
- Released models' accuracy falls with depth at fixed step count (low confidence).
- Hops: item hops flat across depth 1-4 (within +-.3) for codi, sim, coconut; simcoconut also flat (no staircase on GSM8K).
- Readability: per-item latent gain over QA (op_2, op_3, big_2, big_3) shows no trend with depth (|depth 3+ minus depth <= 2| < .05).
- Position: for op_2 / big_2, the best single H position is the same or later when step 2 depends on step 1 than when it does not
  (low confidence, small n).

### t42 scoring, 2026-11-14 (record 20261114-040437_codi_gsm8k-serial-depth-reanalysis; strict rule, loose agrees in direction)
- Depth < step count, median at 5+ steps <= 3: held for depth < steps (median 3 at 4 steps); median at 5+ steps is 4: missed narrowly.
- CODI - CODI-0 gain falls with depth: held at d4+ only (+.094 / +.089 / +.078 / -.005). Lower depth larger within a fixed step count:
  missed (3 steps: d2 +.11, d3 +.13). Teacher falls less with depth than steps: held (both near flat).
- Released accuracy falls with depth at a fixed step count: missed (no consistent fall).
- Hops flat across item depth within +-.3: missed for codi, sim, simcoconut; held for coconut. Not predicted: within item, +.10 to +.17
  crossings per serial step in all four models (CIs above 0).
- Readability no depth trend: missed for op_2 in sim, coconut, simcoconut (independent steps read better); held for big targets.
- Best position same or later for serial steps: not consistent across models.

### t43: per-latent faithfulness aligned to the hop count, written 2026-11-13 20:48 PST before running
Question: is the value a monitor reads at latent z_j the one the answer is computed from, and is that true exactly at the latents the
hop count says the answer crosses (z_1 .. z_C, C = the item's crossings)? Script scripts/necessity/faithful.py.
Faithful = on items the model answers WRONG, continuing the chain from the value read at z_j (re-applying the remaining rules /
re-executing the remaining solution steps) reproduces the model's own wrong answer. Natural runs only: in hop-test mixed runs a latent
does not feed the next pass of the same run, so continuation is undefined there.
RuleChain (gated, sim, simsf, startlast, relay, stab, simcoconut_rc): 100-class ridge probe per (latent z_j, step k) on natural latents,
fit on train L2-L7, pooled over levels; z_j's step k*(j, L) = the step it reads best on correct items. Scored on valid + test L2-L7.
C = hops of the item's start-edit pair (t31 / t41 / t37). Chance = share of the 100 start values whose continuation gives the answer.
- R1 readability: gated, sim, simsf, simcoconut_rc read their best step at >= .90 at z_j for j <= L; startlast and stab < .50 at every z_j.
- R2 (main): in gated, sim, simsf, simcoconut_rc, faithful rate at z_j with j <= C is >= .50 and >= 5x chance; at j > C it is lower by
  >= .20 (n >= 20 wrong items per group, else not scored).
- R3: startlast, relay, stab: faithful rate <= .20 at every z_j with j >= 3.
- R4 depth: in the chaining models, z_3 is faithful more often on items with C >= 3 than on items with C < 3 (by >= .20).
GSM8K (gsm8k_codi, gsm8k_sim; Coconut has no valid readout, section 5.4): readout = numbers among the top-10 vocabulary-projection
tokens of z_j (t30). Faithful at z_j = some read number, put in place of a solution step and re-executed, gives the model's wrong
answer. Control: the same with the next wrong item's z_j tokens. C = mean item hops (t31 pairs).
- G1: faithful rate exceeds the shuffled control at z_1 and z_2 by >= .05 for gsm8k_codi (low confidence for gsm8k_sim).
- G2: no difference >= .05 between j <= C and j > C (low confidence).

### t44: crossing-aware monitor vs reading all latents, written 2026-11-13 21:01 PST before running
Script scripts/necessity/monitor_select.py. Probes (dual ridge, penalty on a held-out 15%) fit on train items, scored on test items that
have an item crossing count C (RuleChain: start-edit hops, t31 / t41 / t37, valid split; GSM8K: mean item hops, t31 / t36).
Probes are POOLED over chain lengths / depths, so a latent's role can differ between items. Monitors (latents z_1 .. z_(K+1); z_1 =
question-pass output): ALL = every latent; CROSS = z_1 .. z_C for that item (a probe per prefix length, applied per item); SHUF = the
prefix length of a random other test item (same sizes, not item-matched); OUT = z_(C+1) .. end; DEPTH (GSM8K) = prefix length from the
item's serial depth (t42 strict) via the model's mean hops at that depth (no minimal pairs needed). T1 and T3 add the answer features.
Targets: T1 gold steps (RuleChain: branch of rules 2..6; GSM8K: op_k / big_k, k = 1..3), balanced accuracy averaged over targets.
T2 the model's own answer, without answer features (RuleChain: 100 classes; GSM8K: quintile of the answer), accuracy on wrong items.
T3 the answer is wrong, balanced accuracy. Paired item bootstrap for CROSS - ALL and CROSS - SHUF.
- RuleChain chaining models (gated, sim, simsf, simcoconut_rc): T1 CROSS within .02 of ALL; OUT below CROSS by >= .10.
- RuleChain startlast / relay / stab: every monitor within .03 of each other on T1.
- GSM8K T1: ALL > CROSS by >= .02 for gsm8k_coconut and gsm8k_simcoconut (readable content beyond C); CROSS within .02 of ALL for
  gsm8k_codi and gsm8k_sim. CROSS > SHUF by >= .01 (low confidence).
- T2 (wrong items): CROSS >= ALL (low confidence). T3: no monitor differs from ALL by more than .03.

### t43 / t44 interim note, 2026-11-13 22:24 PST (before t43b / t44b)
t43's faithfulness on wrong items scores latents BEFORE the chain's error as unfaithful (they hold the gold value, whose continuation is
the gold answer), so it measures where the error is, not whether a latent is used. t44 fit every prefix probe on all train items whatever
their crossings, which handicaps CROSS (pooled roles). Both are kept as run; t43b / t44b are the corrected tests.

### t43b: local consistency between successive latents, written 2026-11-13 22:24 PST before running
Same models, probes and eval items as t43 (RuleChain; scripts/necessity/faithful.py with LOCAL=1). Link z_j -> z_(j+1) is CONSISTENT when
the value read at z_(j+1) (its step k*(j+1)) equals the rules k*(j)+1 .. k*(j+1) applied to the value read at z_j (equality if
k*(j+1) <= k*(j)). Scored on wrong items, on links where at least one side differs from the gold value (non-trivial links). Chance = the
share of the 100 values at z_j that would give the z_(j+1) reading. Crossings C = level mean of start-edit hops (t43 post hoc).
- In gated, sim, simsf, simcoconut_rc: non-trivial links with j + 1 <= C are consistent at >= .50 and >= 5x chance, and beat links with
  j + 1 > C by >= .15 (scored only where both groups have >= 20 links).
- startlast, relay, stab: non-trivial consistency <= .20 at every link (their latents are barely readable).

### t44b: matched-fit crossing-aware monitor, written 2026-11-13 22:24 PST before running
scripts/necessity/monitor_match.py. Every train and test item gets c = its group crossings: RuleChain round(level mean start-edit hops),
GSM8K round(the model's mean hops at the item's t42 serial depth). For each c: CROSS_m = prefix z_1..z_c and ALL_m = all latents, both fit
only on train items with that c (same data, different features); ALL_p = all latents fit on all train items (t44's ALL). Test: items of
each group read by that group's probes; secondary CROSS_meas = measured item C on RuleChain valid items, probe of group C (nearest).
Same targets T1-T3, paired item bootstrap.
- T1, chaining models: CROSS_m within .02 of ALL_m (one crossing per step: the crossed latents hold every step).
- T1, startlast / relay / stab and GSM8K: ALL_m >= CROSS_m (low confidence); CROSS_m within .03 of ALL_m.
- ALL_m vs ALL_p on T1: no difference > .03 (low confidence).
- T2 / T3: CROSS_m within .03 of ALL_m everywhere it is scored.
- t43b addendum (22:26, after the smoke test, before the full run): links are also reported by kind, compute (z_(j+1) reads a
  later step than z_j) vs copy (same or earlier step). The registered predictions above are scored on all non-trivial links as written.

### t43c: readout ceiling for t43b, written 2026-11-13 23:50 PST before running
Same probes and link test as t43b, scored on CORRECTLY answered items (all links; true values known), env CEIL=1, for simsf,
simcoconut_rc, startlast, relay. Consistency there = the rate a perfectly faithful chain would reach given readout error.
- If compute-link consistency on correct items is >= .50 (inside the crossings), the t43b wrong-item rates (.13 simsf, .04 simcoconut_rc)
  are well below the readout ceiling: report that the crossed latents' readouts do not track the model's own errors.
- If it is < .30, the t43b null is readout-limited: report per-item faithfulness at the crossings as untested.
- Prediction: simsf and simcoconut_rc >= .50 (low confidence); startlast and relay < .30 (their latents are barely readable).

### t45: branch faithfulness on wrong answers, written 2026-11-13 23:55 PST before running
Script scripts/necessity/branch_faith.py (RuleChain: gated, sim, simsf, startlast, relay, stab, simcoconut_rc). Binary ridge probe per
(latent z_j, rule k) on natural latents (train L2-L7, pooled over levels); for each (level, rule) the latent that reads the branch best on
correctly answered eval items (valid + test). On WRONG items: replay the chain from the question's start value with every branch forced
to the one read, and score whether it gives the model's own answer. Scored on branch-explainable errors (some branch pattern gives the
model's answer). Chance = share of the 2^L patterns that give it. Gold branches give the gold answer (checked). Hop tie-in: the same with
each rule's latent chosen only among z_j <= C (IN) or only among z_j > C (OUT), C = round(level-mean start-edit hops).
- B1: simsf and simcoconut_rc reproduce the model's answer on >= .30 of branch-explainable errors and >= 5x chance (n >= 20 errors).
- B2: IN >= OUT for simsf and simcoconut_rc (low confidence).
- B3: startlast, relay, stab reproduce <= .20.
- gated and sim: reported only (too few errors expected).
- t45 addendum (before the full run): where a level has too few correct items to choose a rule's latent, it is chosen on correct items
  of all levels that have that rule.

### t43-t45 scoring, 2026-11-14 00:25 PST (records 20261114-0818xx_*)
t43 (continuation faithfulness; metric flawed, see interim note): R1 held for gated (.91-1.00) and for startlast / stab (< .50); missed
for sim (.78-.95), simsf (.50-.91) and simcoconut_rc (.34-1.00) at later latents. R2 missed (inside-C lower than outside in every model
with n; the metric scores pre-error latents as unfaithful). R3 held (low crossers <= .15 at z >= 3). R4 not scorable (n). G1 missed
narrowly (gsm8k_codi z1 +.048; z2 holds no numbers). G2 held (codi .047 vs .052; sim .025 vs .048).
t43b (local consistency, wrong items): chaining-model prediction missed (simsf inside .13 vs outside .18; simcoconut_rc .05 vs .15;
gated / sim not scorable, 4 and 10 links). Low crossers <= .20 at every link: missed as registered (copy links .58-.70); compute links
held (<= .03).
t43c (ceiling, correct items): simsf compute-inside .63 and simcoconut_rc .57 (>= .50: held) -> their t43b wrong-item rates (.13, .04)
are well below the readout ceiling; startlast .05 and relay .06 (< .30: held) -> readout-limited, untested.
t44 (pooled probes): chaining CROSS within .02 of ALL missed (-.06 to -.11); OUT below CROSS by .10 held for sim only; low crossers
within .03 missed narrowly for all three (spread .032-.040); GSM8K ALL > CROSS by .02 held for coconut / simcoconut, CROSS within .02
missed for codi / sim (-.034); T2 CROSS >= ALL missed; T3 within .03 missed for gsm8k codi, sim, coconut.
t44b (matched fit): chaining CROSS_m within .02 of ALL_m held (gated +.001, sim .000, simsf -.009, simcoconut_rc -.016). ALL_m >= CROSS_m
held everywhere else; within .03 held for startlast, relay, missed for stab (-.039) and all GSM8K (-.032 to -.065). ALL_m vs ALL_p
within .03 held except gated (-.034). T2 / T3 within .03 missed for simsf and GSM8K (T2 up to -.28 for gsm8k_simcoconut).
t45 (branch faithfulness): B1 held for simcoconut_rc (.42 [.34, .50] of 163 branch-explainable L2-L6 errors vs .056 chance; 383 L7
errors not scorable, rule 7 has no latent) and missed for simsf (.03, n 34). B2 held (simcoconut_rc IN .42 vs OUT .10; simsf .03 vs 0).
B3 held (startlast .04, relay .03, stab .02; chance .02-.03). gated .09 (IN .27, n 11), sim n 3: reported only.
- t45 post hoc caveat (2026-11-14): B2 (IN >= OUT) is confounded by readability. On correct items the best IN latent reads branches at
  .97 / .88 / .81 (simcoconut_rc / simsf / gated) and the best OUT latent at .68 / .64 / .58, so a replay from OUT latents fails more often
  regardless of faithfulness. B2 shows that branches are readable at the crossings (as t38c), not that the crossings are where latents are
  faithful. B1 (simcoconut_rc .42 vs .056 chance) stands; not used in the paper.

### t46: hop count on Liang & Pan's released polynomial-iteration CODI, written 2026-11-15 20:25 PST before the full run
Handoff HANDOFF_LIANG_PAN_HOPS_2026-11-15.md. Script scripts/necessity/hops_poly.py, adapter adapters.PolyCODI, their repo
../latent-cot-thinking (commit 815fe40, imported by path). Models totalseq_{10,8}_mod{50,41}, pytorch_model.bin. Items: their shipped
test split, k = 2..N, 80 items per k (seeded shuffle), every edit position i = 1..k on the same items.
Gates, done before writing this: gate 1 (their test.py, CPU) reproduces test_results.csv to < 1e-7 at every length, all four models.
Gate 2 (adapter natural run vs their CODI.forward on the test.py batch, 200 items per length) 200/200 at every length, all four models.
Deviations from the handoff, decided before the full run:
- Padding: questions are left-padded with masked PAD to the length-N width. Their training and test.py collate mixed lengths, so the
  question always ends at absolute position N + 1; an unpadded k = 1 question scores 0 instead of 1. "No padding" (handoff step 5) would
  be off distribution. Each batch still holds one k, so every row has the same pad count.
- Edits: every replacement v in 1..m-1 (v != x_i) is run instead of resampling up to a cap. The pair's A is drawn uniformly among the
  replacements that change the model's answer. Recorded per (item, i): the share of replacements that change the model's answer, the
  share that change the gold answer, and whether none changes the model's answer.
- Direction as the handoff: B = the test item (the question run), A = the edited item (latent source). M_0 = B's question with A's
  natural latent inputs; hops = the first n from which the answer stays the model's own B answer (QUAL=own). M_6 == B's run is asserted.
Smoke (SMOKE=1, k = 3, 4, 14 pairs per model; seen before writing this): mod 50 hops 0.00 / 1.50 / 3.25 at s = 1 / 2 / 3 (chain-like);
mod 41 0.00 / 1.83 / 0.50. About 92-97% of single-input edits changed the model's answer in both. The predictions below are the
handoff's, kept unchanged after the smoke so that the smoke does not shape them.
- P1 (mod 50, N = 10 and N = 8): mean hops <= 2.0 at every s (1..9 for N = 10, 1..7 for N = 8) with n >= 20 pairs. Miss: any such s with a
  point estimate above 2.0.
- P2 (mod 50): within-item slope of hops on s (item fixed effects; bootstrap over items) below +.25 crossings per step. Miss: point
  estimate >= .25 (a chain gives about +1).
- P3 (mod 50): edits to the last input (s = 1) need <= .3 hops on average. Miss: mean > .3.
- P4 (mod 41): at k >= 4, fewer than half of the single-input replacements of x_1..x_{k-2} change the model's answer (pooled share of
  replacement values). Where pairs qualify, mean hops <= 2.0 at every s with n >= 20. Miss: share >= .5, or any such s above 2.0.
- P5: accuracy by length reproduces their table (gate 1; held before this block). The hop count does not rise with k where accuracy
  falls: for every k >= 4, mean hops by k is no more than .5 above mean hops at k = 3 (same model). Miss: any k >= 4 exceeding that.
Reported, not predicted: the same statistics for the other modulus, the gold-differs subset, the both-right subset, the share of M_n
answers that are neither A's nor B's (n = 0..6), and the share of (item, i) with no answer-changing replacement.

### t46 scoring, 2026-11-15 20:30 PST (JSON ../codi_audit_artifacts/t46/poly_{10,8}_{50,41}.json; all pairs qualify, M_6 == B on every pair)
- P1 MISSED (mod 50, both models). N = 10: s = 1..9 mean hops .03 / 2.03 / 2.53 / 2.01 / .62 / .49 / .36 / .43 / .65 (s = 3 CI
  [2.40, 2.65]). N = 8: .03 / 2.34 / 2.26 / .83 / .19 / .30 / .16. Above 2.0 at s = 2-4 (N = 10) and s = 2-3 (N = 8).
- P2 HELD as registered, but for the wrong reason. Pooled within-item slope -.07 [-.09, -.05] (N = 10) and -.07 [-.11, -.03] (N = 8). The
  pooled slope is set by early inputs that the model mostly ignores (s >= 5: .09-.29 of replacements change the model's answer against
  .42-.70 for the gold answer; the edits that do change it take about .5 hops). Post hoc, not registered: at k = 3, 4 (accuracy 1.00 and
  .48 / .39) the within-item slope is +1.54 [1.44, 1.64] (N = 10) and +1.28 [1.15, 1.43] (N = 8). By cell, N = 10: k = 3 hops 1.83 / 1.87 / .00
  for x_1 / x_2 / x_3 (s = 2, 2, 1); k = 4 3.08 / 2.96 / 1.82 / .03 (s = 3, 3, 2, 1). Both-right subset: s = 2 1.94 (n 251), s = 3 3.08 (n 97).
  Where the model computes, hops are about s for s <= 3. That is a chain.
- P3 HELD. s = 1: .03 [.01, .06] (N = 10), .03 [.01, .05] (N = 8).
- P4 MISSED on the first clause. Early replacements (x_1..x_{k-2}, k >= 4) change the mod-41 model's answer at .66 (N = 10) and .74
  (N = 8), although its accuracy is at chance (.02-.03). The second clause held: max mean hops 1.97 at s = 9 (N = 10, n 160) and 1.01
  (N = 8).
- P5: accuracy reproduced (gate 1). Hops by k against k = 3: HELD for poly_8_50 (k = 4 +.47, max), poly_10_41 (+.03) and poly_8_41 (all
  lower). MISSED for poly_10_50: k = 4 1.94 against 1.22 at k = 3 (+.72). From k = 5 on, the count falls in all four models.
- Reported: share of M_1 answers that are neither A's nor B's .16 / .15 / .14 / .05 (poly_10_50 / 10_41 / 8_50 / 8_41); at M_0 .13 / .21 /
  .13 / .12. (item, i) cells with no answer-changing replacement: mod 50 s >= 5 .33-.58, s <= 2 <= .015; mod 41 <= .11 everywhere.
  Gold-differs subset: same picture (mod 50 slope -.02 / -.01). Mod 41 at k = 3 (right): s = 2 1.56-1.73 (N = 10), 2.29-2.30 (N = 8).
Reading: these models use up to about three loop crossings, one per serial step over the last three inputs. Beyond that they do not
extend the chain: early inputs are mostly ignored on mod 50, and both accuracy and hops fall. This fits Liang & Pan's "two to three hops",
and it contradicts "zero to two crossings" as a general claim about standard CODI. The count is an upper bound (copy crossings).

### t47: hop count on the released Llama-3.2-1B CODI (GSM8K), written 2026-11-17 14:20 PST before the full run
Model zen-E/CODI-llama3.2-1b-Instruct through adapters.GSM8KCODILlama (args of codi/scripts/test_llama1b.sh: LoRA r128/a32 on all
Llama projections, projection 2048 + LayerNorm, 6 latents, remove_eos, greedy; fp32 on MPS). Base config/tokenizer from the ungated
mirror unsloth/Llama-3.2-1B-Instruct with Meta's tokenizer settings restored (adapters._llama_base); the CODI checkpoint loads with 0
missing keys (4 unexpected: an unused dynamic_cls head), so every weight comes from the release. Same pairs and settings as the GPT-2
released hop counts (t31): t22 twins (1,471) + t26 long twins (1,305), QUAL=own. FIRST=0: Llama-3 tokenizes " 18" as " " + "18", so the
GPT-2 first-token prefix does not carry over; the main count (greedy answer) is unaffected.
Gate (running in parallel, read before scoring): GSM8K-Aug test accuracy within .03 of the published .556. Smoke (seen before writing
this): 8 pairs, write-back check passes, M_6 == donor run 8/8, hops 1.0 at s = 2 (n 5), 2.0 at s = 3 (n 3).
Predictions (reference values: released GPT-2 CODI, t42 strict / steps-per-pass reanalysis):
- P1 gate: accuracy in [.526, .586].
- P2: M_6 equals the donor's own run on every pair run.
- P3: mean hops over qualified pairs <= 2.5 at every serial distance 2-5 with n >= 20 (GPT-2 CODI 1.42-1.91). Miss: any above 2.5.
- P4: within-item slope of hops on serial distance (item-demeaned, item bootstrap, strict) below +.35 (GPT-2 CODI +.16). Miss: >= .35.
- P5: on pairs answered right for both questions with the edit at the start of the item's longest path, crossings at depth 3 are at
  most GPT-2 CODI's 1.94 (a deeper backbone should need no more passes for the same depth). Miss: > 1.94 with n >= 20.
Reported, not predicted: accuracy by item depth, hops by depth on correct pairs (steps per pass), share of M_1 answers neither R's nor D's.

### t47 scoring, 2026-11-17 20:05 PST (JSON ../codi_audit_artifacts/t47/{gsm8k_llama_acc,gsm8k_llama,long_gsm8k_llama,steps_per_pass_with_llama,gsm8k_llama_compare}.json)
- P1 HELD. Accuracy .555 on 1,319 (published .556). By item depth (strict) d1-d4+: .556 / .671 / .509 / .343 (GPT-2 CODI .393 / .581 / .321 / .239).
- P2 HELD. M_6 == donor run on 1,315 of 1,315 (t22 twins) and 1,142 of 1,142 (long twins). Qualified 1,315 / 1,471 and 1,142 / 1,305.
- P3 HELD. Mean hops by serial distance 2 / 3 / 4 / 5, all qualified on-path pairs: 1.13 / 1.41 / 1.48 / 1.29 (n 954 / 783 / 311 / 68;
  GPT-2 CODI 1.42 / 1.75 / 1.91 / 1.87).
- P4 HELD. Within-item slope +.002 [-.060, +.062] (336 items; GPT-2 CODI +.157); correct pairs +.043 [-.041, +.133] (106 items).
- P5 HELD. Correct pairs, edit at the start of the longest path: depth 2 / 3 / 4 crossings .91 [.84, .99] / 1.46 [1.38, 1.56] / 1.74
  [1.57, 1.91] (n 236 / 157 / 54) against GPT-2 CODI 1.07 / 1.94 / 2.39. Steps per pass 1.15 / 1.29 / 1.59 (GPT-2 CODI 1.01 / 1.05 / 1.29).
- Reported: share of correct pairs with hops >= s - 1 at s = 3 / 4: .49 / .04 (GPT-2 CODI .86 / ~.5); hops 0 on .11 of correct s = 2 pairs.
  Pooled hops 1.27 (t22) / 1.35 (long); first refresh follows D on .70 / .66 of pairs (GPT-2 CODI .46 on long twins).
Reading: the 16-layer Llama CODI chains on items it solves (crossings rise with item depth, .91 to 1.74) but needs fewer crossings than
12-layer GPT-2 CODI at the same depth, about 1.2-1.6 steps per pass against 1.0-1.3. Within an item the count does not depend on the edit's
distance (slope ~0), so it tracks the item's depth, not the edit position. Same direction as GPT-2's correct pairs, stronger.

### t48: readability by refresh, released Llama-3.2-1B CODI and GPT-2 CODI with a matched readout, written 2026-11-17 21:05 PST before the full runs
Script scripts/necessity/gsm8k_lens.py (t30's design: same refreshes, step depths d from the re-executed reference solution, own-readable
subset, stale-A and chance controls), new env READ. READ=z is t30's readout (top-10 vocabulary projection of the fed-back latent inputs).
On Llama it reads nothing (6-pair smoke: own-readable 0 of 15 values; a 3-problem diagnostic shows the projected latents map to unrelated
tokens). READ=h reads the output hidden state of each pass before CODI's projection with the LM head (HF hidden_states[-1], after the final
norm): the standard logit lens, at the same positions (z_s = prj(h_{s-1})). Runs: gsm8k_llama READ=h on t47 pairs answered right on both
questions; gsm8k_codi READ=h on t30's pairs (t26 + long, QUAL=gold) as the matched GPT-2 reference; t30's READ=z numbers stay as they are.
Single-token values only (Llama-3: numbers below 1,000 without a leading space). Smoke (seen before writing this, 6 pairs each): READ=h reads
1.00 / .67 of values at d = 1 / 2 on both models; B arrives at refresh d or earlier.
Predictions:
- P1 (readout works): own-readable rate at d = 1 >= .5 for both models under READ=h. Miss: below .5.
- P2 (timing at d = 1): among own-readable values, B is readable at refresh 0 on <= .25 and at refresh 1 on >= .85, both models.
- P3 (more per pass in Llama): at d = 2 and d = 3, the share of own-readable B values already readable at refresh d - 1 is higher for Llama
  than for GPT-2 CODI under READ=h (t47: Llama uses fewer crossings per step). Miss: not higher at either depth (n >= 15 per cell).
- P4 (controls): chance <= .10 at every refresh for d with n >= 15; stale A readable at refresh d - 1 on >= .3 of own-readable values.

### Correction to the t47 / t48 header times, appended 2026-11-17 20:59 PST
The clock times typed in three headers above were estimates and are wrong; the order of events is unchanged. Evidence: run logs and file times.
- t47 predictions: header says 14:20. They were written after the 8-pair smoke (finished about 14:15) and before the full hop runs, whose
  logs start at 14:17 (codi_audit_artifacts/t47/{gsm8k_llama,long_gsm8k_llama}.log). The accuracy gate was running in parallel (its log
  starts 14:14), as the block says.
- t47 scoring: header says 20:05; it was written about 19:50, after the original-twins run ended at 19:32.
- t48 predictions: header says 21:05. The block was saved at 20:46:56 (this file's modification time when it was appended); both full runs
  start at 20:47 (codi_audit_artifacts/t48/*.log).

### t48 scoring, 2026-11-17 21:35 PST (JSON ../codi_audit_artifacts/t48/gsm8k_{llama,codi}_lens_h.json)
Own-readable rate d = 1 / 2 / 3 / 4: Llama .90 / .60 / .37 / .39 (n 727 / 672 / 356 / 126); GPT-2 CODI .87 / .53 / .42 / .26 (n 432 / 421 /
212 / 78); t30's READ=z on GPT-2 CODI was .37 / .17 / .14. Among own-readable values, B readable by refresh 0, 1, 2, 3:
Llama d1 .33 .98; d2 .31 .92 1.00; d3 .31 .74 .98 1.00; d4 .49 .92 .98 1.00. GPT-2 d1 .34 .99; d2 .30 .60 .98; d3 .30 .52 .73 1.00;
d4 .20 .20 .75 .95.
- P1 HELD (Llama .90, GPT-2 .87 at d = 1).
- P2 MISSED on the first clause for both: B already readable at refresh 0 on .33 (Llama) / .34 (GPT-2), above .25; the second clause held
  (.98 / .99 at refresh 1). Refresh 0 reads A's own latents, so this is a pair-specific false-positive level of READ=h (A and B share
  most numbers; the h lens spreads mass over many numbers), about .30 at every depth and above the cross-item chance.
- P3 HELD. Share readable at refresh d - 1: d = 2 Llama .92 vs GPT-2 .60; d = 3 .98 vs .73. Relative to each model's refresh-0 level,
  (B[d-1] - B[0]) / (1 - B[0]): d = 2 .88 vs .43, d = 3 .97 vs .61. At d = 4, Llama .92 by refresh 1 against GPT-2 .20.
- P4 MISSED. Chance up to .19 (Llama) / .14 (GPT-2), above .10. Stale A at refresh d - 1 >= .3 held at d = 1-3 for Llama and d = 1, 2, 4 for
  GPT-2; missed at Llama d = 4 (.18) and GPT-2 d = 3 (.26).
Reading: READ=h makes far more values legible than t30's readout, so "legibility is low" was a readout property. Timing: in Llama nearly
every value at depth 2-4 is readable within one or two refreshes, whatever its depth; in GPT-2 CODI about half arrive one refresh early.
That matches t47 (Llama about 1.2-1.6 steps per pass, GPT-2 about 1.0-1.3): the readout and the causal count agree that Llama packs more
serial steps into a pass. The higher false-positive level of READ=h means early-arrival rates should be read against refresh 0.

### t49: hop count and readability on the released SIM-CODI Llama-3.2-1B (GSM8K), written 2026-11-17 22:15 PST (from `date`) before the full runs
Model internlm/SIM_COT-LLaMA3-CODI-1B through adapters.GSM8KCODILlama (README: LoRA r128/a32, projection 2048, 6 latents, bf16 in their
eval; fp32 here). Loads with 0 missing keys; 147 unexpected keys are the training-only step decoder (decoder.*), unused at inference.
Same pairs and settings as t47 (t22 twins + t26 long twins, QUAL=own, FIRST=0), then t48's readability (READ=h) on pairs answered right
on both. Smoke (seen): 20 items, .50 accuracy, 0.6 s/item. Published GSM8K-Aug accuracy .561 (SIM-CoT README, LLaMA 3.2 1B, CODI + SIM-CoT).
Purpose: a within-recipe depth series for SIM-CODI (GPT-2 12 layers -> Llama 1B 16 layers; 3B 28 layers to follow if it fits locally).
Predictions:
- P1 gate: accuracy in [.531, .591].
- P2: M_6 equals the donor's own run on every pair run.
- P3: correct start-of-path crossings at depth 3 below GPT-2 SIM-CODI's 1.98 (the deeper backbone carries more per pass). Miss: >= 1.98.
- P4: the step loss changes little at a fixed backbone: correct start-of-path crossings at depth 2 / 3 / 4 each within .30 of Llama CODI's
  .91 / 1.46 / 1.74 (t47). Miss: any depth with n >= 20 off by more than .30.
- P5: within-item slope of crossings on serial distance below +.35.
- P6 (readability, READ=h): share of own-readable values readable at refresh d - 1 at d = 2 and 3 above GPT-2 CODI's .60 / .73 (t48).

### t50: hop count and readability on the released SIM-CODI Llama-3.2-3B (GSM8K), with a bf16 control on the 1B, written 2026-11-17 23:53:13 PST (from `date`) before any 3B output
Model internlm/SIM_COT-LLaMA3-CODI-3B@a2db0e5 (28 layers; README: LoRA r128/a32, projection 3072, 6 latents, bf16 eval). Only the
inference tensors (codi.*, prj.*; 653 tensors, 8.07 GB) were fetched by byte range (scripts/necessity/fetch_simcot3b.py); the step decoder
was not. Loaded in bf16, as the release's eval, with a streaming loader (adapters._build_codi_streaming); the 1B runs (t47, t49) are fp32.
Spot checks seen: lm_head equals the embeddings (tied); LoRA B non-zero. Nothing else seen: no smoke run yet.
Same pairs and settings as t47 / t49 (t22 twins + t26 long twins, QUAL=own, FIRST=0), then READ=h readability on pairs right on both.
Published GSM8K-Aug accuracy .623 (SIM-CoT Table 3, CODI + SIM-CoT, LLaMA 3.2 3B; plain CODI 3B .608).
Dtype control: the 1B SIM-CODI loaded as the 3B is (bf16, streaming loader) on all 1,319 test items and the first 300 t22 pairs.
Purpose: within-recipe depth series GPT-2 (12 layers) -> Llama 1B (16) -> Llama 3B (28): does the count of loop crossings per serial
step keep falling as the backbone deepens?
Predictions:
- P0 load: 0 missing and 0 unexpected keys.
- P1 gate: accuracy in [.593, .653]. Miss: stop before the hop count.
- P2: M_6 equals the donor's own run on every pair run.
- P3: correct start-of-path crossings at depth 3 and at depth 4 below SIM-CODI 1B's (t49, same pairs; cells with n >= 20). Miss: either
  depth at or above t49's value.
- P4: within-item slope of crossings on serial distance below +.35.
- P5 dtype control: on 1,319 items the bf16 1B's accuracy is within .02 of t49's fp32 .557; among the first 300 pairs qualified in both
  runs, at least .80 have the same hop count. Miss: the 1B vs 3B contrast is also a dtype contrast; compare the 3B with the bf16 1B.
- P6 (readability, READ=h): share of own-readable values readable at refresh d - 1 at d = 2 and 3 above .80.
- P7 (exploratory, two steps in one pass): at d = 3, share readable by refresh 1 above Llama CODI 1B's .74 (t48).

### t49 scoring, 2026-11-18 03:02 PST (from `date`; JSON ../codi_audit_artifacts/t49/{gsm8k_simllama1b_acc,gsm8k_simllama1b,long_gsm8k_simllama1b,gsm8k_simllama1b_lens_h,steps_per_pass_with_simllama1b,gsm8k_llama_compare}.json)
- P1 held: accuracy .557 (published .561).
- P2 held: M_6 = donor run on 1,313/1,313 (t22) and 1,166/1,166 (long) pairs.
- P3 held on the letter: depth-3 correct start-of-path crossings 1.93 [1.85, 2.01] < GPT-2 SIM-CODI's 1.98 [1.89, 2.07]; the CIs overlap, so equal, not below.
- P4 missed: 1.00 / 1.93 / 2.59 vs Llama CODI 1B's .91 / 1.46 / 1.74 (depth 3 +.47, depth 4 +.85). The step loss changes loop use a lot.
- P5 held: within-item slope +.103 [+.020, +.191].
- P6 missed narrowly: readable at refresh d - 1, d = 2 / 3: .58 / .71 vs GPT-2 CODI's .60 / .73 (Llama CODI 1B .92 / .98).
Reading: SIM-CODI 1B uses about one crossing per serial step (steps per pass 1.04 / 1.07 / 1.15), like the GPT-2 checkpoints and unlike plain
CODI 1B (1.15 / 1.29 / 1.59), at the same accuracy by depth. t50's P3 (3B below t49) is now a test of depth within SIM-CoT.

### t51: per-layer logit lens inside each pass, released CODI-family checkpoints on GSM8K, written 2026-11-18 09:30:55 PST (from `date`) before any t51 output
Script scripts/necessity/gsm8k_layer_lens.py (docstring has the definitions). Models: GPT-2 CODI and GPT-2 SIM-CODI (12 layers, fp32, CPU),
Llama-3.2-1B CODI and SIM-CODI (16 layers, fp32), SIM-CODI 3B (28 layers, bf16); the Llama runs wait for the t50 chain. Natural runs on the
1,319 test items, items answered right; logit lens (final norm + output head, top 10) at every layer of pass 0 (question) and passes 1-6.
Kept steps: single-token integer values not among the question's numbers and not repeated in the rationale. s* = first pass whose output
reads the value; onset = lowest layer from which it stays read through the output layer in pass s*.
Seen before writing: the t50 3B hop count on the t22 pairs (correct start-of-path crossings .93 / 1.77 / 2.04, steps per pass 1.09 / 1.14 /
1.47); no per-layer lens output of any model.
Purpose: where inside a pass a step's value appears, and whether two steps in one pass appear in serial order across layers.
Predictions:
- P1 (sanity): at pass 0, layer 0 (the input embedding of the last question token), every depth reads at most chance + .02.
- P2: share of read depth-2 values whose s* is pass 0 (two steps in the question pass): Llama CODI 1B above SIM-CODI 1B by at least .10.
- P3: same-pass parent/child pairs, pooled over models: child onset later than parent onset more often than earlier (one-sided sign test
  p < .05).
- P4: layers left after the onset (L - median onset) for depth-1 values with s* = pass 0: SIM-CODI 3B >= SIM-CODI 1B >= GPT-2 SIM-CODI.
- P5 (exploratory): the onset as a fraction of L, by model; no direction predicted (the logit lens favours late layers).

### t52: robustness of the GSM8K depth results to the depth rule, written 2026-11-18 19:45:49 PST (from `date`) before any t52 output
Script scripts/necessity/gsm8k_depth_robust.py (offline; reuses the t42/t47/t49/t50 pairs, the t39/t47/t49/t50 accuracy files, the
t48-t50 READ=h files and the t51 layer-lens files). Rules: strict (t42, as used so far); loose (a number in a step's expression links
to the latest earlier step with that result even if the number is also in the question); resolved (loose, but a number that is also in
the question links to the earlier step only when that step's result is otherwise unused, i.e. a dangling intermediate); clean (items
with no ambiguous number, no implicit final step and no unexplained operand; all rules agree there); ops (exploratory: arithmetic
operations along the path instead of steps, on the resolved links).
Seen before writing: the strict-rule depth by step-count table (46% of items have depth below their step count; 87 of 117 depth-1
items have 2+ steps), eight depth-1 items with 3+ steps (the strict rule drops a link when a computed value equals a question number,
including number words such as "six"; some rationales skip a computation, e.g. 102 = 20.25 + 15.75 + 66 never written), and the
strict/loose disagreement on 9% of steps. No re-analysis output.
Purpose: whether the t47-t51 conclusions depend on mislabeled depth.
Predictions:
- P1: SIM-CODI 3B's correct start-of-path crossings at depth 4 stay below SIM-CODI 1B's under loose, resolved and clean (cells n >= 15).
- P2: Llama CODI 1B's steps per pass at depth 3 exceed SIM-CODI 1B's by at least .10 under every rule.
- P3: GPT-2 SIM-CODI and SIM-CODI 1B steps per pass at depth 2 and 3 stay within [.95, 1.20] under every rule.
- P4: the depth-1 accuracy dip is a labeling artifact: under resolved and under clean, depth-1 accuracy >= depth-2 accuracy for at
  least 4 of the 5 CODI-family models (under strict it is below for all 5).
- P5: on clean items, READ=h readability at refresh d - 1 (d = 2, 3) is within .05 of the full-set value for every model with a file.
- P6: t51 same-pass parent/child pairs on clean items, pooled: child onset later more often than earlier, one-sided sign p < .05.
- P7 (exploratory): operations per pass; no direction predicted.

### t50 scoring, 2026-11-18 19:50:05 PST (from `date`; JSON ../codi_audit_artifacts/t50/{gsm8k_simllama3b_acc,gsm8k_simllama3b,long_gsm8k_simllama3b,gsm8k_simllama1b_bf16_acc,gsm8k_simllama1b_bf16,gsm8k_simllama3b_lens_h,steps_per_pass_with_simllama3b,gsm8k_llama_compare}.json)
- P0 held: 0 missing, 0 unexpected keys.
- P1 held: accuracy .622 (published .623).
- P2 held: M_6 = donor run on 1,347/1,347 (t22) and 1,177/1,177 (long) pairs.
- P3 held: correct start-of-path crossings at depth 3 / 4: 1.89 [1.80, 1.98] (n 196) / 1.99 [1.84, 2.15] (n 80) vs SIM-CODI 1B 1.93 / 2.59.
  Depth 3 only on the letter (CIs overlap); depth 4 clearly below (steps per pass 1.44 vs 1.15).
- P4 held: within-item slope +.122 [+.020, +.222].
- P5 held: bf16 1B accuracy .5595 (fp32 .557); .911 of the first 300 pairs have the same hop count. Only 4 bf16 depth-4 pairs, so the
  control covers depth 2-3, not depth 4.
- P6 missed: readable at refresh d - 1, d = 2 / 3: .69 / .92 (needed both above .80). SIM-CODI 1B .58 / .71; Llama CODI 1B .92 / .98.
- P7 missed: d = 3 readable by refresh 1 .65 (Llama CODI 1B .74; SIM-CODI 1B .43). Depth 4 by refresh 1: .68 (SIM-CODI 1B .25).
Reading: at 28 layers SIM-CoT still runs about one step per pass at depth 2-3, but packs about 1.4 steps per pass at depth 4, and
readability moves the same way. Accuracy is .05-.08 higher than the 1B at every depth bin, not only the deep ones.

### t51 scoring, 2026-11-18 19:50:05 PST (from `date`; JSON ../codi_audit_artifacts/t51/layer_lens_gsm8k_{codi,sim,llama,simllama1b,simllama3b}.json)
- P1 held: pass 0, layer 0 reads 0 at every depth in every model (chance 0).
- P2 held: depth-2 share with s* = pass 0: Llama CODI 1B .60 vs SIM-CODI 1B .25 (+.35).
- P3 held: same-pass child onset later / earlier / tie: GPT-2 CODI 38/9/18, GPT-2 SIM-CODI 17/6/7, Llama CODI 1B 31/18/28, SIM-CODI 1B
  39/19/12, SIM-CODI 3B 34/8/16; pooled 159/60, one-sided sign p = 8.0e-12 (each model alone p < .05).
- P4 missed: L - median onset for depth-1 values with s* = 0: SIM-CODI 3B 2 (26/28), SIM-CODI 1B 3 (13/16), GPT-2 SIM-CODI 1 (11/12);
  3B is not >= 1B. Means 1.86 / 2.47 / .93.
- P5 (exploratory): median onset / L for those values: GPT-2 CODI .83, GPT-2 SIM-CODI .92, Llama CODI 1B .88, SIM-CODI 1B .81, SIM-CODI
  3B .93. In every model the value becomes readable in the last 1-3 layers.
Exploratory, not predicted: in SIM-CODI 3B, depth-2 and depth-3 values first read at passes 3-5 have onsets at layers 19-22 of 28.

### t52 scoring, 2026-11-18 19:50:05 PST (from `date`; JSON ../codi_audit_artifacts/t52/depth_robust.json)
Rules: 1,301 parsed items; 919 clean (222 with an ambiguous number, 101 with an unexplained operand, 93 implicit). loose and resolved
give the same item depth on every item; both agree with strict on .879.
- P1 held: SIM-CODI 3B depth-4 crossings 2.02 (loose, resolved; n 102) and 2.01 (clean; n 72) vs SIM-CODI 1B 2.56 / 2.67.
- P2 held: depth-3 steps per pass, Llama CODI 1B vs SIM-CODI 1B: strict 1.29 vs 1.07, loose/resolved 1.29 vs 1.09, clean 1.30 vs 1.10.
- P3 held: GPT-2 SIM-CODI 1.02 / 1.03-1.04 and SIM-CODI 1B 1.03-1.04 / 1.07-1.10 at depth 2 / 3 under every rule.
- P4 missed: depth-1 accuracy stays below depth-2 accuracy for all 5 models under resolved and clean (clean: e.g. SIM-CODI 3B .50 vs
  .79, n 36 depth-1 items). The clean depth-1 items are mostly rationales that skip work (e.g. <<4*4=16>> for 60 days / 15 servings
  = 4 cartons, or an algebra problem written as <<4*4=16>> <<8=8>>); the unexplained operand there is a small number, which the
  constant list lets through. Depth 1 under-describes these items; the bin should be dropped or footnoted.
- P5 held (barely): clean vs all, readable at refresh d - 1: GPT-2 CODI .58/.77 vs .60/.73; Llama CODI 1B .90/.98 vs .92/.98; SIM-CODI
  1B .55/.66 vs .58/.71 (depth 3 differs by .0497); SIM-CODI 3B .70/.90 vs .69/.92.
- P6 held: clean pooled same-pass child later / earlier 132/49, sign p = 2.7e-10.
- P7 (exploratory): operations per pass on resolved start-of-path pairs, depth 2 / 3 / 4: GPT-2 CODI 1.09/1.11/1.48, GPT-2 SIM-CODI
  1.10/1.09/1.46, Llama CODI 1B 1.29/1.37/1.77, SIM-CODI 1B 1.14/1.13/1.27, SIM-CODI 3B 1.26/1.15/1.58. A depth-D path carries about
  D + .2 to .6 operations.
Reading: the hop-count, readability and layer-lens conclusions do not depend on the depth rule. Accuracy at depths 2-4 rises by .01-.05
on clean items, keeping the same order across models. Only the depth-1 accuracy bin is unreliable.

### t51b / t52b: Coconut checkpoints in the per-layer lens and the depth-rule check, written 2026-11-18 20:04:16 PST (from `date`) before any t51b / t52b output
Models: GPT-2 Coconut (connordilgren/gpt2-gsm8k-coconut) and GPT-2 SIM-Coconut (internlm/SIM_COT-GPT2-Coconut, checkpoint_28), 12 layers,
fp32, CPU, adapters.GridCoconut. t51b: gsm8k_layer_lens.py extended to these (pass 0 = the <|start-latent|> position; the latent fed
back is the post-ln_f last hidden state, so the pass output is read as is). t52b: gsm8k_depth_robust.py with both added (t31 / t36
pairs, t39 accuracy).
Seen before writing: the strict-rule hop counts (Coconut 1.24 / 1.71 crossings at depth 2 / 3, steps per pass .93 / 1.16, depth 4 n < 15;
SIM-Coconut 1.43 / 2.43 / 3.29, steps per pass .89 / .91 / .95), t39 accuracy .331 / .447, the t26 Coconut READ=z file (refresh-0 level
about .5, chance about .3), and all t51 / t52 outputs for the CODI-family models.
Predictions:
- t51b P1 (sanity): pass 0, layer 0 reads at most chance + .02 at every depth.
- t51b P2: depth-2 share with s* = pass 0 below GPT-2 CODI's .30 for both Coconut models.
- t51b P3: same-pass child onset later than parent more often than earlier in each model (no test threshold; pairs may be few).
- t51b P4: depth-1 values with s* = pass 0 have median onset at layer 9 or later (of 12), as in the CODI-family models.
- t51b P5 (sanity): accuracy in the natural run within .01 of t39 (.331 / .447).
- t52b P1: SIM-Coconut steps per pass at depth 2 and 3 stay below 1.00 under every rule (cells n >= 15).
- t52b P2: Coconut's steps per pass at depth 3 stay above SIM-Coconut's under every rule.

### t51b / t52b scoring, 2026-11-18 20:15:52 PST (from `date`; JSON ../codi_audit_artifacts/t51/layer_lens_gsm8k_{coconut,simcoconut}.json, ../codi_audit_artifacts/t52/depth_robust_b.json)
Config error, fixed before scoring: the first SIM-Coconut layer-lens run used 6 latents (GridCoconut default); its accuracy was .382, so
P5 flagged it. SIM-CoT's GSM8K config uses 10 (c_thought 2 x max_latent_stage 5), as t36 and t39 did. Rerun with 10 latents; the 6-latent
files are kept as layer_lens_gsm8k_simcoconut_K6_wrong.{json,log}. Scores below are for the 10-latent run.
- t51b P1 held: pass 0, layer 0 reads 0 at every depth in both models.
- t51b P2 held: depth-2 share with s* = pass 0: Coconut .13, SIM-Coconut .06 (GPT-2 CODI .30).
- t51b P3 held: same-pass child later / earlier / tie: Coconut 44/10/16 (p 1.7e-6), SIM-Coconut 42/17/16 (p 7.7e-4).
- t51b P4 held: depth-1 values with s* = pass 0 have median onset 12 of 12 in both (latent passes: 10-11 of 12).
- t51b P5 held for Coconut (.331 = t39); SIM-Coconut .447 = t39 after the fix (the 6-latent run missed, .382).
- t52b P1 held: SIM-Coconut steps per pass at depth 2 / 3: .88-.89 / .91-.94 under every rule.
- t52b P2 held: depth-3 steps per pass, Coconut vs SIM-Coconut: 1.14-1.16 vs .91-.94 under every rule.
Exploratory, not predicted: both Coconut models read values at almost every depth somewhere in the run (.79-.99 vs .25-.55 for the
CODI family at depth >= 2), on a fixed schedule. SIM-Coconut: depth 1 at pass 1 (.76), depth 2 at pass 3 (.71), depth 3 at passes 5-9
(.72-.78), depth 4+ at passes 7-9 (.75-.91); even passes read near chance. Coconut: depth 1 at pass 1 (.77), depth 2 at pass 4 (.77),
depth 3 at pass 6 (.84).

### t53: prefix substitution and latent distances: do A's and B's latents before the edit agree, and where does the edit enter the loop? written 2026-11-23 19:01:48 +07 (from `date`) before any full t53 output
Script scripts/necessity/prefix_sub.py. Models: rule-gated CODI and start-last CODI (RuleChain valid L2-L6, t31 pairs, CAP 25 per
(level, edit) cell, seed 53); released GPT-2 CODI and SIM-CODI (GSM8K, t31 short + long twin pairs; 414 and 387 pairs). Only pairs
answered right on both questions. For each pair: d_j = ||z_j(A) - z_j(B)|| for the natural inputs z_0 (question pass output) .. z_K,
normalised by ref_j = ||z_j(A) - z_j(X)|| with X another item's recipient; and for c = 0..K a prefix run M'_0 = B's question with B's
natural z_0..z_{c-1} and A's natural z_c..z_{K-1}, refreshed as in hops.py; h_c = first n from which the answer is B's own.
S_c = h_0 - h_c. Entry e = smallest c with S_c > 0 (the first installed latent that carries the edit), e = K + 1 if none.
Edit position: RuleChain i (start = 0, rule i = i); GSM8K k = the gold step index of the edited step; distance s as gsm8k_depth.py (strict).
Seen before writing: smoke runs only (gated n = 8: mean h_c 1.00 1.00 0.00 ...; median d_j/ref_j .00 .91 .89 .55 .45 .45 .47;
GSM8K CODI n = 4: mean h_c 1.75 1.00 1.00 .25 .25 0 0; d_j/ref_j .74 .23 .79 .24 2.54 .27 2.50). Plus everything in t26-t52, notably
the within-item slope +.16 (CODI) / +.12 (SIM-CODI), the route split (single-latent recurrent swaps move <= .04 on GSM8K), early
arrival in READ=h (.60 / .73 at refresh d - 1).
Purpose: whether the flat within-item hop count on GSM8K comes from early latents already carrying the edit (copies / redundancy)
rather than from a chain, with a positive control where the chain is exact (rule gate) and a standard model of known distance.
Predictions:
- P0: checks, every model: h_0 equals the t31 hop count, own answers reproduce, and the c = K run gives B's answer, each on >= .99 of pairs.
- P1 (positive control, rule-gated, rule edits): e - i takes one value on >= .90 of pairs, and the within-level slope of e on i is in [.8, 1.2].
- P2 (positive control, rule-gated, rule edits): median d_j/ref_j <= .05 for every j < e and >= .20 for j = e.
- P3 (standard RuleChain, start-last CODI, known distance): the within-level slope of e on i is < .30.
- P4 (GSM8K, CODI and SIM-CODI): the within-item slope of e on k is < .30 (a step-per-pass chain would give about 1 or more).
- P5 (GSM8K, CODI and SIM-CODI): for late edits (k >= 2), mean S_1 >= .30, i.e. B's question-pass output alone already saves refreshes.
- P6 (GSM8K, CODI and SIM-CODI): for late edits (k >= 2), median d_0/ref_0 >= .25: A's and B's latents differ before the edited step.
- P7 (exploratory): S_c curves by edit position and by item depth; whether start-last CODI's pattern matches GSM8K's.

### Correction to t53 P2, appended 2026-11-23 19:02:34 +07 (from `date`), before reading any full t53 output (the queue is running; no log or JSON opened)
e counts installed latents (z_0 .. z_{e-1}), so the first latent that carries the edit is z_{e-1}, not z_e. P2 as written compares the
wrong index. Corrected P2: for rule-gated rule edits, median d_j/ref_j <= .05 for every j < e - 1, and >= .20 for j = e - 1.
No other prediction changes.

### t54: the model's own route: depth and edit distance from the CoT the released checkpoints write themselves, written 2026-11-23 19:53:48 +07 (from `date`) before any full t54 output
Script scripts/necessity/cot_route.py (generation), cot_route_score.py (scoring). Released GPT-2 CODI and SIM-CODI write their CoT in
teacher mode (same weights; tok(question) then greedy CoT, base vocabulary only) for all 1,319 GSM8K test items and for the R and D
questions of every t31 pair answered right on both sides. CODI trains the teacher with the last step dropped (include_last_cot False),
so the written CoT ends one step early; the parse (gsm8k_depth.depth_of on the written steps, with the model's own answer) then adds
the implicit final step over the sink steps. Model-route depth = that depth on R's own CoT. Model-route edit distance = longest path
(gsm8k_depth.down) from the written steps of R's CoT whose expression contains the edited number to the answer; 1 if no written step
contains it (the number enters only the unwritten final computation). Hop counts are the t31 ones (QUAL own) on both-right pairs.
Seen before writing: a 6-item CPU smoke of CODI (4 of 6 answers right; e.g. '<<16-3-4=9>>The answer is: 18', the 9*2 step not
written). No other t54 output. All of t26-t53 setup; t53 RuleChain runs have finished but their JSON and logs are unopened.
Purpose: control for the model's real distance (instead of the gold solution's) in the within-item hop-count slope.
Predictions:
- P0: >= .95 of generations contain "The answer is: <number>"; CODI teacher accuracy on the 1,319 test items in [.35, .50].
- P1: model-route item depth equals the gold strict depth on between .40 and .80 of test items (they differ often but not mostly).
- P2: within-item slope of hops on model-route edit distance, both-right pairs, < .30 for CODI and for SIM-CODI.
- P3: start-of-path crossings by model-route depth rise: mean at model-route depth 3 minus mean at depth 2 >= .50 (CODI and SIM-CODI;
  start of path = model-route edit distance equals model-route depth; cells n >= 15).
- P4: pairs whose edited number enters only the unwritten final step (distance 1): mean hops >= 1.0 for CODI and SIM-CODI (a chain
  whose answer pass reads late operands from the question would give about 0, as SIM-CODI does for last-rule edits on RuleChain).
- P5 (exploratory): agreement of model-route and gold edit distance; with t53 done, the relation of the entry point e to the
  model-route position of the edit.

### t54 scoring, 2026-11-23 20:11:20 +07 (from `date`; JSON ../codi_audit_artifacts/t54/score.json)
- P0 held: parse 1.00 for both; teacher accuracy CODI .462, SIM-CODI .470.
- P1 held: model-route depth equals gold strict depth on .60 / .61 of items (within one: .90).
- P2 MISSED (both): within-item slope of hops on model-route edit distance +.88 [+.63, +1.18] for CODI (15 items, 50 pairs) and
  +.87 [+.62, +1.00] for SIM-CODI (11 items, 43 pairs), against +.16 / +.03 on the labelled gold distance for the same pairs.
- P3 held (both): start-of-path crossings by model-route depth 2 / 3 / 4: CODI 1.08 / 1.95 / 2.43; SIM-CODI 1.14 / 2.01 / 2.60.
- P4: CODI missed (.90, n 10), SIM-CODI held (1.18, n 11); both cells tiny.
- P5 (exploratory): model-route and labelled gold edit distance agree on .74 / .72 of pairs.
Post hoc (not pre-registered), after reading the P2 items: the slope difference is mostly not the model's route but the pair labels.
The t22 twin pairs carry the distance of one gold step ('step k'), but the edited question number is often used by several gold
steps (e.g. '<<17*2=34>> <<17+34=51>>', labelled as an edit of step 1). Recomputing the GOLD distance as the longest path from any
gold step whose expression contains the edited number: it differs from the label on .19 (CODI) / .20 (SIM-CODI) of both-right pairs,
and the within-item slope becomes +.84 [+.59, +1.26] (CODI, 13 items, 44 pairs) and +1.00 [+.73, +1.32] (SIM-CODI, 8 items, 25 pairs);
on all qualified CODI pairs +.36 [+.20, +.53] (106 items) against +.16 labelled. So the flat within-item slope reported since t42
is largely a labelling artefact; on correct pairs the count follows the edit's distance at about one crossing per step, on few items.
Script: scripts/necessity/gsm8k_anyuse_distance.py.

### t55: does chaining grow with training? Hop count over start-last CODI epochs and on the 40-epoch round-1 CODI, written 2026-11-23 20:52:59 +07 (from `date`) before any t55 output
Script scripts/necessity/hops.py unchanged (QUAL=own, RuleChain valid L2-L6, CAP 80), with env CODI_CKPT (new in adapters.RuleChainCODI)
pointing at another checkpoint of the same model family: start-last CODI checkpoint-8910 / 11880 / 14850 / 17820 (epochs 6 / 8 / 10 /
12; 17820 = t31's model, a reproduction check) on start-last data, and round-1 CODI gpt2_depth2_codi_bs64/checkpoint-31280 (start-first,
canonical L1-12 data, 40 epochs, lr 3e-3; test .401 vs CODI-0 .405) on start-first L1-7w valid (model key stab).
Seen before writing: b18_track per-epoch accuracy and all-latent ablation (start-last CODI's ablation drop at L3 grows from .04 at
epoch 5 to .45 at epoch 12, L4 .01 to .28; CODI-0 gap at L3 +.21 at epoch 12; still rising at epoch 12); round-1 latent ablation
(single latents cost <= .008, all six .22, latents near-copies); t31 epoch-12 hops. No hop count on any other epoch or on round 1.
Metrics on pairs answered right on both sides: A = mean hops at s = 3-4 pooled; B = across-pair OLS slope of hops on s over s = 1-5.
Predictions:
- P0: the epoch-12 run reproduces t31 startlast hop counts on >= .99 of pairs.
- P1: start-last A rises with training: A(epoch 12) - A(epoch 6) >= .30 (cells n >= 15).
- P2: start-last B rises with training: B(epoch 12) - B(epoch 6) >= .10.
- P3: round-1 40-epoch CODI: A <= .50 and B <= .20 (40 epochs do not make start-first CODI chain).
- P4 (exploratory): epochs 8 and 10; whether A or B is still rising between epochs 10 and 12.

### t53 note, 2026-11-23 20:54:34 +07 (from `date`): the GSM8K SIM-CODI run was stopped by decision of the team at pair 241 of 387 (no JSON written,
the script saves only at the end). P4-P6 are scored for GSM8K CODI only; SIM-CODI is not scored.

### t55 scoring, 2026-11-24 09:49:29 +07 (from `date`; JSON ../codi_audit_artifacts/t55/score.json; the queue paused overnight 21:49-09:04, Mac asleep)
- P0 held: epoch-12 hops equal t31 on 2000 / 2000 pairs.
- P1 MISSED (cell size): A = 1.14 (n 7) / 1.60 (10) / 1.52 (31) / 1.47 (51) at epochs 6 / 8 / 10 / 12; the epoch-6 cell is below n 15, and
  from epoch 8 or 10 on A does not rise. By s, both-right pairs: s = 2 1.00-1.05 at every epoch; s = 3 1.67 / 1.62 / 1.57 (epochs 8 / 10 / 12);
  s = 4 1.40 / 1.09 (epochs 10 / 12). Accuracy rises over the same epochs (valid L3 .14 / .24 / .41 / .52), crossings per pair do not.
- P2 MISSED: B falls .83 -> .55; B is driven by which s have correct pairs (only s <= 2 early), so it is not comparable across epochs.
- P3 MISSED in level, direction held: round-1 40-epoch start-first CODI uses about one crossing at every distance (s = 2..6: 1.05 / 1.12 /
  1.14 / 1.31 / 1.25; A 1.12, B .33), not ~0 as predicted; flat in s, consistent with its near-copy latents.

### t56: does the RuleChain pattern survive inference-only distribution shift? written 2026-11-24 13:32:52 +07 (from `date`) before any full t56 output
Advisor feedback (Nov 24): RuleChain and the hop count need validity checks; shift the distribution at inference. Scripts
scripts/necessity/validity_sets.py (sets) and validity.py (mode sets). Models: rule-gated, SIM-CODI, start-last CODI (start-last order),
SIM-CODI start-first, relay, stabilized (start-first); CODI-0 start-last and stabilized (accuracy only). Sets, L2-L7, 300 items per level,
generator filters applied, none in train: orig (first 300 canonical test items per level), rw (orig items, each rule type reworded, same
'Rule i:' / 'Start with x.' frame), hc (every constant outside its training range: P a 26-49; T t 81-95, a, b 21-35; D t, b 51-70),
hc1 (one rule per item with held-out constants), sv (start value 100-115, never seen; rule 1 brings it into 0..99), l8 (L = 8, run with
K = 6, 7, 8 latent passes at inference). Pairs: QUAL=own, one structural edit per (recipient, position) inside the set's ranges, CAP 40
per (level, position); hop count as hops.py; number-token logits stored per refresh.
Seen before writing: smoke runs on rule-gated CODI only (24 items per level at L3, L4; 16 at L8; 36 pairs per set): accuracy orig 1.00 /
1.00, rw .29 / .13, hc .29 / .17, sv .04 / .00 (L3 / L4); l8 .00 / .19 / .88 at K = 6 / 7 / 8; mean hops (L3 + L4 pooled, all edit
positions) orig 1.69, rw 1.91, hc 1.79, sv 1.76, l8 2.73 / 3.00 / 3.67. Everything in t26-t55. hc1 was added after the smoke run.
Predictions (qualified pairs; cells n >= 15; "start edits" = edits of the start value):
- P0: M_K equals B's own run on >= .99 of pairs, every model and set; orig accuracy within .05 of the paper's test accuracy at every L.
- P1 (chaining survives shift): for rule-gated CODI, SIM-CODI and SIM-CODI start-first, on each of rw, hc, hc1, sv, mean hops on start
  edits is >= L - 1 at every L in 2..6.
- P2 (exact schedule survives shift): rule-gated CODI, rule edits: hops = L - i on >= .90 of pairs, on each of rw, hc, hc1, sv.
- P3 (standard stays low): start-last, relay and stabilized CODI: mean hops on start edits <= 2.5 at every L in 2..6, on every set.
- P4 (dose): accuracy pooled over L3-L5 is higher on hc1 than on hc, for each of the six looped models.
- P5 (more passes at inference): at L8, rule-gated CODI and SIM-CODI score >= .70 with K = 8 and <= .30 with K = 6; start-last, relay
  and stabilized CODI gain <= .10 from K = 6 to K = 8.
- P6 (exploratory): accuracy by set for every model incl. CODI-0 (single-pass reach under shift); whether the chaining models lose
  less accuracy than the standard models; hop counts by s on each set; at l8 / K = 8 start-edit hops of the chaining models.

### t57: robustness of the hop count itself, written 2026-11-24 13:32:52 +07 (from `date`) before any full t57 output
Script scripts/necessity/validity.py (mode robust). Same six looped models. On each model's t31 pairs (QUAL=own, qualified): repro (the
count again), swap (B as recipient, A as donor; target A's own answer), unrelated (M_0 seeded with the natural latents of another
recipient X of the same level; target B's own answer; qualifies if B's answer differs from X's and from A's). On new valid-split pairs,
L2-L6, CAP 25 per (level, position): small edits (|delta| <= 2 on the edited number) and large edits (|delta| 10-40).
Margin counts from the stored logits, m_n = logit(target) - logit(reference answer), u_n = (m_n - m_0) / (m_K - m_0):
margin hop = first n from which m_n > 0 for every later n; onset = first n with u_n >= .1; half hop = first n from which u_n >= .5 stays.
Seen before writing: smoke on rule-gated CODI (24 t31 pairs, 16 pairs per size): repro equals t31 on 24 / 24; mean hops repro 1.71,
swap 1.83, unrelated 1.52, small 1.44, large 1.44 (n tiny). Everything in t26-t55.
Predictions:
- P0: repro equals the t31 count on >= .99 of pairs, every model.
- P1 (symmetry): |swap - repro| <= 1 on >= .90 of pairs, every model; equal on >= .95 for rule-gated CODI.
- P2 (threshold): margin hop equals the argmax hop on >= .90 of pairs, every model. For rule-gated CODI and SIM-CODI, onset equals the
  argmax hop on >= .80 of pairs (no part of B's information reaches the answer before its crossings exist). For start-last, relay and
  stabilized CODI, mean half hop <= 2.5 in every s cell (s = 2..6).
- P3 (edit size): |mean hops large - mean hops small| <= .3 in every (model, s) cell with n >= 15 in both, and <= .2 pooled over s.
- P4 (donor): rule-gated CODI and SIM-CODI: mean unrelated-donor hops by level within .5 of their t31 start-edit hops at that level;
  start-last, relay and stabilized CODI: mean unrelated-donor hops <= 2.5 at every level.

### t56 / t57 scope change and wide-set addendum, written 2026-11-24 13:40:59 +07 (from `date`)
Scope (team decision, before any output of the affected models): t56 and t57 run only on the three start-last models of the main
figures, rule-gated CODI, SIM-CODI and start-last CODI, plus start-last CODI-0 (t56 accuracy only). SIM-CODI start-first, relay,
stabilized and stabilized CODI-0 are dropped; every prediction that names them is not scored. Seen so far of the full runs: the
queue log of rule-gated CODI's sets run, which printed its accuracy on orig (L2-L6 1.00, L7 .987) and hc (L2-L7 .44 / .32 / .26 /
.22 / .19 / .12) before this note; no hop count of a full run.
Addendum, set wide (validity_sets.py build_wide, validity.py mode wide): training-range rules, every value (start included) in
0..115, at least one value >= 100 per item, filters as the other sets; 300 items per level L2-L7 (133 of 1800 answers >= 100, 61 at
L2). The answer readout is extended to the 116 single tokens 0..115 for this set only. Pairs as t56, edits inside 1..115.
Seen before writing: smoke on start-last CODI (24 items at L3 and L4: accuracy .00 / .04; 36 pairs, mean hops 1.55).
Predictions:
- W0: M_K equals B's own run on >= .99 of pairs; every number 0..115 decodes to its own single token.
- W1: accuracy on items whose answer is >= 100 is <= .10 for every model (no training answer was >= 100).
- W2: for rule-gated CODI and SIM-CODI, accuracy on wide items whose answer is <= 99 is at least .30 below their orig accuracy,
  pooled over L2-L5 (unfamiliar intermediate values break the computation, as the sv smoke suggests).
- W3: for rule-gated CODI and SIM-CODI, mean hops on start edits >= L - 1 at every L in 2..6 (cells n >= 15), as t56 P1.
- W4 (exploratory): accuracy by whether and where the first value >= 100 occurs; start-last CODI's hop counts by s.

### t56 + wide scoring, 2026-11-24 15:14:15 +07 (from `date`; JSON ../codi_audit_artifacts/t56/score_t56.json; scorer scripts/necessity/validity_score.py)
Scored models: rule-gated CODI, SIM-CODI, start-last CODI, start-last CODI-0 (scope note). Predictions naming simsf / relay / stab: not scored.
- P0 held: M_K equals B's own run on every pair of every set (1320 / 1320 per set; 360 / 360 at l8); orig accuracy within .032 of the test record at every L.
- P1 held: rule-gated and SIM-CODI start-edit hops >= L - 1 in every L2-L6 cell of rw, hc, hc1, sv (lowest: rule-gated sv L2 1.23, SIM-CODI rw L6 5.73).
- P2 MISSED as scored (exact share .77-.78 on every set, orig included), because the predicted L - i is wrong at L7. Post hoc: at L2-L6
  the count is exact on 800/800 (orig), 551/562 (rw), 695/696 (hc), 767/768 (hc1), 578/578 (sv) rule edits; at L7 every edit of rules 1-6
  gives L - i - 1 (280/280, 183/184, 238/238, 272/272, 198/198): with K = 6 and seven rules, rule 6 is computed in the last latent pass,
  whose output is read by attention and never fed back. l8 with K = 8: 248/313 exact under L - i, the rest one lower (exploratory).
- P3 held: start-last CODI start-edit hops <= 2.07 in every L2-L6 cell of every set.
- P4 held: L3-L5 accuracy hc1 > hc for every model (rule-gated .73 vs .27; SIM-CODI .64 vs .14; start-last .16 vs .04).
- P5: rule-gated held (L8: K = 6 .00, K = 8 .76); SIM-CODI MISSED (K = 6 / 7 / 8: .00 / .06 / .02, although its start-edit count at K = 8 is
  7.89); start-last held (.06 at every K).
- P6 (exploratory): accuracy collapses under rw, hc, sv for every model, CODI-0 included (rule-gated L2-L7: rw .26-.04, hc .44-.12, sv
  .00-.06; SIM-CODI rw .23-.02, hc .33-.05, sv .03-.08); hc1 is flat in L for the chaining models (rule-gated .68-.76, SIM-CODI .60-.66),
  falls with L for start-last (.66 -> .06). The crossing pattern of every model survives every shift (as P1 / P3).
- Wide: W0 held (1320 / 1320; 0..115 decode to own single tokens, asserted). W1 held (answers >= 100: .00 / .03 / .01, n 133). W2 held
  (rule-gated 1.00 -> .04, SIM-CODI 1.00 -> .07 on answers <= 99, L2-L5). W3 held for both chaining models (rule-gated 1.20 / 2.10 / 3.15 /
  4.13 / 5.10, SIM-CODI 1.90 / 2.98 / 3.95 / 4.97 / 5.95 at L2-L6). W4: start value >= 100 .01-.03 vs only a later value >= 100: rule-gated
  .13, SIM-CODI .26, start-last .07.
Reading: the serial structure (crossings per rule) is robust to every inference-time shift; per-step accuracy is not. The models
learned template- and range-specific step computation; errors localise at the unfamiliar rule (hc1 flat in L).

### t57 scoring, 2026-11-24 21:29:08 +07 (from `date`; JSON ../codi_audit_artifacts/t56/score_t56_t57.json; scope: rule-gated, SIM-CODI, start-last CODI)
- P0 held: repro equals the t31 count on 1.00 of pairs (1978 / 1966 / 1735).
- P1: |swap - repro| <= 1 on 1.00 (rule-gated), 1.00 (SIM-CODI), .857 (start-last: MISSED); equal on .938 for rule-gated (MISSED, >= .95
  predicted). Means agree exactly (2.15 vs 2.16; 2.77 vs 2.77; 1.34 vs 1.34): symmetric on average, noisy per pair for start-last.
- P2: margin hop equals argmax hop on .984 / .975 (held) and .657 for start-last (MISSED); where they differ the margin hop is always
  LOWER (32 / 49 / 595 pairs), never higher. Onset equals argmax on .93 for rule-gated (held) and .53 for SIM-CODI (MISSED); onsets are
  earlier, never later, for s > 1 (rule-gated 111, SIM-CODI 576, start-last 584 pairs). Start-last half hop 1.04-1.19 at every s (held).
  Reading (post hoc): the argmax count is an upper bound, as the paper states; for start-last CODI the margin-based count is about 1.
  SIM-CODI's onset one refresh before the flip on 37% of pairs fits its count lying one above the rule gate's (B's information
  outweighs A's one refresh after it arrives).
- P3: large - small edits: rule-gated +.01, SIM-CODI +.10 (held); start-last +.28 pooled and +.47 / +.33 at s = 3 / 4 (MISSED): larger
  edits take more crossings in the standard model.
- P4 held: unrelated-donor counts within .10 of the t31 start-edit counts at every level for rule-gated and SIM-CODI; start-last <= 1.96.

### t58: single-pass reach of GPT-2 on two candidate third tasks, written 2026-11-24 22:17:14 +07 (from `date`) before any full t58 output
Script scripts/necessity/third_task_pilot.py. Pretrained HF gpt2, full fine-tune WITHOUT CoT (question -> one answer token, loss on
that token), 4000 training items per k = 1..8 pooled, 5 epochs, lr 5e-5, batch 16, 300 test items per k; accuracy over the task's
label set; train accuracy per k logged. Tasks:
  sn   S5 word problem over 6 fixed generator permutations (verified to generate S5; single-token words), answer = label of the
       product (120 labels). Chance 1/120.
  brew in-context permutations: 10 colours, 3 ingredients per item (random derangements printed as a rule table), start colour given
       last, k stirs; all path states distinct, last-ingredient guess rejected, 8-reordering order filter; our own generator. Chance .10.
Seen before writing: smoke runs only (40 items per k, 1 epoch): sn all k <= .013 (before the generator-set change: k = 1 .977,
others <= .017); brew all k .07-.12. Plus the t56 SwapChain / nocot-bench CPU audits (a small from-scratch transformer learns neither
brew nor config_patch, h = 1 included).
Metric: reach = interpolated k at which final-epoch test accuracy crosses the midpoint between chance and accuracy at k = 1.
Reference: RuleChain start-last CODI-0 reaches about 2.6 by this rule (.98 at L2, .23 at L3), start-first CODI-0 about 5.
Decision rule (fixed now): a task PASSES if accuracy at k = 1 >= .90 and 1.5 <= reach <= 3.5. Choose sn if it passes; else brew if it
passes; else neither (no third task; GPU budget to the 40-epoch pair and seeds). If accuracy at k = 1 < .90 (format or lookup not
learned), the task is INCONCLUSIVE and is rerun once with 10 epochs; the decision then uses that run.
Predictions:
- P1 (sn): accuracy at k = 1 >= .90 and reach >= 4 (the fixed generators compose cheaply in log depth): sn does not pass.
- P2 (brew): accuracy at k = 1 >= .90 and reach <= 3.5: brew passes.
- P3 (exploratory): train vs test accuracy by k (memorisation at small k for sn, where test items repeat training items).

### t53 scoring, 2026-11-24 22:25:59 +07 (from `date`; JSON ../codi_audit_artifacts/t53/score.json; GSM8K SIM-CODI stopped, not scored)
- P0 held: h_0 equals t31, own answers reproduce and the c = K run gives B's answer on 1.00 of pairs (rule-gated, start-last, GSM8K CODI 414).
- P1 MISSED as scored (rule-gated: mode offset 1 on .80, slope 1.29) and P3 MISSED as scored (start-last slope 1.11): pairs with h_0 = 0
  cannot save a refresh and were coded e = K + 1, which drives both slopes (coding flaw in the prereg, noted on 2026-11-23).
- P2 held: rule-gated d_j / ref_j = .00 for every latent before the carrier, .75 at the carrier.
- P4 MISSED (GSM8K CODI entry slope on gold step index +.56 [+.25, +.86]); P5 MISSED (S_1 = .26, n 38); P6 MISSED (d_0 / ref_0 = .05).
Exploratory (post hoc, pairs with h_0 >= 1): rule-gated within-level slope of e on edit position .96, e = rule + 1 on 375 of 375 rule
edits; start-last slope -.09 (mean e 3.9-4.3 at every position, n 129). GSM8K CODI on the any-use distance: entry -2.2 passes per step
[-3.4, -1.7] (11 items), hops +.86 per step [+.60, +1.30] (13 items); edit-specific information in z_0, z_2, z_4 (~.8 of reference)
and barely in z_1, z_3 (.15-.20).
