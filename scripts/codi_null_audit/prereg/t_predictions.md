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
