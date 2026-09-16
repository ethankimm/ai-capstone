# Milestone 1 Proposal — Do Different Latent Scratchpads Think Alike?

**Group:** Ethan Kim, Henning Lindig, Kevin Nguyen
**Track:** Research

This is the official milestone 1 proposal (submitted Sep 2026), committed here so every
collaborator — and every AI session working in this repo — reads the same plan. Read it
before proposing new experiments. Supersedes any earlier draft.

## Research area and initial topic direction

We are working in interpretability, focused on latent reasoning mechanisms. There is
growing research interest in moving from explicit Chain-of-Thought to latent space
reasoning for richer cognitive representations and improved performance. While various
paradigms have been suggested, such as filler tokens, hidden state-based thinking, and
recurrent networks, the underlying mechanics are poorly understood. We look to study how
different latent reasoning methods operate and whether their internal mechanisms are
transferable between them.

## Success criteria and evaluation plan

Our success criteria is extracting and verbalizing intermediate latent reasoning across
mechanisms; since this is a long-shot goal, we plan for negative/null findings (more
details in the final section).

We plan to accomplish this goal in the following steps:

1. Reproduce previous papers' implementations of different latent reasoning mechanisms,
   and verify their accuracies (starting with GPT-2 small and GSM8K-aug)
2. Train a decoder to extract intermediate latent reasoning; consider individual vs
   shared decoders
3. Use counterfactual interventions to determine causality of a decoded intermediate
   state
4. Train a lightweight mapping to accommodate different mechanism architectures
5. Transplant intermediate latent reasonings and determine whether the downstream
   reasoning holds true

With desired results, we aim to answer the following research questions:

1. What task-specific intermediate states are encoded in each latent scratchpad?
2. Does intervening at a decoded state produce the predicted downstream computation?
3. Can an internal reasoning state from one mechanism be transplanted into another?

We use the following metrics for evaluation: intermediate-state decoding accuracy,
necessity under early termination (previous work has shown that certain datasets are not
relevant for latent reasoning, e.g. PrOntoQA and ProsQA), counterfactual intervention
accuracy, and cross-mechanism transplant success.

## Scope and key deliverables

We scope our project to the primary mechanisms of inducing latent reasoning following a
recent survey paper: token-wise/horizontal level and layer-wise/vertical level. From
these categories, we choose one or two mechanisms from each in order to compare within
and across categories. Since reproducing results is often challenging, we limit ourselves
to the frontier mechanisms.

Following previous work, we also scope our problem to an augmented version of GSM8K, and
will expand to a second dataset if we achieve our desired findings for generalization.
Our key deliverable is a research manuscript (introduction, relevant background, methods,
evaluation, discussion, and conclusion) with figures included comparing the different
latent reasoning mechanisms.

## Timeline and faculty alignment

We are ordering the work by highest risk and lowest time for completion. We will be first
recreating or replicating the findings for the recurrent depth paper, explicit chain of
thought, filler tokens, and CODI. We then test whether latent reasoning survives
transplantation, both sequential-to-sequential and sequential-to-recurrent-depth. We will
survey adjacent questions in parallel and redirect if the results point elsewhere,
bringing the updated plan to the second faculty office hour and checking in with staff
regularly.

**Sep 25, 2026 (Meeting with a TA — Phase 2)**
- Literature review largely completed
- Latent reasoning taxonomy and mechanistic interpretability methodologies to be used
  identified
- Set up the first version of training and eval infrastructure. Involves securing GPU
  resources, and building out a shared git repository
- Initial experiments reproducing CoT baseline, filler tokens, CODI, and recurrent depth
  from their respective papers on GPT-2 small on GSM8K-aug

**Oct 9, 2026**
- Have attempted decoding latent representations per model using linear probes,
  non-linear probes, J-lens, logit lens

**Oct 16, 2026**
- Applied causality tests on extracted intermediate reasoning such as activation
  patching, ablations, EAP/ReIP, etc.
- Train a lightweight mapping to accommodate different mechanism architectures and
  experiment with transplanting representations across models

**Oct 30, 2026**
- Sweep experiments across most relevant modes of: additional latent reasoning models
  (Coconut, Soft Thinking, System-1.5 Reasoning, PCCoT), benchmarks (GSM8K-aug-NL,
  HotpotQA, graph reasoning), and mechanistic interpretability approaches

**Nov 16, 2026**
- Full draft report ready

**Early December**
- Iterated on feedback from mentor to produce polished final report

---

## References (added; not part of the submitted text)

Mechanisms in scope:

- Filler tokens — Pfau, Merrill & Bowman 2024, *Let's Think Dot by Dot*,
  [arXiv:2404.15758](https://arxiv.org/abs/2404.15758)
- CODI — Shen et al. 2025, *Continuous Chain-of-Thought via Self-Distillation*,
  [arXiv:2502.21074](https://arxiv.org/abs/2502.21074),
  [github.com/zhenyi4/codi](https://github.com/zhenyi4/codi)
- Recurrent depth — Geiping et al. 2025, *Scaling up Test-Time Compute with Latent
  Reasoning*, [arXiv:2502.05171](https://arxiv.org/abs/2502.05171),
  [github.com/seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)

Adjacent / Oct 30 sweep candidates:

- Coconut — Hao et al., *Training LLMs to Reason in a Continuous Latent Space*,
  [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- Pause tokens — Goyal et al., *Think Before You Speak*,
  [arXiv:2310.02226](https://arxiv.org/abs/2310.02226)
- Dilgren & Wiegreffe, *Are Latent Reasoning Models Easily Interpretable?*,
  [arXiv:2604.04902](https://arxiv.org/abs/2604.04902)
- *Looped Transformers under the Jacobian Lens* — loop iterations largely overwrite
  rather than accumulate state; read before choosing how to decode recurrent-depth
  intermediate states
- *Reading Between the Dots: Decoding Hidden Computation across Filler Tokens* — logit-lens
  pipeline for predicting internal reasoning from filler positions

Dataset: GSM8K-Aug (`zen-E/GSM8k-Aug`, calculator rationales) and GSM8K-Aug-NL
(`zen-E/GSM8k-Aug-NL`, natural-language rationales) — the GPT-2-scale training set used
by CODI and Coconut.
