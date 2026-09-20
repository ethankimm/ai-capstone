import json

from latentreasoning.data.gsm8k_aug import load_local_sample
from latentreasoning.eval.counterfactual import (
    counterfactual_answer, num_equal, parse_steps, score_patch, score_patch_unaligned,
)
from latentreasoning.runlog.manifest import RESULTS_DIR

POSITIVE_CONTROL_RUN = "20260919-185332_explicit_cot_patch-positive-control"


def _rationale():
    return "<<16-3-4=9>> <<9*2=18>>"


def test_parse_steps_and_reeval_chain():
    steps = parse_steps(_rationale())
    assert [s["expr"] for s in steps] == ["16-3-4", "9*2"]
    assert [s["val"] for s in steps] == ["9", "18"]
    cf = counterfactual_answer(steps, 0, "10", "18")
    assert cf == "20"  # 10*2


def test_counterfactual_answer_none_when_answer_not_downstream():
    steps = parse_steps(_rationale())
    assert counterfactual_answer(steps, 0, "10", "999") is None


def test_score_patch_step_aligned():
    steps = parse_steps(_rationale())
    hit = score_patch(answer_base="18", answer_patched="20", recipient_gold="18",
                       donor_final="99", recipient_chain=steps, donor_value="10", step=0)
    assert hit == {"answer_changed": True, "matches_donor_final": False,
                    "matches_cf": True, "outcome": "counterfactual"}

    unchanged = score_patch(answer_base="18", answer_patched="18", recipient_gold="18",
                             donor_final="99", recipient_chain=steps, donor_value="10", step=0)
    assert unchanged["outcome"] == "unchanged" and unchanged["answer_changed"] is False

    other = score_patch(answer_base="18", answer_patched="7", recipient_gold="18",
                         donor_final="99", recipient_chain=steps, donor_value="10", step=0,
                         recipient_values=["9", "18"], donor_values=["5", "99"])
    assert other["outcome"] == "other_number" and other["matches_cf"] is False

    unparseable = score_patch(answer_base="18", answer_patched=None, recipient_chain=steps,
                               donor_value="10", step=0)
    assert unparseable["outcome"] == "unparseable"


def test_score_patch_donor_final_and_intermediate_buckets():
    steps = parse_steps(_rationale())
    donor_final_hit = score_patch(answer_base="18", answer_patched="99", donor_final="99",
                                   recipient_chain=steps, donor_value="10", step=0)
    assert donor_final_hit["outcome"] == "donor_final"

    recipient_intermediate_hit = score_patch(answer_base="18", answer_patched="9",
                                              recipient_chain=steps, donor_value="10", step=0,
                                              recipient_values=["9", "18"])
    assert recipient_intermediate_hit["outcome"] == "recipient_intermediate"

    donor_intermediate_hit = score_patch(answer_base="18", answer_patched="5",
                                          recipient_chain=steps, donor_value="10", step=0,
                                          donor_values=["5", "40"])
    assert donor_intermediate_hit["outcome"] == "donor_intermediate"


def test_score_patch_unaligned_tries_every_donor_value_and_step():
    steps = parse_steps(_rationale())
    # donor's second value (40) injected at step 0 gives 40*2=80 -- only findable by
    # sweeping both donor values and both steps, which is what the unaligned scorer does.
    hit = score_patch_unaligned(answer_base="18", answer_patched="80", recipient_chain=steps,
                                 donor_values=["3", "40"])
    assert hit["matches_cf"] is True and hit["outcome"] == "counterfactual"

    miss = score_patch_unaligned(answer_base="18", answer_patched="123", recipient_chain=steps,
                                  donor_values=["3", "40"])
    assert miss["matches_cf"] is False


def test_num_equal_handles_thousands_separators_and_none():
    assert num_equal("2,125", "2125")
    assert not num_equal(None, "1")
    assert not num_equal("1", None)


def test_local_sample_rationales_parse_cleanly():
    # sanity check that parse_steps handles every bundled example without raising, and
    # every parsed step's value is a number gsm8k_aug's own intermediate-value parser agrees on.
    for ex in load_local_sample():
        steps = parse_steps(ex.rationale)
        assert [s["val"] for s in steps] == ex.intermediate_values


def test_matches_cf_matches_positive_control_regression_fixture():
    """Regression fixture for the `parse_steps` / `counterfactual_answer` extraction:
    re-derive each pair's counterfactual answer from the logged positive-control run's
    own `predictions.jsonl` (recipient's generated chain) and check it reproduces every
    real_L* / text condition's recorded `matches_cf` -- computed by the pre-extraction
    copy of this same logic in `patch_explicit_cot.py`, which (unlike `score_patch`)
    doesn't guard against cf == base_answer, so this compares against
    `counterfactual_answer` directly rather than through `score_patch`."""
    run_dir = RESULTS_DIR / POSITIVE_CONTROL_RUN
    preds_by_idx = {}
    with (run_dir / "predictions.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            preds_by_idx[r["idx"]] = r

    n_checked = 0
    with (run_dir / "patch_pairs.jsonl").open() as f:
        for line in f:
            pair = json.loads(line)
            recipient = preds_by_idx[pair["recipient_idx"]]
            steps = parse_steps(recipient["raw_output"])
            k = pair["step"] - 1
            cf = counterfactual_answer(steps, k, pair["donor_value"], pair["baseline_answer"])
            assert cf == pair["cf_answer"], pair["recipient_idx"]
            for cond_name, cond in pair["conditions"].items():
                if not (cond_name == "text" or cond_name.startswith("real_L")):
                    continue
                expected = num_equal(cond["answer"], cf) if cf is not None else None
                assert expected == cond["matches_cf"], (pair["recipient_idx"], cond_name)
                n_checked += 1
    assert n_checked > 100
