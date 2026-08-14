"""Identity / join / provenance regression tests for the analysis scripts.

Each test here corresponds to a bug where records were joined by POSITION, by
search ORDER, or by a name FALLBACK instead of by the identity the record
actually carries -- failures that produced wrong numbers while reporting
success.  Nothing here touches an estimator; the assertions are about which
record gets attached to which, and about failing loudly instead of silently.
"""
import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from energy_candidate_ranking.aggregate_confirmation import (  # noqa: E402
    apply_correction, corrected_metric_name, match_corrections)
from energy_candidate_ranking.recompute_corruption_monotonicity import ordered_levels  # noqa: E402


def load_funcs(path, names, extra_globals=None):
    """Exec only the named top-level functions of a script-style module."""
    source = Path(path).read_text()
    tree = ast.parse(source)
    namespace = dict(extra_globals or {})
    exec("import json, os, zlib\nimport numpy as np", namespace)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef,)) and node.name in names:
            exec(ast.get_source_segment(source, node), namespace)
    missing = [n for n in names if n not in namespace]
    assert not missing, f"could not extract {missing} from {path}"
    return namespace


# --------------------------------------------------------------------------
# B7 -- corruption corrections must join on recorded provenance, not position.
# --------------------------------------------------------------------------

def _bank(seed, estimate):
    return {"config": {"seed": seed, "corruption_severities": [0.25, 0.75]},
            "metrics": [{"score": "direct_energy", "metric": "corruption_increases_all_families",
                         "estimate": estimate}]}


def _write_bank(tmp_path, name, seed, estimate):
    path = tmp_path / name
    path.write_text(json.dumps(_bank(seed, estimate)))
    return path


def _write_correction(tmp_path, name, source_path, seed, estimate):
    path = tmp_path / name
    path.write_text(json.dumps({
        "source_metrics": str(source_path),
        "source_metrics_resolved": str(Path(source_path).resolve()),
        "source_config_seed": seed,
        "metrics": [{"score": "direct_energy",
                     "metric": "corruption_increases_all_families_clean_included",
                     "corrects_metric": "corruption_increases_all_families",
                     "estimate": estimate}]}))
    return path


def test_corrections_join_on_provenance_not_argument_order(tmp_path):
    a = _write_bank(tmp_path, "bank_a.json", 1, 0.10)
    b = _write_bank(tmp_path, "bank_b.json", 2, 0.20)
    corr_a = _write_correction(tmp_path, "corr_a.json", a, 1, 0.11)
    corr_b = _write_correction(tmp_path, "corr_b.json", b, 2, 0.22)

    for order in ([corr_a, corr_b], [corr_b, corr_a]):
        banks = [json.loads(a.read_text()), json.loads(b.read_text())]
        for bank, (correction, path) in zip(banks, match_corrections([a, b], banks, order)):
            apply_correction(bank, correction, path)
        # Bank A must always receive A's correction, whatever the CLI glob order was.
        assert banks[0]["metrics"][0]["estimate"] == 0.11
        assert banks[1]["metrics"][0]["estimate"] == 0.22


def test_correction_from_a_foreign_bank_is_an_error(tmp_path):
    a = _write_bank(tmp_path, "bank_a.json", 1, 0.10)
    b = _write_bank(tmp_path, "bank_b.json", 2, 0.20)
    other = _write_bank(tmp_path, "bank_other.json", 9, 0.90)
    corr_a = _write_correction(tmp_path, "corr_a.json", a, 1, 0.11)
    corr_other = _write_correction(tmp_path, "corr_other.json", other, 9, 0.99)
    banks = [json.loads(a.read_text()), json.loads(b.read_text())]
    with pytest.raises(ValueError, match="does not resolve to exactly one"):
        match_corrections([a, b], banks, [corr_a, corr_other])


def test_legacy_relative_path_sidecars_still_join(tmp_path):
    """Pre-existing sidecars record a repo-relative source path with no seed field."""
    a = _write_bank(tmp_path, "bank_a.json", 1, 0.10)
    b = _write_bank(tmp_path, "bank_b.json", 2, 0.20)
    legacy = tmp_path / "legacy_corr_b.json"
    legacy.write_text(json.dumps({
        "source_metrics": f"{tmp_path.name}/{b.name}",  # repo-relative, as legacy sidecars record it
        "metrics": [{"score": "direct_energy",
                     "metric": "corruption_increases_all_families_clean_included",
                     "estimate": 0.22}]}))
    other = tmp_path / "legacy_corr_a.json"
    other.write_text(json.dumps({
        "source_metrics": f"{tmp_path.name}/{a.name}",
        "metrics": [{"score": "direct_energy",
                     "metric": "corruption_increases_all_families_clean_included",
                     "estimate": 0.11}]}))
    banks = [json.loads(a.read_text()), json.loads(b.read_text())]
    for bank, (correction, path) in zip(banks, match_corrections([a, b], banks, [legacy, other])):
        apply_correction(bank, correction, path)
    assert [bk["metrics"][0]["estimate"] for bk in banks] == [0.11, 0.22]


def test_correction_seed_must_agree_with_bank_seed(tmp_path):
    a = _write_bank(tmp_path, "bank_a.json", 1, 0.10)
    corr = _write_correction(tmp_path, "corr_a.json", a, 7, 0.11)  # wrong recorded seed
    with pytest.raises(ValueError, match="records source seed"):
        match_corrections([a], [json.loads(a.read_text())], [corr])


def test_unmatched_correction_row_is_an_error_not_a_silent_passthrough(tmp_path):
    bank = _bank(1, 0.10)
    correction = {"metrics": [{"score": "direct_energy",
                               "metric": "a_metric_that_is_not_in_the_bank_clean_included",
                               "estimate": 0.5}]}
    with pytest.raises(ValueError, match="no matching bank row"):
        apply_correction(bank, correction, Path("corr.json"))
    # and the uncorrected value must not have been quietly published
    assert bank["metrics"][0]["estimate"] == 0.10


def test_clean_included_is_stripped_as_a_suffix_not_replaced_everywhere():
    # str.replace deletes the interior occurrence too: this used to yield "a_b".
    row = {"metric": "a_clean_included_b_clean_included"}
    assert corrected_metric_name(row) == "a_clean_included_b"
    with pytest.raises(ValueError, match="does not end with"):
        corrected_metric_name({"metric": "no_suffix_here"})


# --------------------------------------------------------------------------
# B8 -- the corruption ladder must be ordered before pairwise comparison.
# --------------------------------------------------------------------------

def test_severity_levels_are_ordered_and_disorder_is_rejected():
    assert ordered_levels({"corruption_severities": [0.25, 0.75]}) == (0., 0.25, 0.75)
    with pytest.raises(ValueError, match="increasing order"):
        ordered_levels({"corruption_severities": [0.75, 0.25]})
    with pytest.raises(ValueError, match="unique"):
        ordered_levels({"corruption_severities": [0.25, 0.25]})
    with pytest.raises(ValueError, match="positive"):
        ordered_levels({"corruption_severities": [0., 0.25]})


# --------------------------------------------------------------------------
# B6 -- stale-vs-corrected file resolution must be unambiguous.
# --------------------------------------------------------------------------

def test_stage1_file_resolution_fails_on_ambiguity_instead_of_preferring_the_stale_file(tmp_path):
    ns = load_funcs(ROOT / "documentation/stage1_longer_training_2026-07-21/analyze_stage1.py",
                    ["find_file"])
    stale_dir = tmp_path / "holylabs"
    fixed_dir = tmp_path / "home03"
    stale_dir.mkdir()
    fixed_dir.mkdir()
    patterns = ["stage1_mask_seed0_v3_epoch{}", "stage1_mask_seed0_epoch{}"]
    roots = [str(stale_dir), str(fixed_dir)]

    # Only the corrected re-run exists -> resolves to it.
    (fixed_dir / "stage1_mask_seed0_v3_epoch3_fid.json").write_text("{}")
    assert ns["find_file"](patterns, 3, "_fid.json", roots).startswith(str(fixed_dir))

    # Pre-incident file also present in the earlier directory: previously the
    # directory-outer loop returned the STALE one; now it is a hard failure.
    (stale_dir / "stage1_mask_seed0_epoch3_fid.json").write_text("{}")
    with pytest.raises(AssertionError, match="ambiguous file resolution"):
        ns["find_file"](patterns, 3, "_fid.json", roots)

    assert ns["find_file"](patterns, 4, "_fid.json", roots) is None


# --------------------------------------------------------------------------
# B9 / B10 -- the plotter's arm identity and admission policy.
# --------------------------------------------------------------------------

def _record(**over):
    rec = {"record_id": "r0", "model_arm": "gaussian", "checkpoint_id": "gaussian_seed0",
           "sample_id": "s0", "dataset_index": 0, "projection_mode": "none",
           "mask_family": "combined", "requested_visible_fraction": .5,
           "lpips_missing_composite": .5, "missing_model_mse": .5,
           "completion_status": "ok"}
    rec.update(over)
    return rec


def test_plotter_arm_uses_first_class_field_and_refuses_to_guess():
    import plot_frozen_prior_constraint as P
    assert P.arm(_record(model_arm="mixed", checkpoint_id="whatever")) == "mixed"
    # present-but-null: `.get` with a default never fires, so the arm used to
    # become None and vanish from every figure.
    with pytest.raises(RuntimeError, match="no model_arm"):
        P.arm(_record(model_arm=None))
    with pytest.raises(RuntimeError, match="unrecognized model_arm"):
        P.arm(_record(model_arm="gaussain"))


def test_plotter_admission_matches_the_analyzer(tmp_path):
    import plot_frozen_prior_constraint as P
    import analyze_frozen_prior_constraint as A
    shard = tmp_path / "shard.jsonl"
    shard.write_text("\n".join(json.dumps(r) for r in [
        _record(record_id="r0"), _record(record_id="r1", completion_status="failed")]))
    # The analyzer refuses this directory; the plotter must refuse it identically
    # rather than rendering a figure over the surviving subset.
    with pytest.raises(RuntimeError):
        A.load_rows(str(tmp_path))
    with pytest.raises(RuntimeError):
        P.rows_at(str(tmp_path))


def test_plotter_rejects_duplicate_pair_keys(tmp_path):
    import argparse
    import plot_frozen_prior_constraint as P
    shard = tmp_path / "shard.jsonl"
    rows = [_record(record_id="a", projection_mode="none"),
            _record(record_id="b", projection_mode="hard"),
            # re-run shard: distinct record_id, identical pairing key
            _record(record_id="c", projection_mode="hard", lpips_missing_composite=.9)]
    shard.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "figs"
    with pytest.raises(RuntimeError, match="duplicate pair keys"):
        P.main(argparse.Namespace(input_dir=str(tmp_path), output_dir=str(out)))


# --------------------------------------------------------------------------
# B3 -- analysis seeds must be a fixed function of their inputs.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("script", [
    "documentation/severity_sweep_2026-07-16/analyze_severity.py",
    "documentation/convergence_2026-07-19/analyze_convergence.py",
])
def test_analysis_seeds_are_process_stable(script):
    path = ROOT / script
    assert "hash((" not in path.read_text(), "salted builtin hash() must not seed an analysis"
    program = textwrap.dedent(f"""
        import ast, sys, zlib
        src = open({str(path)!r}).read()
        ns = {{'zlib': zlib}}
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name == 'stable_seed':
                exec(ast.get_source_segment(src, node), ns)
        print([ns['stable_seed'](t, k) for t in ('g', 'm') for k in ('0.1', '250')])
    """)
    seen = set()
    for salt in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        seen.add(subprocess.run([sys.executable, "-c", program], env=env,
                                capture_output=True, text=True, check=True).stdout.strip())
    assert len(seen) == 1, f"seeds differ across PYTHONHASHSEED: {seen}"
