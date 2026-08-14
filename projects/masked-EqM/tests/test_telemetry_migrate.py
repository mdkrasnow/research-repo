"""Tests for the legacy parsers, the reconstruction rules, and the migrator.

The emphasis is deliberate: almost every test below is about a case where the
correct behaviour is *to refuse* -- to leave a field unknown, to decline to
synthesize a terminal record, to classify a record as ``unknown`` rather than
guess.  Those are the behaviours that make a reconstruction trustworthy, and
they are exactly the behaviours a future refactor is most likely to "simplify"
away, because each one looks like a missing feature until you know what it
prevents.

Run with::

    python -m pytest projects/masked-EqM/tests/test_telemetry_migrate.py -q
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telemetry import legacy, migrate, contradictions  # noqa: E402
from telemetry.schema import EventType, RunStatus, validate_record  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _fact(**kwargs) -> legacy.LegacyFact:
    base = dict(source=legacy.SRC_PIPELINE_COMPLETED, locator="completed_runs[0]",
                path="/tmp/pipeline.json")
    base.update(kwargs)
    return legacy.LegacyFact(**base)


def _write_project(root: str, *, pipeline=None, manifest=None, dec=None,
                   delt=None, tsv_rows=None) -> str:
    os.makedirs(os.path.join(root, ".state"), exist_ok=True)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    with open(os.path.join(root, ".state", "pipeline.json"), "w") as handle:
        json.dump(pipeline or {"active_runs": [], "completed_runs": []}, handle)
    if manifest is not None:
        os.makedirs(os.path.join(root, "results", "btm"), exist_ok=True)
        with open(os.path.join(root, "results", "btm", "manifest.jsonl"), "w") as handle:
            for row in manifest:
                handle.write(json.dumps(row) + "\n")
    if dec is not None:
        d = os.path.join(root, "results", "direct_energy_campaign")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "status.json"), "w") as handle:
            json.dump(dec["status"], handle)
        with open(os.path.join(d, "events.jsonl"), "w") as handle:
            for row in dec["events"]:
                handle.write(json.dumps(row) + "\n")
    if delt is not None:
        d = os.path.join(root, "results", "direct_energy_longer_training")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "status.json"), "w") as handle:
            json.dump(delt["status"], handle)
        with open(os.path.join(d, "events.jsonl"), "w") as handle:
            for row in delt["events"]:
                handle.write(json.dumps(row) + "\n")
    if tsv_rows is not None:
        with open(os.path.join(root, "results_variants.tsv"), "w") as handle:
            handle.write("run_id\tjob_id\tdate\tphase\tmetric_value\n")
            for row in tsv_rows:
                handle.write("\t".join(row) + "\n")
    return root


# ---------------------------------------------------------------------------
# job-id recognition -- the phantom-id defence
# ---------------------------------------------------------------------------

class TestJobIdRecognition(unittest.TestCase):

    def test_accepts_real_job_ids(self):
        for value in ("35436507", "39090540", "35438088_3", 39090541):
            self.assertTrue(legacy.is_job_id(value), value)

    def test_rejects_float_metric_values(self):
        """The concrete phantom-id case from the longer-training ledger.

        ``129.12085939165866`` sits inside a dict named ``jobs``.  An unanchored
        ``\\d{6,}`` scan finds ``12085939`` in its mantissa and invents a job that
        never existed.
        """
        for value in (129.12085939165866, 118.43803581913176, "129.12085939165866"):
            self.assertFalse(legacy.is_job_id(value), value)

    def test_rejects_state_words_and_short_numbers(self):
        for value in ("complete", "pending", "", None, "12", True):
            self.assertFalse(legacy.is_job_id(value), value)


class TestSeedSuffix(unittest.TestCase):

    def test_decodes_strict_suffixes_only(self):
        self.assertEqual(legacy.split_seed_suffix("btm_IIA_G_s0"), ("btm_IIA_G", 0))
        self.assertEqual(legacy.split_seed_suffix("none_seed12"), ("none", 12))

    def test_leaves_ambiguous_names_undecoded(self):
        """Under-decoding is marked unknown; over-decoding mislabels an arm."""
        for label in ("btm_scalar_fd_directional4", "epoch08_fid_none_value",
                      "stage3_pilot"):
            stem, seed = legacy.split_seed_suffix(label)
            self.assertEqual(stem, label)
            self.assertIsNone(seed)

    def test_does_not_collapse_prefix_related_arms(self):
        """The original identity bug: one arm name is a prefix of another."""
        a, _ = legacy.split_seed_suffix("btm_scalar_fd_directional_s0")
        b, _ = legacy.split_seed_suffix("btm_scalar_fd_directional4_s0")
        self.assertNotEqual(a, b)


# ---------------------------------------------------------------------------
# metric-record classification -- the two-shapes-one-file defect
# ---------------------------------------------------------------------------

GRAD_RECORD = {
    "step": 50, "loss": 0.31, "grad_norm": 4.2, "head_grad_norm": 0.1,
    "backbone_grad_norm": 4.19, "max_grad_norm": 6.87141, "clipped": False,
    "learning_rate": 1e-4, "adaptive_clip": False, "weight_decay": 0.0,
}
PROBE_RECORD = {
    "step": 50, "delta_theta_norm": 0.004, "probe_loss_pre": 10.1,
    "probe_loss_post": 10.0, "probe_delta_L": -0.1,
    "probe_cos_field_update_vs_neg_r": 0.42, "field_delta_norm": 0.9,
    "P_t": 1.2, "eta_func": 300.0,
}
WFB_RECORD = {
    "step": 0, "lambda_max": 633839.5, "lam": 63.38, "m_lanczos": 96,
    "eta_star": 3.3e-06, "eta_used": 3.3e-06, "n_backtracks": 0,
    "actual_delta_L": -0.23, "predicted_delta_L": -0.245, "r_dot_q": 2.4e9,
}


class TestMetricClassification(unittest.TestCase):

    def test_grad_record(self):
        kind, reason = legacy.classify_metric_record(GRAD_RECORD)
        self.assertEqual(kind, "grad")
        self.assertIn("grad_norm", reason)

    def test_probe_record(self):
        kind, reason = legacy.classify_metric_record(PROBE_RECORD)
        self.assertEqual(kind, "probe")
        self.assertIn("eta_func", reason)

    def test_wfb_record(self):
        self.assertEqual(legacy.classify_metric_record(WFB_RECORD)[0], "wfb")

    def test_shapes_are_key_disjoint_apart_from_step(self):
        """The soundness precondition of the whole heuristic.

        If the two writers ever shared a non-``step`` key, witness membership
        would stop being a discriminator and this test is where that shows up.
        """
        shared = set(GRAD_RECORD) & set(PROBE_RECORD)
        self.assertEqual(shared, {"step"})

    def test_grad_record_with_optional_blocks_still_grad(self):
        """The real file adds fwrev/wfb/btm blocks to the grad record."""
        record = dict(GRAD_RECORD, loss_main=0.3, field_norm=1.0, btm_mode="btm_vector",
                      btm_tc=0.9, r_norm=594.0)
        self.assertEqual(legacy.classify_metric_record(record)[0], "grad")

    def test_wfb_enabled_grad_record_is_still_grad_not_ambiguous(self):
        """`--wfb-backward` adds lambda_max/lam/m_lanczos to the GRAD record.

        Those keys also appear in the Stage-3 optimizer trace, so they cannot be
        WFB witnesses: if they were, every WFB-enabled training run -- the runs
        the whole investigation is about -- would classify as ``unknown``.
        """
        record = dict(GRAD_RECORD, r_norm=594.0, g_raw_norm_hypothetical=1e5,
                      lambda_max=633839.5, lam=63.38, T_eigmax=1.0, m_lanczos=96,
                      wfb_breakdown=False, wfb_rho=0.1, wfb_k=8, wfb_alpha=1.0)
        self.assertEqual(legacy.classify_metric_record(record)[0], "grad")

    def test_witness_sets_are_pairwise_disjoint(self):
        """Soundness precondition: no key may witness two different writers."""
        sets = {"grad": legacy._GRAD_WITNESS, "probe": legacy._PROBE_WITNESS,
                "wfb": legacy._WFB_WITNESS}
        for left in sets:
            for right in sets:
                if left < right:
                    self.assertEqual(sets[left] & sets[right], set(),
                                     f"{left}/{right} share witness keys")

    def test_ambiguous_record_is_not_guessed(self):
        merged = dict(GRAD_RECORD)
        merged.update(PROBE_RECORD)
        kind, reason = legacy.classify_metric_record(merged)
        self.assertEqual(kind, "unknown")
        self.assertIn("ambiguous", reason)

    def test_unrecognized_shape_is_not_guessed(self):
        kind, reason = legacy.classify_metric_record({"step": 1, "mystery": 2})
        self.assertEqual(kind, "unknown")
        self.assertIn("no_witness_key", reason)
        self.assertIn("mystery", reason)


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------

class TestParsers(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def test_pipeline_locators_address_duplicate_run_ids(self):
        root = _write_project(self.tmp, pipeline={
            "active_runs": [{"run_id": "a", "job_id": "1111", "status": "pending"}],
            "completed_runs": [
                {"run_id": "dup", "job_id": "2222", "status": "completed"},
                {"run_id": "dup", "job_id": "3333", "status": "failed"},
            ]})
        facts = legacy.parse_pipeline(os.path.join(root, ".state", "pipeline.json"))
        self.assertEqual(len(facts), 3)
        self.assertEqual([f.locator for f in facts],
                         ["active_runs[0]", "completed_runs[0]", "completed_runs[1]"])
        self.assertEqual(facts[0].source, legacy.SRC_PIPELINE_ACTIVE)

    def test_inverted_jobs_dict_is_read_in_the_right_orientation(self):
        status = {"commit": "abc1234", "started_at": "2026-07-25T00:00:00Z",
                  "jobs": {"none_seed0": "35335932",
                           "epoch08_fid_none_value": 129.12085939165866,
                           "epoch15_checkpoints": "complete"}}
        d = os.path.join(self.tmp, "results", "direct_energy_longer_training")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "status.json"), "w") as handle:
            json.dump(status, handle)
        with open(os.path.join(d, "events.jsonl"), "w") as handle:
            handle.write("")
        facts = legacy.parse_longer_training(os.path.join(d, "status.json"),
                                             os.path.join(d, "events.jsonl"))
        with_job = [f for f in facts if f.job_id]
        self.assertEqual(len(with_job), 1)
        self.assertEqual(with_job[0].job_id, "35335932")
        self.assertEqual(with_job[0].label, "none_seed0")
        # The float and the state word are preserved, not silently dropped.
        self.assertEqual(len([f for f in facts if f.raw.get("_not_a_job")]), 2)

    def test_campaign_event_naming_n_jobs_becomes_n_facts(self):
        event = {"at": "2026-07-25T00:00:00Z", "stage": "retry", "status": "RETRYING",
                 "jobs": ["35438088", "35438089", "35438100"]}
        facts = legacy._campaign_event_facts(event, legacy.SRC_DELT_EVENTS, "line[0]",
                                             "/tmp/e.jsonl")
        self.assertEqual(sorted(f.job_id for f in facts),
                         ["35438088", "35438089", "35438100"])
        self.assertTrue(all(f.raw == event for f in facts))

    def test_campaign_event_accepts_both_timestamp_keys(self):
        for key in ("at", "timestamp"):
            facts = legacy._campaign_event_facts(
                {key: "2026-07-25T00:00:00Z", "job": "35335932", "status": "PASS"},
                legacy.SRC_DELT_EVENTS, "line[0]", "/tmp/e.jsonl")
            self.assertEqual(facts[0].ts, "2026-07-25T00:00:00Z")

    def test_metric_stream_path_identity_only_when_job_id_present(self):
        tagged = "/x/RESULTS/btm_IIA_G_s0_job39090540/000-EqM-B-2/gradient_metrics.jsonl"
        tag, job, expdir = legacy._identity_from_path(tagged)
        self.assertEqual((tag, job), ("btm_IIA_G_s0", "39090540"))
        untagged = "/x/RESULTS/stage3_pilot/none_seed0/metrics.jsonl"
        tag, job, _ = legacy._identity_from_path(untagged)
        self.assertIsNone(job)
        self.assertEqual(tag, "none_seed0")


# ---------------------------------------------------------------------------
# reconstruction rules
# ---------------------------------------------------------------------------

class TestReconstruction(unittest.TestCase):

    def test_unknown_fields_are_marked_not_guessed(self):
        spec, source, unknown, confidence = migrate._reconstruct_spec(
            None, [_fact(label=None, status="completed")])
        self.assertEqual(spec.arm, legacy.UNKNOWN)
        self.assertEqual(spec.seed, legacy.UNKNOWN_SEED)
        self.assertEqual(spec.git_sha, legacy.UNKNOWN)
        for field in ("arm", "seed", "git_sha", "phase", "job_id"):
            self.assertIn(field, unknown)
        self.assertEqual(confidence, "none")

    def test_btm_manifest_gives_high_confidence_and_real_params(self):
        fact = _fact(source=legacy.SRC_BTM_MANIFEST, locator="line[0]",
                     job_id="39067773", label="btm_IIA_btm_vector_s0",
                     status="submitted", git_sha="6310e02", phase="IIA",
                     raw={"btm_mode": "btm_vector", "fd_k": 1, "fd_eps": 0.001,
                          "tc": 0.9, "max_steps": 20000, "seed": 0, "ebm": "none",
                          "global_batch": 256, "epochs": 80})
        spec, source, unknown, confidence = migrate._reconstruct_spec("39067773", [fact])
        self.assertEqual(confidence, "high")
        self.assertEqual(spec.seed, 0)
        self.assertEqual(spec.planned_steps, 20000)
        self.assertEqual(spec.params["btm_mode"], "btm_vector")
        self.assertEqual(spec.params["fd_k"], 1)
        self.assertNotIn("params", unknown)

    def test_pipeline_only_reconstruction_admits_missing_params(self):
        fact = _fact(job_id="39090540", label="btm_IIA_G_s0", status="completed",
                     git_sha="0f96580", phase="BTM-II-A",
                     raw={"description": "prose that is not a spec"})
        spec, _, unknown, confidence = migrate._reconstruct_spec("39090540", [fact])
        self.assertEqual(confidence, "medium")
        self.assertIn("params", unknown)
        self.assertEqual(spec.arm, "btm_IIA_G")
        self.assertEqual(spec.seed, 0)

    def test_identical_specs_collapse_to_one_run_uid(self):
        """Content addressing must join two executions of one experiment."""
        a = _fact(job_id="1111", label="arm_x_s0", status="completed", git_sha="abc",
                  phase="P")
        b = _fact(locator="completed_runs[1]", job_id="2222", label="arm_x_s0",
                  status="completed", git_sha="abc", phase="P")
        spec_a, *_ = migrate._reconstruct_spec("1111", [a])
        spec_b, *_ = migrate._reconstruct_spec("2222", [b])
        self.assertEqual(spec_a.run_uid, spec_b.run_uid)

    def test_prefix_related_arms_do_not_collapse(self):
        a = _fact(job_id="1", label="btm_scalar_fd_directional_s0", git_sha="abc",
                  phase="P", status="completed")
        b = _fact(job_id="2", label="btm_scalar_fd_directional4_s0", git_sha="abc",
                  phase="P", status="completed")
        spec_a, *_ = migrate._reconstruct_spec("1", [a])
        spec_b, *_ = migrate._reconstruct_spec("2", [b])
        self.assertNotEqual(spec_a.run_uid, spec_b.run_uid)


class TestTerminalDecision(unittest.TestCase):

    def test_single_terminal_status_seals(self):
        terminal, notices = migrate._terminal_decision(
            [_fact(job_id="1", status="completed", ts="2026-01-01T00:00:00Z")])
        self.assertIsNotNone(terminal)
        self.assertEqual(terminal["status"], RunStatus.COMPLETED.value)

    def test_conflicting_statuses_refuse_to_seal(self):
        """The 35436507 case: one ledger says failed, another says completed."""
        facts = [_fact(job_id="35436507", status="failed", locator="completed_runs[10]"),
                 _fact(job_id="35436507", status="completed", locator="completed_runs[20]")]
        terminal, notices = migrate._terminal_decision(facts)
        self.assertIsNone(terminal)
        codes = {n["code"] for n in notices}
        self.assertIn("disputed_terminal_status", codes)

    def test_stranded_non_terminal_is_reported_but_not_sealed(self):
        terminal, notices = migrate._terminal_decision(
            [_fact(job_id="1", status="pending", source=legacy.SRC_PIPELINE_COMPLETED)])
        self.assertIsNone(terminal)
        self.assertIn("stranded_non_terminal", {n["code"] for n in notices})

    def test_annotation_statuses_never_seal(self):
        for status in ("INVALID -- discarded", "superseded"):
            terminal, notices = migrate._terminal_decision(
                [_fact(job_id="1", status=status)])
            self.assertIsNone(terminal, status)
            self.assertIn("non_lifecycle_status", {n["code"] for n in notices})

    def test_write_error_status_seals_with_a_caveat(self):
        terminal, notices = migrate._terminal_decision(
            [_fact(job_id="1", status="completed_with_write_error")])
        self.assertEqual(terminal["status"], RunStatus.COMPLETED.value)
        self.assertTrue(terminal["caveats"])
        self.assertIn("qualified_terminal_status", {n["code"] for n in notices})

    def test_gate_verdicts_from_event_logs_never_seal(self):
        """A stage saying PASS is not a process saying it exited."""
        terminal, _ = migrate._terminal_decision(
            [_fact(job_id="1", status="PASS", source=legacy.SRC_DEC_EVENTS),
             _fact(job_id="1", status="COMPLETED", source=legacy.SRC_DELT_EVENTS)])
        self.assertIsNone(terminal)

    def test_manifest_submitted_alone_does_not_seal(self):
        terminal, _ = migrate._terminal_decision(
            [_fact(job_id="1", status="submitted", source=legacy.SRC_BTM_MANIFEST)])
        self.assertIsNone(terminal)


# ---------------------------------------------------------------------------
# rendering and end-to-end
# ---------------------------------------------------------------------------

class TestRendering(unittest.TestCase):

    def _run(self):
        fact = _fact(job_id="39090540", label="btm_IIA_G_s0", status="completed",
                     git_sha="0f96580", phase="BTM-II-A", ts="2026-08-13T00:00:00Z",
                     raw={"exit_code": "0:0", "final_metric": "fid=31.4"})
        runs, _, _ = migrate.build_reconstructions([fact], [])
        return runs[0]

    def test_records_validate_and_are_sequenced(self):
        records = migrate.render_records(self._run())
        for index, record in enumerate(records):
            validate_record(record)
            self.assertEqual(record["seq"], index)

    def test_ledger_facts_are_observed_not_start(self):
        records = migrate.render_records(self._run())
        events = [r["event"] for r in records]
        self.assertNotIn(EventType.START.value, events)
        self.assertIn(EventType.OBSERVED.value, events)

    def test_terminal_end_is_marked_inferred(self):
        records = migrate.render_records(self._run())
        ends = [r for r in records if r["event"] == EventType.END.value]
        self.assertEqual(len(ends), 1)
        self.assertTrue(ends[0]["inferred"])
        self.assertTrue(ends[0]["inference_basis"])
        self.assertEqual(ends[0]["last_step"], -1)

    def test_reconstruction_header_carries_provenance(self):
        records = migrate.render_records(self._run())
        header = records[0]
        self.assertEqual(header["event"], EventType.NOTICE.value)
        self.assertEqual(header["provenance"], "reconstructed")
        self.assertIn(header["confidence"], migrate.CONFIDENCE_ORDER)

    def test_original_record_survives_the_migration(self):
        records = migrate.render_records(self._run())
        observed = [r for r in records if r["event"] == EventType.OBSERVED.value][0]
        self.assertEqual(observed["legacy_record"]["exit_code"], "0:0")

    def test_rendering_is_deterministic(self):
        a = migrate.render_records(self._run())
        b = migrate.render_records(self._run())
        self.assertEqual([json.dumps(r, sort_keys=True) for r in a],
                         [json.dumps(r, sort_keys=True) for r in b])


class TestStreamReplay(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.stream_dir = os.path.join(
            self.tmp, "results", "btm_IIA_G_s0_job39090540", "000-EqM-B-2")
        os.makedirs(self.stream_dir)
        self.stream_path = os.path.join(self.stream_dir, "gradient_metrics.jsonl")
        with open(self.stream_path, "w") as handle:
            handle.write(json.dumps(GRAD_RECORD) + "\n")
            handle.write(json.dumps(PROBE_RECORD) + "\n")
            handle.write("{ this is not json\n")

    def test_interleaved_shapes_get_distinct_kinds(self):
        streams = legacy.discover_metric_streams(os.path.join(self.tmp, "results"))
        self.assertEqual(len(streams), 1)
        self.assertEqual(streams[0].job_id, "39090540")
        fact = _fact(job_id="39090540", label="btm_IIA_G_s0", status="completed",
                     git_sha="abc", phase="P")
        runs, _, _ = migrate.build_reconstructions([fact], streams)
        records = migrate.render_records(runs[0])
        progress = [r for r in records if r["event"] == EventType.PROGRESS.value]
        self.assertEqual([r["kind"] for r in progress], ["grad", "probe"])
        self.assertTrue(all(r["replayed"] for r in progress))
        self.assertTrue(all(r["run_uid"] == runs[0].spec.run_uid for r in progress))
        # The unparseable line becomes an error notice, not a silent drop.
        codes = {r.get("code") for r in records}
        self.assertIn("unparseable_metric_line", codes)

    def test_pathless_stream_becomes_its_own_low_confidence_run(self):
        loose = os.path.join(self.tmp, "results", "stage3_pilot", "none_seed0")
        os.makedirs(loose)
        with open(os.path.join(loose, "metrics.jsonl"), "w") as handle:
            handle.write(json.dumps(GRAD_RECORD) + "\n")
        streams = legacy.discover_metric_streams(os.path.join(self.tmp, "results"))
        runs, _, _ = migrate.build_reconstructions([], streams)
        pathless = [r for r in runs if r.job_id.startswith("nojob-")]
        self.assertEqual(len(pathless), 1)
        self.assertIn("job_id", pathless[0].unknown_fields)
        self.assertIn(pathless[0].confidence, ("low", "none"))
        self.assertIn("path_only_identity",
                      {n["code"] for n in pathless[0].notices})


class TestIdempotence(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        _write_project(
            self.tmp,
            pipeline={"active_runs": [],
                      "completed_runs": [
                          {"run_id": "arm_a_s0", "job_id": "1000",
                           "status": "completed", "git_sha": "abc", "phase": "P"},
                          {"run_id": "arm_b_s0", "job_id": "2000",
                           "status": "failed", "git_sha": "abc", "phase": "P"},
                          {"run_id": "arm_b_s0", "job_id": "2000",
                           "status": "completed", "git_sha": "abc", "phase": "P"}]})
        self.telemetry_root = os.path.join(self.tmp, "results", "telemetry")

    def _line_counts(self):
        counts = {}
        for dirpath, _dirs, files in os.walk(self.telemetry_root):
            for name in files:
                if name.endswith(".jsonl"):
                    with open(os.path.join(dirpath, name)) as handle:
                        counts[name] = sum(1 for line in handle if line.strip())
        return counts

    def test_second_run_writes_nothing_new(self):
        first = migrate.migrate(self.tmp, self.telemetry_root)
        counts_first = self._line_counts()
        second = migrate.migrate(self.tmp, self.telemetry_root)
        counts_second = self._line_counts()
        self.assertEqual(counts_first, counts_second)
        self.assertEqual(second["outcomes"].get("unchanged"),
                         second["totals"]["reconstructed_executions"])
        self.assertEqual(first["totals"], second["totals"])

    def test_partial_write_is_repaired_on_rerun(self):
        migrate.migrate(self.tmp, self.telemetry_root)
        target = None
        for dirpath, _dirs, files in os.walk(self.telemetry_root):
            for name in files:
                if name.endswith(".jsonl"):
                    target = os.path.join(dirpath, name)
        with open(target, "w") as handle:
            handle.write("")  # simulate an interrupted migration
        report = migrate.migrate(self.tmp, self.telemetry_root)
        self.assertGreaterEqual(report["outcomes"].get("rewritten", 0), 1)
        with open(target) as handle:
            self.assertGreater(sum(1 for line in handle if line.strip()), 0)

    def test_duplicate_job_id_yields_one_unsealed_execution(self):
        report = migrate.migrate(self.tmp, self.telemetry_root)
        by_job = {r["job_id"]: r for r in report["runs"]}
        self.assertEqual(by_job["2000"]["terminal_status"], None)
        self.assertIn("disputed_terminal_status", by_job["2000"]["notice_codes"])
        self.assertEqual(by_job["1000"]["terminal_status"], "completed")

    def test_refuses_to_delete_in_an_unowned_directory(self):
        os.makedirs(self.telemetry_root, exist_ok=True)
        stray = os.path.join(self.telemetry_root, "x")
        os.makedirs(stray)
        with self.assertRaises(RuntimeError):
            migrate._assert_owned(self.telemetry_root)


# ---------------------------------------------------------------------------
# contradiction checker
# ---------------------------------------------------------------------------

class TestContradictions(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        _write_project(
            self.tmp,
            pipeline={"active_runs": [],
                      "completed_runs": [
                          {"run_id": "dup", "job_id": "1000", "status": "failed",
                           "git_sha": "abc"},
                          {"run_id": "dup", "job_id": "1000", "status": "completed",
                           "git_sha": "abc"},
                          {"run_id": "only_pipeline", "job_id": "3000",
                           "status": "pending", "git_sha": "abc"},
                          {"run_id": "weird", "job_id": "4000",
                           "status": "INVALID -- discarded", "git_sha": "abc"}]},
            manifest=[{"run_tag": "manifest_name_s0", "job_id": "1000",
                       "status": "submitted", "git_sha": "def", "phase": "IIA",
                       "seed": 0, "btm_mode": "btm_vector"},
                      {"run_tag": "orphan_s0", "job_id": "9999",
                       "status": "submitted", "git_sha": "def", "phase": "IIA",
                       "seed": 0, "btm_mode": "btm_vector"}])

    def test_detects_status_disagreement(self):
        report = contradictions.analyze(self.tmp)
        rows = [f for f in report["findings"]
                if f["kind"] == "disagreement" and f.get("field") == "status"]
        self.assertEqual([r["job_id"] for r in rows], ["1000"])

    def test_detects_label_and_sha_disagreement_across_artifacts(self):
        report = contradictions.analyze(self.tmp)
        fields = {f.get("field") for f in report["findings"]
                  if f["kind"] == "disagreement" and f["job_id"] == "1000"}
        self.assertIn("label", fields)
        self.assertIn("git_sha", fields)

    def test_short_and_long_sha_are_not_a_disagreement(self):
        merged = contradictions._collapse_sha_prefixes(
            {"92dc605": [{"source": "a", "locator": "x"}],
             "92dc605f1e2d3c4b5a": [{"source": "b", "locator": "y"}]})
        self.assertEqual(len(merged), 1)

    def test_detects_orphans(self):
        report = contradictions.analyze(self.tmp)
        orphans = {f["job_id"] for f in report["findings"] if f["kind"] == "orphan"}
        self.assertIn("9999", orphans)   # manifest-only
        self.assertIn("3000", orphans)   # pipeline-only
        self.assertNotIn("1000", orphans)

    def test_detects_duplicate_keys(self):
        report = contradictions.analyze(self.tmp)
        collisions = [f for f in report["findings"] if f["kind"] == "collision"]
        keys = {(f.get("key"), f.get("value")) for f in collisions}
        self.assertIn(("run_id", "dup"), keys)
        self.assertIn(("job_id", "1000"), keys)

    def test_detects_status_enum_violation_and_frozen_manifest(self):
        report = contradictions.analyze(self.tmp)
        keys = {f.get("key") for f in report["findings"]}
        self.assertIn("status_enum", keys)
        self.assertIn("stranded_non_terminal", keys)
        self.assertIn("frozen_status", keys)

    def test_markdown_renders(self):
        report = contradictions.analyze(self.tmp)
        text = contradictions.render_markdown(report)
        self.assertIn("# Legacy provenance contradiction report", text)
        self.assertIn("Disagreements", text)

    def test_read_only_with_respect_to_legacy_files(self):
        """The load-bearing safety property: nothing legacy changes."""
        before = {}
        for dirpath, _dirs, files in os.walk(self.tmp):
            for name in files:
                path = os.path.join(dirpath, name)
                with open(path, "rb") as handle:
                    before[path] = handle.read()
        contradictions.analyze(self.tmp)
        migrate.migrate(self.tmp, os.path.join(self.tmp, "results", "telemetry"))
        for path, content in before.items():
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), content, f"{path} was modified")


if __name__ == "__main__":
    unittest.main()


class TestStalePruning(unittest.TestCase):
    """A reconstruction-logic change must not leave a second, ghost copy of a run.

    Two reconstructions of one job under two ``run_uid``s is the exact disease
    the identity scheme cures; the migrator would reintroduce it by accumulating
    output from earlier versions of its own logic.
    """

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        _write_project(self.tmp, pipeline={
            "active_runs": [],
            "completed_runs": [{"run_id": "arm_a_s0", "job_id": "1000",
                                "status": "completed", "git_sha": "abc",
                                "phase": "P"}]})
        self.root = os.path.join(self.tmp, "results", "telemetry")

    def test_stale_execution_is_pruned(self):
        migrate.migrate(self.tmp, self.root)
        ghost_dir = os.path.join(self.root, "ghost__r0123456789abcdef", "events")
        os.makedirs(ghost_dir)
        ghost = "r0123456789abcdef:9999:a0"
        with open(os.path.join(ghost_dir, ghost + ".jsonl"), "w") as handle:
            handle.write("{}\n")
        state = migrate._load_state(self.root)
        state["executions"][ghost] = "deadbeef"
        migrate._save_state(self.root, state)

        report = migrate.migrate(self.tmp, self.root)
        self.assertEqual(report["pruned_stale_executions"], [ghost])
        self.assertFalse(os.path.exists(os.path.dirname(ghost_dir)))
        self.assertNotIn(ghost, migrate._load_state(self.root)["executions"])

    def test_live_executions_survive_pruning(self):
        migrate.migrate(self.tmp, self.root)
        before = sorted(os.listdir(self.root))
        report = migrate.migrate(self.tmp, self.root)
        self.assertEqual(report["pruned_stale_executions"], [])
        self.assertEqual(sorted(os.listdir(self.root)), before)
