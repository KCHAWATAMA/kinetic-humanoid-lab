"""Independent pinned-source semantics checked against the actual ONNX graph."""

import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from diagnose import CommandTrace, HISTORICAL, PROTOCOL, Trace, digest, inspect_graph, probe
from fetch_assets import MODEL_DIR
from rollout import load_model, make_inputs, make_session, reference_pose, run_case, verify_assets, xyzw


@unittest.skipUnless(importlib.util.find_spec("onnx"), "Install research/diagnostics-requirements.txt for graph audit tests.")
class AuthoritativeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(".research-cache/protomotions")
        verify_assets(cls.assets)
        directory = cls.assets / MODEL_DIR
        cls.meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text())
        cls.config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text())
        cls.default = np.asarray(cls.config["robot"]["default_dof_pos"])
        cls.model = load_model(cls.assets, cls.meta, cls.config)
        cls.session = make_session(directory / "compiled_models/unified_pipeline.onnx", cls.meta)
        cls.instrumented, cls.initializers, cls.graph = inspect_graph(directory / "compiled_models/unified_pipeline.onnx", cls.meta, cls.config)

    def test_graph_matches_source_observation_and_action_semantics(self):
        data = mujoco.MjData(self.model)
        data.qpos[:3] = [0.2, -0.3, 1.2]
        quaternion = np.array([0.86, 0.12, 0.20, 0.41])
        data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
        data.qpos[7:] = self.default + np.linspace(-0.1, 0.1, 29)
        data.qvel[:] = np.linspace(-0.3, 0.4, 35)
        mujoco.mj_forward(self.model, data)
        future = []
        for offset in (1, 2, 4, 8):
            angle = offset * 0.03
            future.append({"q": self.default + offset * 0.01, "v": np.linspace(-0.2, 0.3, 29) * offset,
                           "anchor": np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)])})
        previous = self.default + np.linspace(0.01, 0.05, 29)
        feed = make_inputs(data, self.meta, future, previous)
        np.testing.assert_allclose(feed["current_dof_pos"][0], data.qpos[7:])
        np.testing.assert_allclose(feed["current_dof_vel"][0], data.qvel[6:])
        np.testing.assert_allclose(feed["current_root_local_ang_vel"][0], data.qvel[3:6])
        np.testing.assert_allclose(data.xmat[1].reshape(3, 3).T @ data.cvel[1, :3], data.qvel[3:6], atol=1e-12)
        anchor_id = self.model.body("torso_link").id
        np.testing.assert_array_equal(feed["current_anchor_rot"][0], xyzw(data.xquat[anchor_id]))
        self.assertFalse(np.allclose(data.xquat[1], data.xquat[anchor_id]))
        measured = probe(self.instrumented, self.initializers, self.session, feed)
        self.assertLess(measured["observationMaxError"], 2e-6)
        self.assertEqual(len(measured["rawObservation"]), 349)
        self.assertEqual(self.graph["tanhNodeCount"], 0)
        self.assertEqual(set(self.graph["actionStructuralAncestors"]), set(feed))
        for value in feed.values():
            self.assertEqual(value.dtype, np.float32)

    def test_joint_axes_gains_and_gear_match_authoritative_assets(self):
        xml = ET.parse(self.assets / "protomotions/data/assets/mjcf/g1_holo_compat.xml")
        joints = {node.attrib["name"]: node for node in xml.iter("joint") if "name" in node.attrib}
        for index, name in enumerate(self.meta["robot"]["joint_names"]):
            np.testing.assert_allclose(self.model.jnt_axis[index + 1], np.fromstring(joints[name].attrib["axis"], sep=" "))
            info = self.config["robot"]["control"]["control_info"][name]
            self.assertEqual(self.model.actuator_trnid[index, 0], index + 1)
            self.assertAlmostEqual(self.model.actuator_gainprm[index, 0], float(info["stiffness"]))
            self.assertAlmostEqual(-self.model.actuator_biasprm[index, 2], float(info["damping"]))
            self.assertAlmostEqual(self.model.dof_armature[index + 6], float(info["armature"]))
        np.testing.assert_array_equal(self.model.actuator_gear[:, 0], np.ones(29))
        np.testing.assert_array_equal(self.model.actuator_gear[:, 1:], np.zeros((29, 5)))
        self.assertEqual(self.meta["control"]["action_ema_alpha"], 1)
        self.assertIsNone(self.meta["control"]["pd_target_max_accel"])
        self.assertEqual(self.meta["motion"]["future_step_indices"], [1, 2, 4, 8])
        np.testing.assert_allclose(self.meta["motion"]["future_dt_seconds"], [0.02, 0.04, 0.08, 0.16])

    def test_runtime_history_lookahead_and_observation_are_not_assumed(self):
        meta, default, session = self.meta, self.default, self.session
        captured = []

        class RecordingSession:
            def run(self, names, feed):
                step = len(captured)
                np.testing.assert_array_equal(feed["historical_processed_actions"][0, 0],
                                              captured[-1] if captured else np.zeros(29, np.float32))
                for slot, offset in enumerate((1, 2, 4, 8)):
                    q, velocity = reference_pose(default, meta["robot"]["joint_names"], (step + offset) * 0.02, "arm_sweep")
                    np.testing.assert_allclose(feed["mimic_future_dof_pos"][0, slot], q, atol=1e-7)
                    np.testing.assert_allclose(feed["mimic_future_dof_vel"][0, slot], velocity, atol=1e-7)
                outputs = session.run(names, feed)
                captured.append(outputs[1][0].copy())
                return outputs

        protocol = json.loads(PROTOCOL.read_text())
        trace = Trace(protocol, 0.02)
        with_trace = run_case(self.model, RecordingSession(), meta, default, "arm_sweep", 1,
                              observe=trace.observe, observe_substep=trace.substep)
        without_trace = run_case(self.model, session, meta, default, "arm_sweep", 1)
        self.assertEqual(with_trace["frames"], without_trace["frames"])
        self.assertEqual(trace.substeps, 1000)
        self.assertLess(trace.max_pd_force_error, 1e-10)
        self.assertEqual(len(captured), 50)

    def test_partial_trace_has_no_invented_post_hold_motion(self):
        protocol = json.loads(PROTOCOL.read_text())
        trace = Trace(protocol, 0.02)
        data = mujoco.MjData(self.model)
        data.qpos[2], data.qpos[3:7], data.qpos[7:] = 0.8, [1, 0, 0, 0], self.default
        mujoco.mj_forward(self.model, data)
        trace.observe(self.model, data, 0, {"q": self.default})
        report = trace.summarize(CommandTrace(self.session, None, self.meta))
        self.assertIsNone(report["commandedRangePerLegJointRad"])
        self.assertIsNone(report["minimumNormalForceAfterHoldN"])
        self.assertIsNone(report["commandVsRequestedRmsePerLegJointRad"])
        self.assertEqual(report["substeps"], 0)


class RecordedDiagnosisTests(unittest.TestCase):
    def test_recorded_diagnosis_preserves_evidence_and_recomputes_motion(self):
        result = json.loads(Path("public/diagnostics/contract-audit-v1.json").read_text())
        self.assertEqual(result["status"], "complete")
        self.assertEqual([item["name"] for item in result["cases"]], ["stand", "attempt-01", "attempt-02"])
        self.assertEqual(result["historicalSha256"], {name: digest(Path(name)) for name in HISTORICAL})
        for case in result["cases"]:
            trace = case["trace"]
            samples = np.asarray(trace["controlSamples"])
            commands = np.asarray(trace["commandLegJoints"])
            self.assertEqual(samples.shape, (401, 32))
            self.assertEqual(commands.shape, (400, 12))
            self.assertEqual(trace["substeps"], 8000)
            self.assertEqual(trace["maxHistoryErrorRad"], 0)
            self.assertEqual(trace["maxFutureInputError"], 0)
            self.assertLess(trace["maxPdForceEquationErrorNm"], 1e-10)
            after = samples[:, 0] >= 1
            np.testing.assert_allclose(np.ptp(samples[after, 1:13], axis=0), trace["realizedRangePerLegJointRad"], atol=1e-8)
            np.testing.assert_allclose(np.ptp(samples[after, 13:25], axis=0), trace["requestedRangePerLegJointRad"], atol=1e-8)
            np.testing.assert_allclose(np.ptp(commands[samples[:-1, 0] >= 1], axis=0), trace["commandedRangePerLegJointRad"], atol=1e-8)
            np.testing.assert_allclose(np.min(samples[after, 28:30], axis=0), trace["minimumNormalForceAfterHoldN"], atol=1e-8)
            if case["name"] != "stand":
                self.assertLessEqual(case["historicalRootMaxAbsoluteDeltaM"], 1e-10)
        probes = result["staticProbes"]["probes"]
        for item in probes[1:5]:
            self.assertEqual(sum(item["changedInputElements"].values()), 1)
            self.assertGreater(item["targetDeltaMaxRad"], 1e-5)
        passthrough = next(item for item in probes if item["name"] == "passthrough_position_only")
        self.assertEqual(sum(passthrough["changedInputElements"].values()), 0)
        self.assertEqual(passthrough["targetDeltaMaxRad"], 0)


if __name__ == "__main__":
    unittest.main()
