"""The recorded state must match what LeRobot's LIBERO eval pipeline computes."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
env_processor = pytest.importorskip("lerobot.processor.env_processor")


def test_axis_angle_matches_lerobot_processor():
    from p2p.sim import quat2axisangle

    rng = np.random.default_rng(0)
    q = rng.normal(size=(200, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # include the regime that matters: gripper pointing down, w close to 0 on both sides
    q[:4] = [[0.9996, 0.0002, -0.0284, 0.0], [0.9996, 0.0, 0.0284, -0.01],
             [0.9996, 0.0, 0.0284, 0.01], [1.0, 0.0, 0.0, 0.0]]
    ref = env_processor.LiberoProcessorStep()._quat2axisangle(torch.tensor(q)).numpy()
    ours = np.array([quat2axisangle(x) for x in q])
    np.testing.assert_allclose(ours, ref, atol=1e-5)
