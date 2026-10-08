import numpy as np
from scipy.spatial.transform import Rotation

from p2p import retarget as rt

K = np.array([[1780.0, 0, 960], [0, 1780.0, 540], [0, 0, 1]])


def test_ray_plane_recovers_points_on_the_plane():
    """Project known points at height z with a tilted, moving camera, then
    intersect their rays with that plane."""
    rng = np.random.default_rng(0)
    n, z = 20, 0.06
    pts = np.column_stack([rng.uniform(0.1, 0.5, n), rng.uniform(-0.3, 0.1, n), np.full(n, z)])
    T = np.repeat(np.eye(4)[None], n, 0)
    px = np.zeros((n, 2))
    for i in range(n):
        R = Rotation.from_euler("xyz", [180 + 40 + i, 3, 5 - i], degrees=True).as_matrix()
        cam = np.array([0.3, -0.5, 0.45]) + 0.004 * i
        T[i, :3, :3], T[i, :3, 3] = R, -R @ cam
        pc = R @ (pts[i] - cam)
        px[i] = (K @ (pc / pc[2]))[:2]
    px[3] = np.nan       # hand not found
    T[7] = np.nan        # board not found
    out = rt.rays_to_plane(px, T, K, np.zeros(5), np.full(n, z))
    ok = np.ones(n, bool)
    ok[[3, 7]] = False
    np.testing.assert_allclose(out[ok], pts[ok], atol=1e-9)
    assert np.isnan(out[~ok]).all()


def test_contacts_are_the_last_two_places():
    hz = 20
    hold = lambda p, s: np.repeat(np.array(p, float)[None], int(s * hz), 0)  # noqa: E731
    move = lambda a, b, s: np.linspace(a, b, int(s * hz))  # noqa: E731
    xy = np.vstack([hold([0.0, -0.3], 0.6),            # hand still before the reach
                    move([0.0, -0.3], [0.30, -0.2], 1.0),
                    hold([0.30, -0.2], 0.8),          # at the bowl
                    move([0.30, -0.2], [0.32, -0.19], 0.4),
                    hold([0.32, -0.19], 0.5),         # a short adjustment, same place
                    move([0.32, -0.19], [0.40, 0.0], 1.0),
                    hold([0.40, 0.0], 0.8),           # at the plate
                    move([0.40, 0.0], [0.41, 0.01], 0.3),
                    hold([0.41, 0.01], 0.6),          # hand raised above the plate
                    move([0.41, 0.01], [0.2, -0.3], 1.0)])
    xy += np.random.default_rng(1).normal(0, 0.0005, xy.shape)
    g, r = rt.contacts(xy, rt.find_pauses(xy, hz))
    np.testing.assert_allclose(xy[g], [0.32, -0.19], atol=0.005)
    np.testing.assert_allclose(xy[r], [0.40, 0.0], atol=0.01)
    assert g < r


def test_cone_height_is_zero_at_the_contacts_and_capped():
    xy = np.column_stack([np.linspace(-0.3, 0.6, 91), np.zeros(91)])
    g, r = 30, 60  # x = 0.0 and x = 0.3
    h = rt.cone_height(xy, g, r, slope=1.0, ceiling=0.1)
    assert h[g] == 0 and h[r] == 0
    np.testing.assert_allclose(h[45], 0.1)          # mid carry, 15 cm from both: capped
    np.testing.assert_allclose(h[28], 0.02, atol=1e-12)  # 2 cm before the grasp point
    assert h.max() <= 0.1



def test_straight_carry_keeps_the_ends_and_the_rest():
    rng = np.random.default_rng(3)
    xy = np.cumsum(rng.normal(0, 0.01, (60, 2)), axis=0)
    g, r = 15, 45
    out = rt.straight_carry(xy, g, r)
    np.testing.assert_array_equal(out[:g + 1], xy[:g + 1])
    np.testing.assert_array_equal(out[r:], xy[r:])
    # on the segment, and monotonic from grasp to release
    d = xy[r] - xy[g]
    proj = (out[g:r + 1] - xy[g]) @ d / (d @ d)
    np.testing.assert_allclose(out[g:r + 1], xy[g] + proj[:, None] * d, atol=1e-12)
    assert np.all(np.diff(proj) >= 0)
