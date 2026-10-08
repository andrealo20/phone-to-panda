import numpy as np

from p2p.sim_map import Demo, Scene, heading, map_demo, rot_z, trim

BELOW = 0.01
SIM = Scene(np.array([0.0, 0.2, 0.95]), 0.047, np.array([0.15, 0.1, 0.92]), 0.9)


def demo(grasp=np.array([0.30, 0.10, 0.07]), carry=np.array([0.25, -0.1, 0.02]),
         came_from=np.array([-0.1, 0.05, 0.10]), hold=0):
    """Reach the rim from `came_from`, close, optionally stay still `hold`
    frames, carry by `carry`, open, retract."""
    up = np.array([0, 0, 0.1])
    release = grasp + carry
    pinch = np.vstack([np.linspace(grasp + came_from, grasp, 20),
                       np.repeat(grasp[None], hold, 0).reshape(-1, 3),
                       np.linspace(grasp, grasp + up, 10),
                       np.linspace(grasp + up, release + up, 15),
                       np.linspace(release + up, release, 10),
                       np.linspace(release, release + up, 10)])
    g, r = 19 + hold, 19 + hold + 35
    grip = np.zeros(len(pinch))
    grip[g:r] = 1
    return Demo(pinch, grip, g, r)


def test_anchors_land_on_the_simulated_objects():
    d = demo()
    P, _ = map_demo(d, SIM, BELOW)
    g, r = d.grasp, d.release
    np.testing.assert_allclose(np.linalg.norm(P[g, :2] - SIM.bowl_c[:2]), SIM.bowl_r, atol=1e-12)
    np.testing.assert_allclose(P[g, 2], SIM.bowl_c[2] - BELOW, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(P[r, :2] - SIM.plate_c[:2]), SIM.bowl_r, atol=1e-12)
    np.testing.assert_allclose(P[r, 2], SIM.bowl_c[2] - BELOW + 0.02, atol=1e-12)


def test_grasp_is_on_the_side_the_hand_came_from():
    d = demo(came_from=np.array([-0.1, 0.0, 0.1]))
    P, h = map_demo(d, SIM, BELOW)
    outward = P[d.grasp - 10, :2] - P[d.grasp, :2]
    radial = P[d.grasp, :2] - SIM.bowl_c[:2]
    assert outward @ radial > 0
    # fingers close along the radial direction, i.e. across the rim wall
    np.testing.assert_allclose(h[0], heading(radial), atol=1e-12)


def test_rotating_the_real_demo_does_not_change_the_result():
    d = demo()
    P0, h0 = map_demo(d, SIM, BELOW)
    R = rot_z(1.1)
    P1, h1 = map_demo(Demo((R @ d.pinch.T).T, d.grip, d.grasp, d.release), SIM, BELOW)
    np.testing.assert_allclose(P1, P0, atol=1e-12)
    np.testing.assert_allclose(h1, h0, atol=1e-12)


def test_no_sideways_slide_while_the_fingers_close():
    d = demo(hold=8)
    d.grasp -= 8  # close at the start of the pause, then hold still
    P, _ = map_demo(d, SIM, BELOW)
    g = d.grasp
    np.testing.assert_allclose(P[g: g + 9], np.repeat(P[g][None], 9, 0), atol=1e-12)


def test_the_mirror_image_avoids_an_obstacle():
    d = demo(came_from=np.array([-0.1, 0.06, 0.1]))
    P0, h0 = map_demo(d, SIM, BELOW)
    side0 = (P0[d.grasp, :2] - SIM.bowl_c[:2]) / SIM.bowl_r
    # put an object right where the open fingers would go
    blocked = Scene(SIM.bowl_c, SIM.bowl_r, SIM.plate_c, SIM.table_z,
                    obstacles=np.array([SIM.bowl_c[:2] + 0.09 * side0]))
    P1, _ = map_demo(d, blocked, BELOW)
    side1 = (P1[d.grasp, :2] - SIM.bowl_c[:2]) / SIM.bowl_r
    u = (SIM.plate_c - SIM.bowl_c)[:2] / np.linalg.norm((SIM.plate_c - SIM.bowl_c)[:2])
    np.testing.assert_allclose(side1, 2 * (side0 @ u) * u - side0, atol=1e-12)  # mirrored
    np.testing.assert_allclose(np.linalg.norm(P1[d.release, :2] - SIM.plate_c[:2]), SIM.bowl_r, atol=1e-12)


def test_snapped_grasp_needs_no_wrist_turn():
    rest = np.pi / 2  # closing axis along y
    for came_from in ([-0.1, 0.05, 0.1], [0.02, 0.1, 0.1], [0.08, -0.07, 0.1]):
        d = demo(came_from=np.array(came_from))
        P, h = map_demo(d, SIM, BELOW, snap_heading=rest)
        side = (P[d.grasp, :2] - SIM.bowl_c[:2]) / SIM.bowl_r
        assert abs(abs(side[1]) - 1) < 1e-12 and abs(side[0]) < 1e-12  # on the y axis
        # same gripper axis as at rest, modulo the half turn the gripper is symmetric under
        np.testing.assert_allclose(np.sin(h[0] - rest), 0, atol=1e-12)


def test_trim_keeps_only_the_part_near_the_contacts():
    d = demo(came_from=np.array([-0.4, 0.0, 0.1]))  # a long reach from 40 cm away
    t = trim(d, reach=0.10)
    np.testing.assert_array_equal(t.pinch[t.grasp], d.pinch[d.grasp])
    np.testing.assert_array_equal(t.pinch[t.release], d.pinch[d.release])
    start_dist = np.linalg.norm(t.pinch[0, :2] - t.pinch[t.grasp, :2])
    assert start_dist < 0.10 and len(t.pinch) < len(d.pinch)
    # the retreat here only goes straight up, never 10 cm sideways: kept whole
    assert len(t.pinch) - t.release == len(d.pinch) - d.release
