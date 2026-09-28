import numpy as np

from gsedit.reconstruction.fit_completed_wall import plane_support


def test_plane_support_rejects_misaligned_wall():
    origin = np.zeros(3)
    frame = np.eye(3)
    points = np.array([[2., y, 1.] for y in np.linspace(0, 2, 100)])
    ratio, q90 = plane_support(points, origin, frame, np.array([1., 0.]),
                               2., .1)
    assert ratio == 1 and q90 == 0
    ratio, q90 = plane_support(points, origin, frame, np.array([1., 0.]),
                               3., .1)
    assert ratio == 0 and q90 == 1
