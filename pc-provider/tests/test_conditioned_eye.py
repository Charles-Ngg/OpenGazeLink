from types import SimpleNamespace
import unittest

import numpy as np

from opengazelink_pc import features as f
from opengazelink_pc.conditioned_eye import eye_inputs
from opengazelink_pc.conditioned_eye_model import stabilize_runtime_geometry


def synthetic_scene():
    # Front-facing camera, known metric eye origins, asymmetric horizontal texture.
    camera = dict(fx=500., fy=500., cx=159.5, cy=119.5)
    rotation = np.diag([1., -1., -1.])
    translation = np.array([0., 0., 60.])
    landmarks = [SimpleNamespace(x=.5, y=.5, z=0.) for _ in range(478)]
    for prefix in ('RIGHT', 'LEFT'):
        center = getattr(f, prefix+'_EYE_CANONICAL_CENTER')
        indices = getattr(f, prefix+'_EYE_CONTOUR')
        sign = -1 if prefix=='RIGHT' else 1
        for j,index in enumerate(indices):
            theta=2*np.pi*j/len(indices)
            point=center+[sign*1.5*np.cos(theta),-.45*np.sin(theta),0.]
            xyz=rotation@point+translation
            landmarks[index]=SimpleNamespace(x=(500*xyz[0]/xyz[2]+159.5)/320,
                                               y=(500*xyz[1]/xyz[2]+119.5)/240,z=0.)
        xyz=rotation@center+translation
        for index in (*getattr(f,prefix+'_IRIS_RING'),*getattr(f,prefix+'_EYEBROW')):
            landmarks[index]=SimpleNamespace(x=(500*xyz[0]/xyz[2]+159.5)/320,
                                               y=(500*xyz[1]/xyz[2]+119.5)/240,z=0.)
    landmarks[127]=SimpleNamespace(x=.2,y=.5,z=0.)
    landmarks[356]=SimpleNamespace(x=.8,y=.5,z=0.)
    frame=np.repeat(np.tile(np.arange(320,dtype=float)*.7,(240,1))[...,None],3,axis=2).astype(np.uint8)
    pose=dict(rotation=rotation.tolist(),translation=translation.tolist(),reprojectionErrorPx=0.)
    return frame,landmarks,pose,camera


def test_runtime_geometry_reference_replaces_capture_route_constants():
    raw = np.arange(140, dtype=np.float32).reshape(2, 70)
    item = {"runtime_geometry_reference": {
        "indices": [66, 67], "values": [1.1787, 1.4145],
    }}
    actual = stabilize_runtime_geometry(raw, item)
    np.testing.assert_allclose(actual[:, 66:68], [[1.1787, 1.4145]] * 2)
    np.testing.assert_array_equal(actual[:, :66], raw[:, :66])
    np.testing.assert_array_equal(raw[:, 66:68], [[66, 67], [136, 137]])
    near = np.zeros((1, 70), dtype=np.float32)
    near[:, 66:68] = [1.2, 1.3]
    np.testing.assert_array_equal(stabilize_runtime_geometry(near, item)[:, 66:68], near[:, 66:68])


def test_mirror_preserves_physical_ray_and_texture_orientation():
    frame,lm,pose,camera=synthetic_scene()
    target=np.array([6.,-4.,0.])
    for side in ('right','left'):
        eye=eye_inputs(frame,lm,pose,camera,target,side)
        physical=eye['targets'].astype(float).copy()
        if side=='left':
            physical[0]*=-1
        expected=target-eye['center']
        expected/=np.linalg.norm(expected)
        np.testing.assert_allclose(eye['rotation']@physical,expected,atol=1e-6)
        for view in eye['images']:
            delta=int(view[0,18,49])-int(view[0,18,15])
            assert delta>0 if side=='right' else delta<0
            assert view[1].min()==255


def test_target_changes_do_not_change_model_inputs():
    frame,lm,pose,camera=synthetic_scene()
    a=eye_inputs(frame,lm,pose,camera,[0,0,0],'right')
    b=eye_inputs(frame,lm,pose,camera,[12,-8,0],'right')
    for key in ('images','points','head','crop','rotation','center'):
        np.testing.assert_array_equal(a[key],b[key])
    assert not np.allclose(a['targets'],b['targets'])


class ConditionedEyeTests(unittest.TestCase):
    test_geometry_reference = staticmethod(test_runtime_geometry_reference_replaces_capture_route_constants)
    test_mirror = staticmethod(test_mirror_preserves_physical_ray_and_texture_orientation)
    test_no_target_leakage = staticmethod(test_target_changes_do_not_change_model_inputs)
