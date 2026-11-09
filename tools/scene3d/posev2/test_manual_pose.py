"""09e CPU rigid transform/cache regression tests."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import manual_pose

class Manual(unittest.TestCase):
    def test_translation_camera_right_not_world_x(self):
        plate=SimpleNamespace(R=np.array([[0.,-1,0],[1,0,0],[0,0,1.]]))
        X=np.array([[0.,0,0],[1,2,3.]])
        K=np.zeros((70,3));K[69]=[0,0,1]
        A,B,_=manual_pose.transform(X,K,plate,{'body_dx':.3})
        np.testing.assert_allclose(A-X,np.tile([0,.3,0],(2,1)),atol=1e-12)
        np.testing.assert_allclose(B-K,np.tile([0,.3,0],(70,1)),atol=1e-12)

    def test_rotation_preserves_distances_and_pelvis(self):
        rng=np.random.default_rng(2);X=rng.normal(size=(30,3));K=rng.normal(size=(70,3))
        plate=SimpleNamespace(R=np.eye(3))
        A,B,_=manual_pose.transform(X,K,plate,{'body_dyaw':20,'body_dpitch':15,'body_droll':10})
        np.testing.assert_allclose(np.linalg.norm(A[:,None]-A,axis=2),np.linalg.norm(X[:,None]-X,axis=2),atol=1e-12)
        np.testing.assert_allclose((K[9]+K[10])/2,(B[9]+B[10])/2,atol=1e-12)

    def test_no_transform_identical(self):
        rng=np.random.default_rng(3);X=rng.normal(size=(30,3));K=rng.normal(size=(70,3))
        A,B,_=manual_pose.transform(X,K,SimpleNamespace(R=np.eye(3)),{})
        np.testing.assert_allclose(A,X,atol=1e-12);np.testing.assert_allclose(B,K,atol=1e-12)

    def test_key_changes_with_ref_seed_and_height_not_transforms(self):
        import tempfile
        import json
        from unittest.mock import patch
        import shot_spec
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'config').mkdir();(root/'config/placement.yaml').write_text('threshold: .01')
            assets=root/'assets';assets.mkdir()
            for name in ['plate.json','plate_mesh.obj','depth_loc.png']:(assets/name).write_bytes(name.encode())
            ref=root/'ref.npz';ref.write_bytes(b'ref')
            pp=root/'profile.json';pp.write_text('{}')
            profile=SimpleNamespace(path=pp,dir=root,data={'loc_id':'loc','objects':[]})
            cfg={'repo':str(root),'location_asset_dirs':{'loc':str(assets)}}
            spec=shot_spec.ShotSpec('s','stand','floor',['foot.L'])
            key=manual_pose.placement_key(ref,profile,cfg,spec,1.62,np.zeros(3),{'seed':101})
            cfg['body_dx']=.3;cfg['props']='changed'
            self.assertEqual(key,manual_pose.placement_key(ref,profile,cfg,spec,1.62,np.zeros(3),{'seed':101}))
            self.assertNotEqual(key,manual_pose.placement_key(ref,profile,cfg,spec,1.7,np.zeros(3),{'seed':101}))
            self.assertNotEqual(key,manual_pose.placement_key(ref,profile,cfg,spec,1.62,np.zeros(3),{'seed':102}))
            ref.write_bytes(b'changed-ref')
            self.assertNotEqual(key,manual_pose.placement_key(ref,profile,cfg,spec,1.62,np.zeros(3),{'seed':101}))

    def test_cache_roundtrip_full_precision_and_key_miss(self):
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            X=np.array([[1./3,1./7,1./11]])
            K=np.full((70,3),1./13)
            np.savez_compressed(root/'manual_base.npz',X=X,KP=K,reg=np.array(['pelvis']),
                                groups=np.array(['pelvis:target:support']),g_pelvis=np.array([0]))
            (root/'complete.json').write_text(json.dumps({'key':'key'}))
            (root/'fit.json').write_text(json.dumps({'status':'partial'}))
            b=manual_pose.load_placement(root,'key')
            np.testing.assert_array_equal(b['X'],X)
            np.testing.assert_array_equal(b['KP'],K)
            self.assertEqual(b['groups'][0][0],'pelvis')
            self.assertIsNone(manual_pose.load_placement(root,'wrong-key'))

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            manual_pose.transform(np.zeros((3,3)),np.zeros((70,3)),SimpleNamespace(R=np.eye(3)),{'body_dx':float('nan')})

if __name__=='__main__':unittest.main()
