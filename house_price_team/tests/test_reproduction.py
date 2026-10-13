"""Regression, leakage and failure-reporting contracts for the team handoff."""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_experiments as runner
import team_core as core


class RegressionContracts(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(71)
        self.raw={n:rng.normal(size=(24,8)).astype('float32') for n in runner.NAMES}
        self.y=rng.normal(650,100,24)
        self.fit=np.arange(16);self.val=np.arange(16,24)
        self.specs=runner.read(ROOT/'METHODS.json')['methods']

    def test_assessment_labels_cannot_change_predictions(self):
        changed=self.y.copy();changed[self.val]=np.nan
        for spec in self.specs.values():
            first,_=runner.predict_fit(self.raw,self.y,self.fit,self.val,spec)
            second,_=runner.predict_fit(self.raw,changed,self.fit,self.val,spec)
            np.testing.assert_array_equal(first,second)

    def test_fit_assessment_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            runner.predict_fit(self.raw,self.y,self.fit,np.array([2,18]),self.specs['v1'])

    def test_all_models_roundtrip_from_portable_numeric_state(self):
        for name,spec in self.specs.items():
            expected,state=runner.predict_fit(self.raw,self.y,self.fit,self.val,spec)
            with tempfile.TemporaryDirectory() as temp:
                file=Path(temp)/'model.npz';np.savez(file,**state)
                with np.load(file,allow_pickle=False) as z: restored={k:z[k].copy() for k in z.files}
            if name=='baseline':
                x=core.normalize(self.raw['register_large336'][self.val])
                actual=np.maximum((x@restored['coef']+restored['intercept'])*restored['target_scale']+restored['target_mean'],1.)
            else:
                restored['weights']=spec['weights']
                needed=core.expanded_weights(spec['weights'])
                tr={n:core.normalize(self.raw[n][self.fit]) for n in needed}
                ev={n:core.normalize(self.raw[n][self.val]) for n in needed}
                actual=core.predict(restored,tr,ev)
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)

    def test_weighted_kernel_matches_concatenated_scaled_features(self):
        features={n:core.normalize(x) for n,x in self.raw.items()}
        weights=core.expanded_weights(self.specs['v3']['weights'])
        joined=np.concatenate([np.sqrt(w)*features[n] for n,w in weights.items()],axis=1)
        expected=np.exp(-2*core.distance(joined,joined))
        actual=core.kernel(features,features,self.specs['v3']['weights'],2.)
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_pairing_rejects_missing_or_wrong_oof_keys(self):
        frame=pd.DataFrame({'method':['baseline']*2+['v1']*2,'key':['a','b','a','x'],
                            'actual':[1.,2.,1.,2.],'prediction':[2.,3.,2.,3.],'group':[1,2,1,2]})
        with self.assertRaisesRegex(ValueError,'keys differ'): runner.descriptive_intervals(frame,['baseline','v1'],samples=2)

    def test_failure_archives_evidence_and_never_publishes_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'data').mkdir();(root/'logs').mkdir()
            shutil.copy2(ROOT/'METHODS.json',root/'METHODS.json')
            for name in ['ROWS.csv','PROVENANCE.json']: (root/'data'/name).write_text('fixture')
            args=argparse.Namespace(root=str(root),method='all',features=None,no_refit=False)
            with patch.object(runner,'load',side_effect=ValueError('injected mismatch')):
                result=runner.run(args)
            self.assertEqual(result,1)
            latest=runner.read(root/'LATEST_RUN.json');out=root/latest['run']
            self.assertEqual(runner.read(out/'STATUS.json')['status'],'failed')
            self.assertFalse(list((out/'submissions').glob('*.csv')))
            with zipfile.ZipFile(root/'TEAM_RESULTS.zip') as z:
                self.assertEqual(json.loads(z.read('run/STATUS.json'))['status'],'failed')
                self.assertIsNone(z.testzip())

    def test_group_leakage_and_nonfinite_labels_rejected(self):
        rows=pd.read_csv(ROOT/'data/ROWS.csv',float_precision='round_trip')
        train=pd.read_csv(ROOT/'data/train.csv',float_precision='round_trip');test=pd.read_csv(ROOT/'data/test.csv')
        runner.validate_rows(rows,train,test)
        changed=rows.copy();pool=changed.partition.eq('inner_train')
        first=changed.index[pool&changed.outer_fold.eq(0)][0];second=changed.index[pool&changed.outer_fold.eq(1)][0]
        changed.loc[second,'group']=changed.loc[first,'group']
        with self.assertRaisesRegex(ValueError,'crosses folds'): runner.validate_rows(changed,train,test)
        with self.assertRaisesRegex(ValueError,'fit labels'):
            y=self.y.copy();y[0]=np.nan
            runner.predict_fit(self.raw,y,self.fit,self.val,self.specs['baseline'])

    def test_delivery_failure_cannot_leave_success_or_old_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'data').mkdir();(root/'logs').mkdir()
            shutil.copy2(ROOT/'METHODS.json',root/'METHODS.json')
            for name in ['ROWS.csv','PROVENANCE.json']: (root/'data'/name).write_text('fixture')
            (root/'TEAM_RESULTS.zip').write_bytes(b'old run archive')
            args=argparse.Namespace(root=str(root),method='all',features=None,no_refit=True)
            with patch.object(runner,'load',side_effect=ValueError('injected input failure')), patch.object(runner,'collect',side_effect=OSError('disk full')):
                result=runner.run(args)
            self.assertEqual(result,1)
            out=root/runner.read(root/'LATEST_RUN.json')['run']
            self.assertEqual(runner.read(out/'STATUS.json')['status'],'delivery_failed')
            self.assertFalse((root/'TEAM_RESULTS.zip').exists())
            self.assertEqual(len(list((root/'result_history').glob('*.zip'))),1)

    def test_unit_conversion_and_paired_sign(self):
        a=np.array([100.,200.]);p=np.array([110.,180.])
        self.assertEqual(core.metrics(a,p)['mse'],250.)
        frame=pd.DataFrame({'method':['baseline']*2+['v1']*2,'key':['a','b']*2,
                            'actual':[100.,200.]*2,'prediction':[110.,180.,105.,195.],'group':[1,2]*2})
        result=runner.descriptive_intervals(frame,['baseline','v1'],samples=20)[0]
        self.assertEqual(result['delta_mse'],-225.)


if __name__=='__main__': unittest.main()
