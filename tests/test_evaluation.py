import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from lern.cli import cached
from lern.evaluation import classification_metrics, summarize

class EvaluationTests(unittest.TestCase):
    def test_balanced_accuracy(self):
        result=classification_metrics([0,0,0,1],[0,0,0,0])
        self.assertEqual(result['ACC'],75.)
        self.assertEqual(result['BA'],50.)

    def test_cache_round_trip(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'features.npz'
            np.savez(path,h=rng.normal(size=(150,12)),v=rng.normal(size=(150,8)),
                     text=rng.normal(size=(4,8)),scale=10.,indices=np.arange(150),
                     delta=rng.normal(size=(150,5,12)),probe_positions=np.arange(150),
                     y=rng.integers(0,4,size=150))
            args=SimpleNamespace(input=str(path),task='synthetic',device='cpu',seeds=[0,1],
                                 output=str(Path(folder)/'out'),limit=0,overwrite=False,chunk=17)
            with contextlib.redirect_stdout(io.StringIO()):
                cached(args)
                result=summarize(args.output)
            self.assertEqual(result['seeds'],[0,1])
            self.assertEqual(result['tasks'],1)
            with np.load(Path(args.output)/'synthetic_seed0.npz') as z:
                np.testing.assert_array_equal(z['indices'],np.random.default_rng(0).permutation(150))
                self.assertEqual(z['predictions'].shape,(150,))
            with self.assertRaises(FileExistsError):
                cached(args)

    def test_incomplete_orders_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for task,seed in [('a',0),('a',1),('b',0)]:
                row=dict(task=task,seed=seed,n=3,ACC=20.,BA=30.,zero_shot_ACC=10.,zero_shot_BA=20.)
                (Path(folder)/f'{task}_seed{seed}.json').write_text(json.dumps(row))
            with self.assertRaises(ValueError):summarize(folder)

if __name__=='__main__':unittest.main()
