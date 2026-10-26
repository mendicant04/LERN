import unittest
import numpy as np
import torch
from lern import LERN

class LERNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(33)
        self.h=rng.normal(size=(173,13))
        self.v=rng.normal(size=(173,7))
        self.text=rng.normal(size=(4,7))
        self.ids=np.arange(173)+1000

    def run_stream(self,n=173,chunk=1):
        a=LERN(self.text,'synthetic',3,'cpu')
        return a,a.observe(self.v[:n],self.h[:n],self.ids[:n],chunk=chunk)

    def test_chunking_preserves_predictions_and_ranks(self):
        _,expected=self.run_stream()
        for chunk in [7,32,256]:
            _,actual=self.run_stream(chunk=chunk)
            for key in ['online_local_statistic','rank_numerators','rank_denominators','reference_prefix_end']:
                np.testing.assert_array_equal(actual[key],expected[key])

    def test_future_arrivals_cannot_change_prefix(self):
        _,full=self.run_stream(chunk=19)
        _,prefix=self.run_stream(n=95,chunk=31)
        for key in ['online_local_statistic','rank_numerators','rank_denominators']:
            np.testing.assert_array_equal(prefix[key],full[key][:95])
        self.assertTrue(np.all(full['reference_prefix_end']<full['stream_positions']))

    def test_cold_start_and_disjoint_banks(self):
        a,r=self.run_stream()
        np.testing.assert_array_equal(r['online_local_statistic'][:8],r['frozen'][:8])
        self.assertFalse(set(a.reference.ids[:a.reference.n].tolist()) & set(a.calibration.ids[:a.calibration.n].tolist()))
        self.assertEqual(a.reference.n+a.calibration.n,173)

    def test_duplicate_ids_and_nonfinite_features_are_rejected(self):
        a=LERN(self.text,'synthetic',0,'cpu')
        a.observe(self.v[:1],self.h[:1],self.ids[:1])
        with self.assertRaises(ValueError):a.observe(self.v[:1],self.h[:1],self.ids[:1])
        bad=self.v[:1].copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):a.observe(bad,self.h[1:2],self.ids[1:2])

if __name__=='__main__':unittest.main()
