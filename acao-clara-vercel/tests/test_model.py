import unittest, sys
from unittest.mock import patch
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from app import features, calculate, split_indices, ranking_item, UserError

class ModelTests(unittest.TestCase):
    def fixture(self,n=1400):
        rng=np.random.default_rng(41)
        return pd.DataFrame({'date':pd.bdate_range('2016-01-01',periods=n).date,'close':100*np.exp(np.cumsum(rng.normal(0,.013,n))), 'volume':1000})
    def test_features_do_not_use_future(self):
        df=self.fixture(); before=features(df).iloc[300].copy(); df.loc[301:,'close']*=7
        pd.testing.assert_series_equal(before,features(df).iloc[300])
    def test_walkforward_and_valid_probabilities(self):
        r=calculate(self.fixture(),20)
        self.assertTrue(0<=r['probability']<=1)
        self.assertTrue(0<=r['brier']<=1)
        self.assertTrue(np.isfinite(r['projected_return']))
        self.assertTrue(r['projected_return']> -1)
        self.assertGreaterEqual(r['price_mae'],0)
        self.assertTrue(r['count']>=200)
        self.assertEqual(sum(b['count'] for b in r['bins']),r['count'])
    def test_ranking_uses_same_direction_model(self):
        df=self.fixture()
        df['date']=pd.bdate_range(end=datetime.now(ZoneInfo('America/Sao_Paulo')).date(),periods=len(df)).date
        expected=calculate(df,20)['probability']
        with patch('app.history',return_value=(df,{})):
            item=ranking_item({'symbol':'PETR4.SA','name':'PETR4'},20)
        self.assertIsNotNone(item)
        self.assertAlmostEqual(expected,item['probability'])
    def test_training_labels_finish_before_test(self):
        valid=np.ones(1000,dtype=bool)
        for horizon in (5,10,20,60):
            train,test=split_indices(valid,700,800,horizon,900)
            self.assertLess(int(train.max())+horizon,int(test.min()))
            self.assertTrue((test<=900).all())
    def test_insufficient(self):
        with self.assertRaises(UserError): calculate(self.fixture(200),20)
    def test_later_data_cannot_change_first_test_when_before_boundary(self):
        # The feature calculation is causal even across a large future perturbation.
        df=self.fixture(); old=features(df).iloc[:1000].copy(); df.loc[1000:,'close']*=4
        pd.testing.assert_frame_equal(old,features(df).iloc[:1000])
if __name__=='__main__': unittest.main()
