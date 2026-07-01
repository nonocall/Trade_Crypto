import unittest
import numpy as np
import pandas as pd
from data_handler import generate_dummy_data_15m, resample_to_htf
from strategy import calculate_htf_indicators, generate_mtf_signals

class TestMTFStrategy(unittest.TestCase):
    
    def test_resampling_alignment(self):
        """
        Tests resampling from 15m to 2H.
        """
        df_15m = generate_dummy_data_15m(length=100)
        df_htf = resample_to_htf(df_15m, htf_interval='2h')
        
        # 100 15m candles resampled to 2h (8 candles per 2h)
        # Should yield around ceil(100/8) = 13 candles
        self.assertTrue(len(df_htf) >= 12 and len(df_htf) <= 14)
        
        # Check standard columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            self.assertIn(col, df_htf.columns)
            
    def test_htf_indicators(self):
        """
        Tests that HTF indicator calculation generates appropriate columns.
        """
        df_15m = generate_dummy_data_15m(length=500)
        df_htf = resample_to_htf(df_15m, htf_interval='2h')
        df_htf_calc = calculate_htf_indicators(df_htf, N_htf=6, roc_period=3)
        
        required = ['ROC', 'Rolling_Min_Low', 'Rolling_Max_High', 'ROC_at_Min_Low', 'ROC_at_Max_High', 'PRZ_Buy', 'PRZ_Sell']
        for col in required:
            self.assertIn(col, df_htf_calc.columns)
            
    def test_mtf_lookahead_bias_prevention(self):
        """
        Verifies that lookahead-bias-free alignment mapping is correct.
        Assures that the merged HTF start time matches strictly closed bars.
        """
        df_15m = generate_dummy_data_15m(length=24) # 6 hours of data
        # Let's say:
        # T0: 2026-01-01 00:00:00 to 02:00:00 (starts at 00:00, 00:15... ends with 01:45 candle)
        # T1: 2026-01-01 02:00:00 to 04:00:00 (starts at 02:00, 02:15... ends with 03:45 candle)
        
        df_htf = resample_to_htf(df_15m, htf_interval='2h')
        df_signals = generate_mtf_signals(df_15m, df_htf, N_htf=2, roc_period_htf=1, N_ltf=5, atr_period_ltf=5, htf_interval='2h')
        
        # For a candle at 02:15:00, the last closed 2H bar is the 00:00:00 bar (which closed at 02:00:00).
        # Check alignment mapping column
        row_215 = df_signals[df_signals['Timestamp'] == '2026-01-01 02:15:00'].iloc[0]
        self.assertEqual(row_215['HTF_Closed_Start_Time'], pd.Timestamp('2026-01-01 00:00:00'))
        
        # For a candle at 04:00:00, the last closed 2H bar is the 02:00:00 bar (which closed at 04:00:00).
        row_400 = df_signals[df_signals['Timestamp'] == '2026-01-01 04:00:00'].iloc[0]
        self.assertEqual(row_400['HTF_Closed_Start_Time'], pd.Timestamp('2026-01-01 02:00:00'))

if __name__ == '__main__':
    unittest.main()
