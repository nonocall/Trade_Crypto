import numpy as np
import pandas as pd

def calculate_htf_indicators(df_htf: pd.DataFrame, N_htf: int = 12, roc_period: int = 6) -> pd.DataFrame:
    """
    大周期过滤器 (HTF Filter)
    Calculates technical indicators on the HTF (e.g. 2H) DataFrame:
    - ROC (Rate of Change)
    - Rolling extremes and the index where they occurred
    - ROC at the rolling extremes to detect momentum divergence
    - HTF PRZ (Potential Reversal Zone) flags
    """
    df = df_htf.copy()
    
    # 1. HTF ROC (Rate of Change)
    df['ROC'] = df['Close'].pct_change(periods=roc_period) * 100.0
    
    # 2. Rolling extremes of the past N_htf periods (excluding current candle)
    df['Rolling_Min_Low'] = df['Low'].shift(1).rolling(window=N_htf).min()
    df['Rolling_Max_High'] = df['High'].shift(1).rolling(window=N_htf).max()
    
    # 3. Vectorized identification of ROC at the rolling min/max indices
    roc_at_min_low = np.full(len(df), np.nan)
    roc_at_max_high = np.full(len(df), np.nan)
    
    if len(df) > N_htf:
        low_matrix = np.column_stack([df['Low'].shift(i).values for i in range(1, N_htf + 1)])
        high_matrix = np.column_stack([df['High'].shift(i).values for i in range(1, N_htf + 1)])
        
        valid_low_matrix = np.where(np.isnan(low_matrix[N_htf:]), np.inf, low_matrix[N_htf:])
        valid_high_matrix = np.where(np.isnan(high_matrix[N_htf:]), -np.inf, high_matrix[N_htf:])
        
        min_shift_idx = np.argmin(valid_low_matrix, axis=1)
        max_shift_idx = np.argmax(valid_high_matrix, axis=1)
        
        t_indices = np.arange(N_htf, len(df))
        abs_min_low_idx = t_indices - (min_shift_idx + 1)
        abs_max_high_idx = t_indices - (max_shift_idx + 1)
        
        roc_values = df['ROC'].values
        roc_at_min_low[N_htf:] = roc_values[abs_min_low_idx.astype(int)]
        roc_at_max_high[N_htf:] = roc_values[abs_max_high_idx.astype(int)]
        
    df['ROC_at_Min_Low'] = roc_at_min_low
    df['ROC_at_Max_High'] = roc_at_max_high
    
    # 4. Generate HTF PRZ (Potential Reversal Zone) Signals
    # HTF Bullish Divergence SFP: Price breaks HTF rolling low, closes above it, with higher ROC
    df['PRZ_Buy'] = (
        (df['Low'] < df['Rolling_Min_Low']) & 
        (df['Close'] > df['Rolling_Min_Low']) & 
        (df['ROC'] > df['ROC_at_Min_Low'])
    )
    
    # HTF Bearish Divergence SFP: Price breaks HTF rolling high, closes below it, with lower ROC
    df['PRZ_Sell'] = (
        (df['High'] > df['Rolling_Max_High']) & 
        (df['Close'] < df['Rolling_Max_High']) & 
        (df['ROC'] < df['ROC_at_Max_High'])
    )
    
    return df

def generate_mtf_signals(df_15m: pd.DataFrame, df_htf: pd.DataFrame, 
                         N_htf: int = 12, roc_period_htf: int = 6, 
                         N_ltf: int = 20, atr_period_ltf: int = 14, 
                         htf_interval: str = '2h') -> pd.DataFrame:
    """
    Multi-Timeframe Strategy logic.
    Aligns HTF indicators lookahead-bias-free, and evaluates SFP entry triggers on 15m.
    """
    # 1. Calculate HTF indicators
    df_htf_calc = calculate_htf_indicators(df_htf, N_htf=N_htf, roc_period=roc_period_htf)
    
    # 2. Prep LTF DataFrame and calculate LTF ATR (used for Stop Loss)
    df_ltf = df_15m.copy()
    
    # Calculate LTF ATR
    prev_close_ltf = df_ltf['Close'].shift(1)
    tr1 = df_ltf['High'] - df_ltf['Low']
    tr2 = (df_ltf['High'] - prev_close_ltf).abs()
    tr3 = (df_ltf['Low'] - prev_close_ltf).abs()
    df_ltf['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_ltf['ATR'] = df_ltf['TR'].rolling(window=atr_period_ltf).mean()
    df_ltf = df_ltf.drop(columns=['TR'])
    
    # Calculate LTF Rolling Extremes (for SFP check)
    df_ltf['Rolling_Min_Low'] = df_ltf['Low'].shift(1).rolling(window=N_ltf).min()
    df_ltf['Rolling_Max_High'] = df_ltf['High'].shift(1).rolling(window=N_ltf).max()
    
    # 3. LOOKAHEAD-BIAS-FREE ALIGNMENT
    # To prevent lookahead bias, a 15m candle starting at T can only use HTF data closed before T.
    # Therefore, the 15m candle starting at T maps to the HTF candle starting at (T.floor(HTF) - 1 HTF period).
    time_delta = pd.Timedelta(htf_interval)
    df_ltf['HTF_Closed_Start_Time'] = df_ltf['Timestamp'].dt.floor(htf_interval) - time_delta
    
    # Merge HTF indicators into LTF DataFrame
    # Prefix HTF columns to avoid name collisions
    htf_cols_rename = {col: f'HTF_{col}' for col in df_htf_calc.columns if col != 'Timestamp'}
    df_htf_calc_renamed = df_htf_calc.rename(columns=htf_cols_rename)
    
    df_merged = pd.merge(
        df_ltf,
        df_htf_calc_renamed,
        left_on='HTF_Closed_Start_Time',
        right_on='Timestamp',
        how='left',
        suffixes=('', '_htf_col')
    )
    
    # Clean up redundant timestamp column from merge
    if 'Timestamp_htf_col' in df_merged.columns:
        df_merged = df_merged.drop(columns=['Timestamp_htf_col'])
        
    # Forward-fill HTF signals in case of missing intervals
    htf_cols = list(htf_cols_rename.values())
    df_merged[htf_cols] = df_merged[htf_cols].ffill()
    
    # Ensure boolean type for PRZ signals after merge/ffill
    df_merged['HTF_PRZ_Buy'] = np.where(df_merged['HTF_PRZ_Buy'] == True, True, False)
    df_merged['HTF_PRZ_Sell'] = np.where(df_merged['HTF_PRZ_Sell'] == True, True, False)
    
    # 4. 小周期触发器 (LTF Trigger)
    # Check SFP conditions on 15m: Low sweeps rolling min and Close reclaims it, or High sweeps rolling max and Close drops below it.
    df_merged['LTF_SFP_Buy'] = (df_merged['Low'] < df_merged['Rolling_Min_Low']) & (df_merged['Close'] > df_merged['Rolling_Min_Low'])
    df_merged['LTF_SFP_Sell'] = (df_merged['High'] > df_merged['Rolling_Max_High']) & (df_merged['Close'] < df_merged['Rolling_Max_High'])
    
    # Combined Multi-Timeframe Signals
    # Buy when HTF is in PRZ_Buy and LTF SFP Buy occurs
    df_merged['Buy_Signal'] = df_merged['HTF_PRZ_Buy'] & df_merged['LTF_SFP_Buy']
    
    # Sell when HTF is in PRZ_Sell and LTF SFP Sell occurs
    df_merged['Sell_Signal'] = df_merged['HTF_PRZ_Sell'] & df_merged['LTF_SFP_Sell']
    
    # Signal representation: 1 for Buy, -1 for Sell, 0 for Hold
    df_merged['Signal'] = 0
    df_merged.loc[df_merged['Buy_Signal'], 'Signal'] = 1
    df_merged.loc[df_merged['Sell_Signal'], 'Signal'] = -1
    
    return df_merged

if __name__ == '__main__':
    from data_handler import generate_dummy_data_15m, resample_to_htf
    df_15m = generate_dummy_data_15m(1000)
    df_2h = resample_to_htf(df_15m, '2h')
    df_signals = generate_mtf_signals(df_15m, df_2h, N_htf=12, roc_period_htf=6, N_ltf=20, atr_period_ltf=14)
    print("MTF Strategy ran successfully!")
    print(f"Buy signals triggered: {df_signals['Buy_Signal'].sum()}")
    print(f"Sell signals triggered: {df_signals['Sell_Signal'].sum()}")
