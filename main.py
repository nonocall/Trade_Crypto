import os
import pandas as pd
from data_handler import get_binance_btc_data, resample_to_htf, generate_dummy_data_15m
from strategy import generate_mtf_signals
from backtester import Backtester
from analyzer import calculate_performance_metrics, plot_backtest_results

def main():
    print("=" * 60)
    print(" BTC Multi-Timeframe Reversal & SFP Backtester ".center(60, "="))
    print("=" * 60)
    
    symbol = 'BNBUSDT'
    ltf_interval = '15m'
    htf_interval = '2h'
    total_candles = 4000  # ~41 days of 15m data
    proxy = 'http://127.0.0.1:7078'
    cache_file = 'bnb_15m_historical.csv'
    
    # 1. Load or Fetch BTC 15m Data
    print(f"\n[1/5] Retrieving 15m historical price data for {symbol}...")
    try:
        df_15m = get_binance_btc_data(
            symbol=symbol, 
            interval=ltf_interval, 
            total_candles=total_candles, 
            proxy=proxy, 
            cache_file=cache_file
        )
        print(f"      Successfully loaded {len(df_15m)} bars of real Binance BTC/USDT data.")
    except Exception as e:
        print(f"      [WARNING] Failed to fetch Binance data: {e}")
        print("      Falling back to generating synthetic 15m data for offline testing...")
        df_15m = generate_dummy_data_15m(length=total_candles, start_price=65000.0, seed=42)
        print(f"      Generated {len(df_15m)} bars of synthetic 15m data.")
        
    # 2. Resample to HTF (2H)
    print(f"\n[2/5] Resampling 15m data to Large Timeframe (HTF: {htf_interval})...")
    df_htf = resample_to_htf(df_15m, htf_interval=htf_interval)
    print(f"      Generated {len(df_htf)} bars of {htf_interval} data.")
    
    # 3. Calculate MTF Signals
    # HTF parameter definitions
    N_htf = 12            # lookback window for HTF swing points
    roc_period_htf = 6    # HTF ROC speed window
    # LTF parameter definitions
    N_ltf = 20            # lookback window for LTF swing points
    atr_period_ltf = 14   # LTF ATR window
    atr_mult = 2.0        # Stop Loss ATR multiplier
    
    print("\n[3/5] Computing Multi-Timeframe strategy indicators and signals...")
    print(f"      HTF (2H) Config: N={N_htf}, ROC_Period={roc_period_htf}")
    print(f"      LTF (15m) Config: N={N_ltf}, ATR_Period={atr_period_ltf}, SL_ATR_Mult={atr_mult}")
    
    df_signals = generate_mtf_signals(
        df_15m=df_15m, 
        df_htf=df_htf, 
        N_htf=N_htf, 
        roc_period_htf=roc_period_htf, 
        N_ltf=N_ltf, 
        atr_period_ltf=atr_period_ltf, 
        htf_interval=htf_interval
    )
    
    # Analyze signal counts
    buy_signals_count = df_signals['Buy_Signal'].sum()
    sell_signals_count = df_signals['Sell_Signal'].sum()
    htf_prz_buy_count = df_signals['HTF_PRZ_Buy'].sum()
    htf_prz_sell_count = df_signals['HTF_PRZ_Sell'].sum()
    
    print(f"      HTF Potential Reversal Zones (Bars): BUY={htf_prz_buy_count}, SELL={htf_prz_sell_count}")
    print(f"      LTF Micro SFP Execution Triggers: BUY={buy_signals_count}, SELL={sell_signals_count}")
    
    # 4. Simulate Backtest Execution
    print("\n[4/5] Running backtest simulation loop (compounding + fees)...")
    initial_cap = 10000.0
    fee_rate = 0.0006  # 0.06% exchange taker fee
    
    backtester = Backtester(df_signals, initial_capital=initial_cap, fee_rate=fee_rate)
    results = backtester.run(atr_mult=atr_mult)
    
    trades = results['trades']
    
    # Print performance report
    metrics = calculate_performance_metrics(trades, backtester.df, initial_capital=initial_cap)
    
    print("\n" + " MULTI-TIMEFRAME PERFORMANCE REPORT ".center(50, "*"))
    print(f"{'Metric':<30} | {'Value':<15}")
    print("-" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            if '%' in k or 'Return' in k or 'Drawdown' in k or 'Rate' in k:
                print(f"{k:<30} | {v:>12.2f}%")
            elif 'Factor' in k or 'Sharpe' in k:
                print(f"{k:<30} | {v:>13.2f}")
            else:
                print(f"{k:<30} | {v:>13.2f}")
        else:
            print(f"{k:<30} | {v:>13}")
    print("*" * 50)
    
    if not trades.empty:
        print(f"\nLast 5 Executed Trades:")
        cols = ['direction', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'return_pct', 'exit_reason']
        print(trades[cols].tail(5).to_string(index=False))
    else:
        print("\nNo trades executed during backtest period.")
        
    # 5. Export charts
    print("\n[5/5] Exporting visualization charts...")
    artifact_dir = r"C:\Users\Administrator\.gemini\antigravity-ide\brain\f1602911-0409-407a-9763-55e93aaefa8c"
    
    # Save in workspace
    workspace_chart_path = os.path.join(os.getcwd(), "backtest_report.png")
    plot_backtest_results(backtester.df, trades, workspace_chart_path)
    print(f"      Saved workspace chart: [backtest_report.png](file:///{workspace_chart_path.replace(chr(92), '/')})")
    
    # Save in artifacts directory
    if os.path.exists(artifact_dir):
        artifact_chart_path = os.path.join(artifact_dir, "backtest_report.png")
        plot_backtest_results(backtester.df, trades, artifact_chart_path)
        print(f"      Saved artifact chart: [artifact_chart](file:///{artifact_chart_path.replace(chr(92), '/')})")
        
    print("\n" + " Backtest Simulation Completed successfully! ".center(60, "="))

if __name__ == '__main__':
    main()
