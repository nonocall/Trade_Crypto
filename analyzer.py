import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_performance_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    """
    Computes professional performance metrics based on trade history and equity curve.
    """
    metrics = {}
    
    # 1. Equity curve metrics
    final_equity = equity_df['Equity'].iloc[-1]
    total_return_pct = (final_equity - initial_capital) / initial_capital * 100.0
    metrics['Total Return (%)'] = total_return_pct
    
    # Buy & Hold Return
    bh_return_pct = (equity_df['Close'].iloc[-1] - equity_df['Close'].iloc[0]) / equity_df['Close'].iloc[0] * 100.0
    metrics['Buy & Hold Return (%)'] = bh_return_pct
    
    # 2. Drawdown calculation
    equity = equity_df['Equity'].values
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / peaks * 100.0
    max_drawdown = np.max(drawdowns)
    metrics['Max Drawdown (%)'] = max_drawdown
    
    # 3. Trade logs metrics
    if trades_df.empty:
        metrics['Total Trades'] = 0
        metrics['Win Rate (%)'] = 0.0
        metrics['Profit Factor'] = 0.0
        metrics['Avg Return per Trade (%)'] = 0.0
        metrics['Winning Trades'] = 0
        metrics['Losing Trades'] = 0
        metrics['Sharpe Ratio'] = 0.0
        return metrics
        
    total_trades = len(trades_df)
    metrics['Total Trades'] = total_trades
    
    winning_trades_df = trades_df[trades_df['return_pct'] > 0]
    losing_trades_df = trades_df[trades_df['return_pct'] <= 0]
    
    winning_trades = len(winning_trades_df)
    losing_trades = len(losing_trades_df)
    metrics['Winning Trades'] = winning_trades
    metrics['Losing Trades'] = losing_trades
    
    win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
    metrics['Win Rate (%)'] = win_rate
    
    total_profit = winning_trades_df['pnl'].sum()
    total_loss = abs(losing_trades_df['pnl'].sum())
    
    profit_factor = total_profit / total_loss if total_loss > 0 else np.inf if total_profit > 0 else 1.0
    metrics['Profit Factor'] = profit_factor
    
    avg_return = trades_df['return_pct'].mean()
    metrics['Avg Return per Trade (%)'] = avg_return
    
    # 4. 15-Minute Sharpe Ratio (annualized to standard 365 days / 24 hrs / 4 15-minute periods)
    returns = equity_df['Equity'].pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        avg_hourly_ret = returns.mean()
        std_hourly_ret = returns.std()
        hourly_sharpe = avg_hourly_ret / std_hourly_ret if std_hourly_ret != 0 else 0
        # Annualized Sharpe = (Mean Return / Std Return) * sqrt(365 * 24 * 4)
        metrics['Sharpe Ratio'] = hourly_sharpe * np.sqrt(365 * 24 * 4)
    else:
        metrics['Sharpe Ratio'] = 0.0
        
    return metrics

def plot_backtest_results(df_with_signals: pd.DataFrame, trades_df: pd.DataFrame, save_path: str):
    """
    Plots the 15m price chart with trade markers and overlays the 2H PRZ zone as shaded areas.
    """
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
    
    # Colors
    color_bg = '#121214'
    color_card = '#1e1e24'
    color_text = '#f5f5f7'
    color_primary = '#00e676'  # Green
    color_accent = '#ff1744'   # Red
    color_blue = '#29b6f6'     # Blue
    
    # Set dark theme styling
    fig.patch.set_facecolor(color_bg)
    for ax in [ax1, ax2]:
        ax.set_facecolor(color_card)
        ax.tick_params(colors=color_text, labelsize=10)
        ax.xaxis.label.set_color(color_text)
        ax.yaxis.label.set_color(color_text)
        ax.title.set_color(color_text)
        for spine in ax.spines.values():
            spine.set_color('#33333f')
        ax.grid(True, color='#2c2c35', linestyle='--', alpha=0.5)

    timestamps = df_with_signals['Timestamp']
    
    # --- Subplot 1: 15m Price & HTF Filter Shading ---
    ax1.plot(timestamps, df_with_signals['Close'], label='BTC Close (15m)', color='#b0bec5', alpha=0.9, linewidth=1.2)
    
    # Shade the 2H HTF Reversal Zones (PRZs) in the background
    y_min, y_max = df_with_signals['Close'].min() * 0.99, df_with_signals['Close'].max() * 1.01
    
    if 'HTF_PRZ_Buy' in df_with_signals:
        ax1.fill_between(
            timestamps, y_min, y_max, 
            where=df_with_signals['HTF_PRZ_Buy'], 
            color=color_primary, alpha=0.08, 
            label='HTF Potential Buy Zone (PRZ)'
        )
    if 'HTF_PRZ_Sell' in df_with_signals:
        ax1.fill_between(
            timestamps, y_min, y_max, 
            where=df_with_signals['HTF_PRZ_Sell'], 
            color=color_accent, alpha=0.08, 
            label='HTF Potential Sell Zone (PRZ)'
        )
        
    ax1.set_ylim(y_min, y_max)
    
    # Plot trade signals
    if not trades_df.empty:
        longs = trades_df[trades_df['direction'] == 'LONG']
        shorts = trades_df[trades_df['direction'] == 'SHORT']
        
        # Long entries
        ax1.scatter(longs['entry_time'], longs['entry_price'], marker='^', color=color_primary, s=90, label='Long Entry', zorder=5)
        # Short entries
        ax1.scatter(shorts['entry_time'], shorts['entry_price'], marker='v', color=color_accent, s=90, label='Short Entry', zorder=5)
        
        # Exits
        sl_exits = trades_df[trades_df['exit_reason'] == 'STOP_LOSS']
        opp_exits = trades_df[trades_df['exit_reason'] == 'OPPOSITE_SIGNAL']
        
        ax1.scatter(sl_exits['exit_time'], sl_exits['exit_price'], marker='x', color='#ffffff', s=70, label='Stop Loss Exit', zorder=4)
        ax1.scatter(opp_exits['exit_time'], opp_exits['exit_price'], marker='o', color=color_blue, s=70, label='Opposite Flip Exit', zorder=4)
        
    ax1.set_title('BTC Multi-Timeframe (2H HTF Filter / 15m SFP LTF) Backtest', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='upper left', facecolor=color_bg, edgecolor='#33333f', labelcolor=color_text)
    
    # --- Subplot 2: Equity Curve & Drawdown ---
    equity = df_with_signals['Equity'].values
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / peaks * 100.0
    
    ax2.plot(timestamps, equity, label='Portfolio Equity', color=color_primary, linewidth=2)
    ax2.plot(timestamps, peaks, color='#b2dfdb', linestyle='--', alpha=0.4, label='Peak Equity')
    
    # Drawdown shading
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(timestamps, 0, drawdowns, color=color_accent, alpha=0.15, label='Drawdown (%)')
    ax2_twin.set_ylabel('Drawdown (%)', color=color_accent, fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor=color_accent)
    ax2_twin.invert_yaxis()
    
    ax2_twin.spines['top'].set_visible(False)
    ax2_twin.spines['left'].set_visible(False)
    ax2_twin.spines['bottom'].set_visible(False)
    ax2_twin.spines['right'].set_color(color_accent)
    
    ax2.set_title('Portfolio Equity Curve & Drawdown Profile', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel('Equity (USD)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor=color_bg, edgecolor='#33333f', labelcolor=color_text)
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=120)
    plt.close()
