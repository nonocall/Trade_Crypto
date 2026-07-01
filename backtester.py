import pandas as pd
import numpy as np

class Backtester:
    """
    Simulates trading based on signals from the Strategy.
    Tracks positions (Long, Short, Flat), calculates Stop Loss based on ATR,
    and updates the equity curve at each timestamp (mark-to-market).
    """
    def __init__(self, df: pd.DataFrame, initial_capital: float = 10000.0, fee_rate: float = 0.0006):
        self.df = df.copy().reset_index(drop=True)
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        
        # State variables
        self.capital = initial_capital
        self.position = 0.0  # 1.0 for Long, -1.0 for Short, 0.0 for Flat
        self.entry_price = 0.0
        self.entry_idx = -1
        self.entry_time = None
        self.stop_loss = 0.0
        
        # Result logs
        self.trades = []
        self.equity_curve = []
        
    def run(self, atr_mult: float = 1.5) -> dict:
        """
        Runs the backtest simulation.
        """
        n_bars = len(self.df)
        self.equity_curve = np.zeros(n_bars)
        
        for i in range(n_bars):
            row = self.df.iloc[i]
            close = row['Close']
            high = row['High']
            low = row['Low']
            open_p = row['Open']
            timestamp = row['Timestamp']
            signal = row['Signal']
            atr = row['ATR'] if 'ATR' in row else 0.0
            
            # If ATR is missing or NaN, use percentage-based stop loss (e.g., 2% of price)
            if pd.isna(atr) or atr <= 0:
                atr = close * 0.0133  # 1.5 * atr = ~2% of price
                
            # Current equity mark-to-market (MTM) calculation
            current_equity = self.capital
            if self.position == 1.0:
                # Long unrealized return
                unrealized_return = (close - self.entry_price) / self.entry_price
                current_equity = self.capital * (1.0 + unrealized_return)
            elif self.position == -1.0:
                # Short unrealized return
                unrealized_return = (self.entry_price - close) / self.entry_price
                current_equity = self.capital * (1.0 + unrealized_return)
                
            self.equity_curve[i] = current_equity
            
            # --- Check Exits if in Position ---
            if self.position == 1.0:
                # 1. Stop Loss Check
                if low <= self.stop_loss:
                    # Executed Stop Loss
                    exit_price = self.stop_loss
                    # If open was already below stop loss (gap down), exit at open
                    if open_p < self.stop_loss:
                        exit_price = open_p
                        
                    # Calculate trade return
                    trade_return = (exit_price - self.entry_price) / self.entry_price - 2 * self.fee_rate
                    self.capital = self.capital * (1.0 + trade_return)
                    self.equity_curve[i] = self.capital  # Update index value
                    
                    self.trades.append({
                        'entry_idx': self.entry_idx,
                        'entry_time': self.entry_time,
                        'direction': 'LONG',
                        'entry_price': self.entry_price,
                        'exit_idx': i,
                        'exit_time': timestamp,
                        'exit_price': exit_price,
                        'stop_loss': self.stop_loss,
                        'pnl': self.capital - (self.equity_curve[self.entry_idx] if self.entry_idx > 0 else self.initial_capital),
                        'return_pct': trade_return * 100.0,
                        'exit_reason': 'STOP_LOSS'
                    })
                    
                    self.position = 0.0
                    
                # 2. Opposite Signal Exit Check
                elif signal == -1:
                    # Close long position at the close price
                    exit_price = close
                    trade_return = (exit_price - self.entry_price) / self.entry_price - 2 * self.fee_rate
                    self.capital = self.capital * (1.0 + trade_return)
                    self.equity_curve[i] = self.capital
                    
                    self.trades.append({
                        'entry_idx': self.entry_idx,
                        'entry_time': self.entry_time,
                        'direction': 'LONG',
                        'entry_price': self.entry_price,
                        'exit_idx': i,
                        'exit_time': timestamp,
                        'exit_price': exit_price,
                        'stop_loss': self.stop_loss,
                        'pnl': self.capital - (self.equity_curve[self.entry_idx] if self.entry_idx > 0 else self.initial_capital),
                        'return_pct': trade_return * 100.0,
                        'exit_reason': 'OPPOSITE_SIGNAL'
                    })
                    
                    # Flip to SHORT
                    self.position = -1.0
                    self.entry_price = close
                    self.entry_idx = i
                    self.entry_time = timestamp
                    self.stop_loss = close + (atr_mult * atr)
                    
            elif self.position == -1.0:
                # 1. Stop Loss Check
                if high >= self.stop_loss:
                    # Executed Stop Loss
                    exit_price = self.stop_loss
                    # If open was already above stop loss (gap up), exit at open
                    if open_p > self.stop_loss:
                        exit_price = open_p
                        
                    # Calculate trade return
                    trade_return = (self.entry_price - exit_price) / self.entry_price - 2 * self.fee_rate
                    self.capital = self.capital * (1.0 + trade_return)
                    self.equity_curve[i] = self.capital
                    
                    self.trades.append({
                        'entry_idx': self.entry_idx,
                        'entry_time': self.entry_time,
                        'direction': 'SHORT',
                        'entry_price': self.entry_price,
                        'exit_idx': i,
                        'exit_time': timestamp,
                        'exit_price': exit_price,
                        'stop_loss': self.stop_loss,
                        'pnl': self.capital - (self.equity_curve[self.entry_idx] if self.entry_idx > 0 else self.initial_capital),
                        'return_pct': trade_return * 100.0,
                        'exit_reason': 'STOP_LOSS'
                    })
                    
                    self.position = 0.0
                    
                # 2. Opposite Signal Exit Check
                elif signal == 1:
                    # Close short position at the close price
                    exit_price = close
                    trade_return = (self.entry_price - exit_price) / self.entry_price - 2 * self.fee_rate
                    self.capital = self.capital * (1.0 + trade_return)
                    self.equity_curve[i] = self.capital
                    
                    self.trades.append({
                        'entry_idx': self.entry_idx,
                        'entry_time': self.entry_time,
                        'direction': 'SHORT',
                        'entry_price': self.entry_price,
                        'exit_idx': i,
                        'exit_time': timestamp,
                        'exit_price': exit_price,
                        'stop_loss': self.stop_loss,
                        'pnl': self.capital - (self.equity_curve[self.entry_idx] if self.entry_idx > 0 else self.initial_capital),
                        'return_pct': trade_return * 100.0,
                        'exit_reason': 'OPPOSITE_SIGNAL'
                    })
                    
                    # Flip to LONG
                    self.position = 1.0
                    self.entry_price = close
                    self.entry_idx = i
                    self.entry_time = timestamp
                    self.stop_loss = close - (atr_mult * atr)
                    
            # --- Check Entry if Flat ---
            else:
                if signal == 1:
                    # Enter LONG
                    self.position = 1.0
                    self.entry_price = close
                    self.entry_idx = i
                    self.entry_time = timestamp
                    self.stop_loss = close - (atr_mult * atr)
                elif signal == -1:
                    # Enter SHORT
                    self.position = -1.0
                    self.entry_price = close
                    self.entry_idx = i
                    self.entry_time = timestamp
                    self.stop_loss = close + (atr_mult * atr)
                    
        # Append equity curve array to dataframe
        self.df['Equity'] = self.equity_curve
        return {
            'trades': pd.DataFrame(self.trades) if len(self.trades) > 0 else pd.DataFrame(),
            'equity_curve': self.df[['Timestamp', 'Close', 'Equity']]
        }
