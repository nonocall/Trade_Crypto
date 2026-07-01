import os
import json
import time
import urllib.request
import pandas as pd
import numpy as np

def fetch_binance_klines(symbol: str = 'BTCUSDT', interval: str = '15m', limit: int = 1000, start_time_ms: int = None, proxy: str = 'http://127.0.0.1:7078') -> list:
    """
    Fetches raw klines from Binance API via the specified proxy.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    if start_time_ms is not None:
        url += f"&startTime={start_time_ms}"
        
    proxies = {'http': proxy, 'https': proxy}
    handler = urllib.request.ProxyHandler(proxies)
    opener = urllib.request.build_opener(handler)
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    
    req = urllib.request.Request(url)
    try:
        with opener.open(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching from Binance: {e}")
        return None

def get_binance_btc_data(symbol: str = 'BTCUSDT', interval: str = '15m', total_candles: int = 3000, proxy: str = 'http://127.0.0.1:7078', cache_file: str = 'btc_15m_historical.csv') -> pd.DataFrame:
    """
    Retrieves historical BTC data. Attempts to load from local cache first.
    If cache is missing, fetches from Binance API via proxy, caches it, and returns the DataFrame.
    """
    if os.path.exists(cache_file):
        print(f"Loading cached BTC data from {cache_file}...")
        df = pd.read_csv(cache_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
        
    print(f"Cache not found. Fetching {total_candles} candles of {interval} {symbol} data from Binance...")
    
    # Calculate start time (approximate total days back based on candle counts + buffer)
    # 15m candle = 900 seconds. 3000 candles = 2,700,000 seconds = 31.25 days.
    start_time_ms = int((time.time() - (total_candles * 900) - 2 * 3600) * 1000)
    
    all_klines = []
    current_start = start_time_ms
    
    while len(all_klines) < total_candles:
        remaining = total_candles - len(all_klines)
        limit = min(remaining, 1000)
        print(f"   Fetching batch: start_time={pd.to_datetime(current_start, unit='ms')}, limit={limit}...")
        
        klines = fetch_binance_klines(symbol, interval, limit, current_start, proxy)
        if not klines:
            print("Failed to fetch klines from Binance.")
            break
            
        all_klines.extend(klines)
        if len(klines) < limit:
            # End of data
            break
            
        # Set next start time to 1ms after the last candle's open time
        last_open_time = klines[-1][0]
        current_start = last_open_time + 1
        
        # Avoid rate-limit blocking
        time.sleep(0.5)
        
    if not all_klines:
        raise ValueError("Could not retrieve any data from Binance and no cache file exists.")
        
    # Format into DataFrame
    # 0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume
    df = pd.DataFrame(all_klines, columns=[
        'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteVolume', 'TradesCount', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore'
    ])
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
        
    df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    # Save cache
    df.to_csv(cache_file, index=False)
    print(f"Cached {len(df)} rows to {cache_file}.")
    return df

def resample_to_htf(df_15m: pd.DataFrame, htf_interval: str = '2h') -> pd.DataFrame:
    """
    Resamples the LTF (15m) DataFrame to a HTF (e.g. 2H) DataFrame.
    """
    df = df_15m.copy()
    df = df.set_index('Timestamp')
    
    # Resample rules: Open = first, High = max, Low = min, Close = last, Volume = sum
    df_htf = df.resample(htf_interval).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    df_htf = df_htf.reset_index()
    return df_htf

def generate_dummy_data_15m(length: int = 2000, start_price: float = 60000.0, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic 15m price data for offline testing.
    """
    np.random.seed(seed)
    timestamps = pd.date_range(start='2026-01-01', periods=length, freq='15min')
    
    # Simulates base walk with trend and cycles
    t = np.arange(length)
    cycle = 2000 * np.sin(t * 2 * np.pi / 500) + 800 * np.cos(t * 2 * np.pi / 100)
    trend = 3 * t
    step_noise = np.random.normal(0, 150, length)
    noise = np.cumsum(step_noise)
    
    base_price = start_price + cycle + trend + noise
    
    opens, highs, lows, closes, volumes = [], [], [], [], []
    prev_close = start_price
    
    for i in range(length):
        price = base_price[i]
        volatility = np.random.uniform(50, 200)
        
        op = prev_close + np.random.normal(0, 30)
        cl = price
        hi = max(op, cl) + np.random.uniform(10, volatility)
        lo = min(op, cl) - np.random.uniform(10, volatility)
        
        # Inject occasional SFP-like sweeps at 15m
        if i > 20 and i % 80 == 0:
            local_min_low = min(lows[-15:])
            lo = local_min_low - np.random.uniform(20, 80)
            cl = local_min_low + np.random.uniform(10, 50)
            op = cl - np.random.uniform(0, 30)
            hi = max(op, cl) + np.random.uniform(10, 50)
        elif i > 20 and i % 80 == 40:
            local_max_high = max(highs[-15:])
            hi = local_max_high + np.random.uniform(20, 80)
            cl = local_max_high - np.random.uniform(10, 50)
            op = cl + np.random.uniform(0, 30)
            lo = min(op, cl) - np.random.uniform(10, 50)
            
        opens.append(op)
        highs.append(hi)
        lows.append(lo)
        closes.append(cl)
        volumes.append(np.random.uniform(5, 50))
        prev_close = cl
        
    return pd.DataFrame({
        'Timestamp': timestamps,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })

if __name__ == '__main__':
    # Offline test
    dummy = generate_dummy_data_15m(100)
    print("Dummy Data Generated successfully!")
    print(dummy.head())
    
    htf = resample_to_htf(dummy, '2h')
    print("Resampled to 2H successfully!")
    print(htf.head())
