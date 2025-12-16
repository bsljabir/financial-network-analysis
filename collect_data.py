"""
Data Collection Script - SUPER SIMPLE VERSION
This version avoids the pandas bug by saving data differently
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import os

print("=" * 70)
print("COLLECTING FINANCIAL DATA - SIMPLE VERSION")
print("=" * 70)
print(f"Files will be saved in: {os.getcwd()}")
print("=" * 70)

# Date range
start_date = "2021-01-01"
end_date = "2024-12-31"

# All assets
all_tickers = [
    # Stocks
    'AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOG', 'META', 'NVDA', 'BRK-B', 
    'JPM', 'JNJ', 'XOM', 'PG', 'V', 'KO', 'WMT',
    # ETFs
    'SPY', 'QQQ', 'IAU', 'VTI', 'EEM', 'AGG', 'SLV', 'XLE', 'REET', 'VIG',
    # Commodities
    'GC=F', 'SI=F', 'CL=F', 'BZ=F', 'NG=F', 'HG=F', 'PL=F', 'ZC=F', 'ZW=F', 'ZS=F',
    # Crypto
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD',
    'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LTC-USD',
    # Fixed Income
    'TLT', 'IEF', 'BND', 'BIL', 'SHV', 'HYG', 'VCSH', 'TIP', 'MBB', 'ICVT'
]

print(f"\nDownloading {len(all_tickers)} assets...")
print("This will take about 5-10 minutes...\n")

# Download ALL data at once (this is the trick!)
successful = []
failed = []

for i, ticker in enumerate(all_tickers, 1):
    try:
        print(f"[{i}/{len(all_tickers)}] {ticker}...", end=" ")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if not data.empty:
            print(f"✅ ({len(data)} days)")
            successful.append(ticker)
        else:
            print("❌ No data")
            failed.append(ticker)
    except Exception as e:
        print(f"❌ Error: {e}")
        failed.append(ticker)
    
    time.sleep(0.3)  # Small delay

print("\n" + "=" * 70)
print(f"SUCCESS: {len(successful)}/{len(all_tickers)} assets downloaded")
if failed:
    print(f"FAILED: {', '.join(failed)}")
print("=" * 70)

# Now download all successful tickers together
print("\n🔄 Combining all data into one file...")
print("Please wait...")

try:
    # Download all at once - yfinance handles the DataFrame creation
    all_data = yf.download(successful, start=start_date, end=end_date, group_by='ticker', progress=False)
    
    print("✅ Data combined successfully!")
    
    # Extract closing prices
    if len(successful) > 1:
        # Multiple tickers - data is grouped
        prices = pd.DataFrame()
        for ticker in successful:
            try:
                prices[ticker] = all_data[ticker]['Close']
            except:
                try:
                    prices[ticker] = all_data[ticker]['Adj Close']
                except:
                    print(f"⚠️  Warning: Could not get close price for {ticker}")
    else:
        # Single ticker
        prices = pd.DataFrame({successful[0]: all_data['Close']})
    
    # Clean data
    prices = prices.ffill().dropna()
    
    print(f"\n📊 Final dataset: {prices.shape[0]} days × {prices.shape[1]} assets")
    
    # Save prices
    prices.to_csv('multi_asset_prices_2021_2024.csv')
    print(f"\n✅ SAVED: multi_asset_prices_2021_2024.csv")
    print(f"   Location: {os.path.abspath('multi_asset_prices_2021_2024.csv')}")
    
    # Calculate and save returns
    returns = prices.pct_change().dropna()
    returns.to_csv('multi_asset_returns_2021_2024.csv')
    print(f"✅ SAVED: multi_asset_returns_2021_2024.csv")
    
    # Calculate and save log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.to_csv('multi_asset_log_returns_2021_2024.csv')
    print(f"✅ SAVED: multi_asset_log_returns_2021_2024.csv")
    
    # Summary statistics
    summary = pd.DataFrame({
        'Annual_Return': returns.mean() * 252,
        'Annual_Volatility': returns.std() * np.sqrt(252),
        'Min_Price': prices.min(),
        'Max_Price': prices.max()
    })
    summary.to_csv('summary_statistics.csv')
    print(f"✅ SAVED: summary_statistics.csv")
    
    print("\n" + "=" * 70)
    print("🎉 SUCCESS! ALL FILES CREATED!")
    print("=" * 70)
    print("\n📁 Check your folder for these 4 CSV files:")
    print("   1. multi_asset_prices_2021_2024.csv")
    print("   2. multi_asset_returns_2021_2024.csv")
    print("   3. multi_asset_log_returns_2021_2024.csv")
    print("   4. summary_statistics.csv")
    print(f"\n📍 Location: {os.getcwd()}")
    print("\n✅ Next step: Run network_analysis.py")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTrying backup method...")
    
    # Backup method: Download one by one and save immediately
    price_data = {}
    for ticker in successful:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            price_data[ticker] = data['Close']
        except:
            pass
    
    # Manually create DataFrame
    prices = pd.concat(price_data, axis=1)
    prices.columns = list(price_data.keys())
    prices = prices.ffill().dropna()
    
    # Save
    prices.to_csv('multi_asset_prices_2021_2024.csv')
    returns = prices.pct_change().dropna()
    returns.to_csv('multi_asset_returns_2021_2024.csv')
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.to_csv('multi_asset_log_returns_2021_2024.csv')
    
    print("\n✅ Backup method succeeded! Files created!")
    print(f"📍 Location: {os.getcwd()}")
