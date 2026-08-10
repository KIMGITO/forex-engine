import os
import io
import zipfile
import requests
import pandas as pd
from bs4 import BeautifulSoup

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
YEARS = list(range(2018, 2027))  # 2018 to 2026

BASE_URL = "https://www.histdata.com/download-free-forex-historical-data"

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.histdata.com/"
}

def get_download_url_and_params(symbol, year):
    """Scrapes the correct ASCII 1-Minute page to retrieve the download form token."""
    # Correct HistData URL path for ASCII M1 data
    url = f"{BASE_URL}/?/ascii/1-minute-bar-quotes/{symbol.lower()}/{year}"
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return None, None
            
        soup = BeautifulSoup(res.text, 'html.parser')
        form = soup.find('form', id='file_down')
        if not form:
            return None, None
            
        action = form.get('action')
        post_url = f"https://www.histdata.com{action}" if action.startswith('/') else action
        
        inputs = form.find_all('input')
        payload = {inp.get('name'): inp.get('value', '') for inp in inputs if inp.get('name')}
        
        return post_url, payload
    except Exception as e:
        print(f"    [Error] Page scrape failed for {symbol} {year}: {e}")
        return None, None

def process_zip_bytes(zip_bytes, output_csv):
    """Extracts raw ASCII M1 CSV, resamples to M15, and writes to disk."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                return False
            
            target_file = csv_files[0]
            with z.open(target_file) as f:
                # ASCII format uses semicolon separation: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
                df = pd.read_csv(
                    f, 
                    sep=';',
                    header=None,
                    names=['DateTimeStr', 'Open', 'High', 'Low', 'Close', 'Volume']
                )
                
                df['DateTime'] = pd.to_datetime(df['DateTimeStr'], format='%Y%m%d %H%M%S')
                df.set_index('DateTime', inplace=True)
                df.drop(columns=['DateTimeStr'], inplace=True)
                df.sort_index(inplace=True)
                
                # Resample 1-minute data into 15-minute OHLCV candles
                df_m15 = df.resample('15Min').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                df_m15.to_csv(output_csv)
                return len(df_m15)
    except Exception as e:
        print(f"    [Error] Resampling failed: {e}")
        return False

for symbol in SYMBOLS:
    target_dir = f"data/raw/histdata/{symbol}"
    os.makedirs(target_dir, exist_ok=True)
    print(f"\n================ Processing Pair: {symbol} ================")
    
    for year in YEARS:
        output_csv = f"{target_dir}/{symbol}_M15_{year}.csv"
        
        if os.path.exists(output_csv) and os.path.getsize(output_csv) > 100:
            print(f"--> {symbol} {year}: File already exists. Skipping.")
            continue

        print(f"--> {symbol} {year}: Scraping download link...", end="", flush=True)
        post_url, payload = get_download_url_and_params(symbol, year)
        
        if not post_url or not payload:
            print(" [Failed: Data not published on HistData yet]")
            continue
            
        print(" Downloading ZIP...", end="", flush=True)
        try:
            r = requests.post(post_url, data=payload, headers=headers, timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                print(" Resampling to M15...", end="", flush=True)
                bar_count = process_zip_bytes(r.content, output_csv)
                if bar_count:
                    print(f" Done! Saved {bar_count} M15 candles -> {output_csv}")
                else:
                    print(" Failed to resample CSV.")
            else:
                print(f" Download HTTP Status {r.status_code}.")
        except Exception as e:
            print(f" Timeout/Request Error: {e}")