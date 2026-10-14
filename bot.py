import os
import sys
import subprocess

# --- AUTO-INSTALLAZIONE DI SICUREZZA PER GITHUB ACTIONS ---
required_packages = ['pandas', 'yfinance', 'requests', 'nltk']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Inizializzazione VADER
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ==========================================
# CONFIGURAZIONE TELEGRAM & PORTAFOGLIO
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

TOTAL_CAPITAL = 500.0  # Capitale totale a 500 $
NUM_SLOTS = 3
SLOT_CAPITAL = TOTAL_CAPITAL / NUM_SLOTS  # ~166 $ per slot
STATE_FILE = "portafoglio_stato.json"

# ==========================================
# WHITELIST ISTITUZIONALE & TOP ESPERTI
# ==========================================
INSTITUTIONAL_WHITELIST = [
    "CoinDesk", "Cointelegraph", "Bloomberg Crypto", "Glassnode", 
    "Vitalik Buterin", "CoinMetrics", "WatcherGuru", "Wu Blockchain"
]

def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Errore Telegram: {e}")

# ==========================================
# GESTIONE STATO SLOT (PERSISTENZA)
# ==========================================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Stato iniziale: Slot 1 mantiene intatto ETH-USD come concordato
    return {
        "slot_1": {"symbol": "ETH-USD", "type": "SHORT", "entry_price": 0.0, "max_price": 0.0, "active": True},
        "slot_2": {"symbol": None, "type": None, "entry_price": 0.0, "max_price": 0.0, "active": False},
        "slot_3": {"symbol": None, "type": None, "entry_price": 0.0, "max_price": 0.0, "active": False}
    }

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Errore salvataggio stato: {e}")

# ==========================================
# PANIERE TOP CRIPTO REALI E LIQUIDE
# ==========================================
CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", 
    "SUI-USD", "NEAR-USD", "RENDER-USD", "FET-USD", "PEPE-USD", "SHIB-USD", "WIF-USD",
    "DOGE-USD", "DOT-USD", "UNI-USD", "ICP-USD", "APT-USD", "ARB-USD", "OP-USD",
    "INJ-USD", "TIA-USD", "SEI-USD", "TAO-USD", "AAVE-USD", "ONDO-USD", "POL-USD",
    "ALGO-USD", "FIL-USD", "ATOM-USD", "KAS-USD", "JUP-USD", "STX-USD", "CRV-USD"
]

# ==========================================
# 1. ANALISI FILTRATA SULLA WHITELIST
# ==========================================
def analyze_institutional_sentiment(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return 0.0
        
        sentiments = []
        for item in news:
            publisher = item.get('publisher', '') or item.get('providerPublishTime', '')
            title = item.get('content', {}).get('title', '') or item.get('title', '')
            
            if title:
                score = sia.polarity_scores(title)['compound']
                if any(source.lower() in str(publisher).lower() or source.lower() in title.lower() for source in INSTITUTIONAL_WHITELIST):
                    score *= 1.5
                sentiments.append(score)
        
        if sentiments:
            return sum(sentiments) / len(sentiments)
        return 0.0
    except Exception as e:
        print(f"Errore analisi whitelist per {symbol}: {e}")
        return 0.0

# ==========================================
# 2. MOTORE TECNICO + WHITELIST SENTIMENT
# ==========================================
def analyze_market(symbol):
    try:
        df = yf.download(symbol, period="5d", interval="1h", progress=False)
        if df.empty or len(df) < 50:
            return None, 0, 0
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close']
        volume = df['Volume']

        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        current_price = close.iloc[-1]
        vol_mean = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]

        # RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        inst_score = analyze_institutional_sentiment(symbol)

        score = 0.0
        signal = "NEUTRAL"

        if current_price > sma20 and sma20 > sma50:
            if current_rsi < 70 and current_vol > (vol_mean * 0.8) and inst_score > -0.2:
                score = 2.5 + (inst_score * 0.5)
                signal = "LONG"
                
        elif current_price < sma20 and sma20 < sma50:
            if current_rsi > 30 and current_vol > (vol_mean * 0.8) and inst_score < 0.2:
                score = 2.5 + (abs(inst_score) * 0.5)
                signal = "SHORT"

        return signal, score, current_price
    except Exception as e:
        print(f"Errore analisi tecnica {symbol}: {e}")
        return None, 0, 0

# ==========================================
# 3. ESECUZIONE PRINCIPALE (GESTIONE 3 SLOT)
# ==========================================
def run_trading_bot():
    print(f"[{datetime.now()}] Avvio scansione v17.0 (Whitelist Istituzionale + 500$ + 3 Slot)...")
    
    state = load_state()
    print(f"Stato Slot attuale -> Slot 1 (ETH fisso): {state['slot_1']['active']} | Slot 2: {state['slot_2']['active']} | Slot 3: {state['slot_3']['active']}")

    free_slots = [key for key, slot in state.items() if not slot['active']]

    if free_slots:
        best_opportunity = None
        max_score = 0

        for crypto in CRYPTO_LIST:
            if any(slot['symbol'] == crypto for slot in state.values() if slot['active']):
                continue
                
            signal, score, price = analyze_market(crypto)
            if score > max_score:
                max_score = score
                best_opportunity = (crypto, signal, price)

        if best_opportunity and max_score >= 1.8:
            crypto, signal, price = best_opportunity
            target_slot = free_slots[0]
            
            state[target_slot] = {
                "symbol": crypto,
                "type": signal,
                "entry_price": float(price),
                "max_price": float(price),
                "active": True
            }
            save_state(state)

            msg = f"🚀 Nuova Apertura (Whitelist & Slot: {target_slot.upper()}) | Asset: {crypto} | Direzione: {signal} | Prezzo: {price:.2f} | Capitale Slot: {SLOT_CAPITAL:.1f}$"
            print(msg)
            send_telegram_alert(msg)
        else:
            print("Nessun asset ha superato i filtri istituzionali in questo ciclo.")
    else:
        print("Tutti i 3 slot sono occupati. Monitoraggio posizioni aperte in corso...")

if __name__ == "__main__":
    run_trading_bot()
