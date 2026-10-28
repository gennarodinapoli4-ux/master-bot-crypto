import os
import sys
import subprocess

# --- AUTO-INSTALLAZIONE DIPENDENZE ---
required_packages = ['pandas', 'numpy', 'yfinance', 'requests']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURAZIONE STRATEGIA DEFINITIVA
# ==========================================
INITIAL_CAPITAL = 500.0
STATE_FILE = "portafoglio_stato.json"
EXCHANGE_FEE_RATE = 0.0005  # 0.05% fee per lato
SLIPPAGE_RATE = 0.0002      # 0.02% slippage stimato

# Paniere Multi-Settore Definitivo
ASSETS = [
    "BTC-USD", "ETH-USD", "SOL-USD",  # Crypto
    "NVDA", "AAPL", "MSFT",          # Big Tech USA
    "^GSPC",                         # Indice S&P 500
    "GC=F"                           # Oro
]

# Parametri Ottimizzati Definitivi
HOURS_PER_DAY = 24
SMA_DAYS = 100
MOM_DAYS = 30
DON_DAYS = 50

VOL_DAYS = 45        # Finestra volatilità ottimizzata a 45 giorni
VOL_SCALE = 0.8      
MAX_WEIGHT = 0.35    # Peso massimo per singolo asset al 35%
REB_THRESH = 0.08    # Soglia di ribilanciamento per tagliare l'over-trading

# ==========================================
# NOTIFICHE TELEGRAM
# ==========================================
def send_telegram(msg):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": msg}, timeout=10)
        except Exception:
            pass

# ==========================================
# GESTIONE STATO E MEMORIA PORTAFOGLIO
# ==========================================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                if isinstance(state, dict) and "weights" in state:
                    return state
        except Exception as e:
            print(f"Errore lettura stato: {e}")
            
    initial_state = {
        "capital": INITIAL_CAPITAL,
        "weights": {a: 0.0 for a in ASSETS},
        "history": []
    }
    save_state(initial_state)
    return initial_state

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Errore salvataggio stato: {e}")

# ==========================================
# MOTORE DI CALCOLO TREND FOLLOWING FINALE
# ==========================================
def run_trend_bot():
    now_dt = datetime.now()
    print(f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Avvio Bot Definitivo Multi-Settore")
    print("==================================================================")
    
    state = load_state()
    current_capital = float(state.get("capital", INITIAL_CAPITAL))
    old_weights = state.get("weights", {a: 0.0 for a in ASSETS})
    history = state.get("history", [])

    print(f"💼 Capitale Attuale in Gestione: {current_capital:.2f} $")

    print("⏳ Scaricamento dati storici orari...")
    try:
        raw = yf.download(ASSETS, period="60d", interval="1h", auto_adjust=True, group_by="ticker", progress=False)
        if raw.empty:
            print("❌ Errore: Dati Yahoo Finance vuoti.")
            return
        
        if isinstance(raw.columns, pd.MultiIndex):
            H = pd.DataFrame({a: raw[a]["High"] for a in ASSETS if a in raw.columns.levels[0]})
            L = pd.DataFrame({a: raw[a]["Low"] for a in ASSETS if a in raw.columns.levels[0]})
            C = pd.DataFrame({a: raw[a]["Close"] for a in ASSETS if a in raw.columns.levels[0]})
        else:
            H = pd.DataFrame({a: raw[f"High_{a}"] if f"High_{a}" in raw.columns else raw['High'] for a in ASSETS})
            L = pd.DataFrame({a: raw[f"Low_{a}"] if f"Low_{a}" in raw.columns else raw['Low'] for a in ASSETS})
            C = pd.DataFrame({a: raw[f"Close_{a}"] if f"Close_{a}" in raw.columns else raw['Close'] for a in ASSETS})

        for d in (H, L, C):
            d.index = d.index.tz_localize(None) if d.index.tz is None else d.index.tz_convert(None)
            d.ffill(inplace=True)
            d.bfill(inplace=True)
            
    except Exception as e:
        print(f"❌ Errore nel download dei dati: {e}")
        return

    n_sma = SMA_DAYS * HOURS_PER_DAY
    n_mom = MOM_DAYS * HOURS_PER_DAY
    n_don = DON_DAYS * HOURS_PER_DAY
    n_vol = VOL_DAYS * HOURS_PER_DAY

    s1 = np.sign(C - C.rolling(n_sma).mean())
    s2 = np.sign(C.pct_change(n_mom))
    hi = H.rolling(n_don).max().shift(1)
    lo = L.rolling(n_don).min().shift(1)
    brk = pd.DataFrame(np.where(C >= hi, 1.0, np.where(C <= lo, -1.0, np.nan)), index=C.index, columns=C.columns)
    s3 = brk.ffill().fillna(0.0).where(hi.notna())

    score = ((s1 + s2 + s3) / 3.0).clip(lower=0)
    vol = C.pct_change().rolling(n_vol).std() * np.sqrt(365 * HOURS_PER_DAY)
    
    avail = score.notna() & vol.notna() & (vol > 0)
    N_avail = avail.sum(axis=1).clip(lower=1)
    
    w_target_df = (score * (VOL_SCALE / vol)).div(N_avail, axis=0).where(avail, 0.0).clip(0, MAX_WEIGHT)
    
    latest_weights = w_target_df.iloc[-1].to_dict()
    current_returns = C.pct_change().iloc[-1].fillna(0.0).to_dict()

    port_return = sum(old_weights.get(a, 0.0) * current_returns.get(a, 0.0) for a in ASSETS)
    new_capital_before_costs = current_capital * (1.0 + port_return)

    total_trade_diff = sum(abs(latest_weights.get(a, 0.0) - old_weights.get(a, 0.0)) for a in ASSETS)
    
    costs_usd = 0.0
    final_weights = old_weights.copy()

    if total_trade_diff > REB_THRESH:
        cost_rate = EXCHANGE_FEE_RATE + SLIPPAGE_RATE
        costs_usd = total_trade_diff * new_capital_before_costs * cost_rate
        current_capital = new_capital_before_costs - costs_usd
        final_weights = latest_weights
        
        trade_msg = (
            f"🔄 RIBILANCIAMENTO DEFINITIVO MULTI-SETTORE\n"
            f"💰 Nuovo Capitale: {current_capital:.2f} $\n"
            f"📉 Costi Operativi: -{costs_usd:.2f} $\n"
            f"📊 Pesi per Asset:\n" + 
            "\n".join([f"   - {k}: {v*100:.1f}%" for k, v in final_weights.items() if v > 0])
        )
        print(trade_msg)
        send_telegram(trade_msg)
    else:
        current_capital = new_capital_before_costs
        print(f"⏸️ Variazione pesi sotto soglia ({total_trade_diff:.3f} <= {REB_THRESH}). Nessun ribilanciamento.")
        print(f"💰 Capitale aggiornato: {current_capital:.2f} $")

    state["capital"] = round(current_capital, 2)
    state["weights"] = {k: round(float(v), 4) for k, v in final_weights.items()}
    
    history.append({
        "date": now_dt.strftime("%Y-%m-%d %H:%M"),
        "capital": round(current_capital, 2),
        "port_return_pct": round(port_return * 100, 3)
    })
    state["history"] = history[-50:]

    save_state(state)
    print("==================================================================")
    print(f"✅ Esecuzione completata. Capitale Finale: {current_capital:.2f} $")
    print("==================================================================")

if __name__ == "__main__":
    run_trend_bot()
    
