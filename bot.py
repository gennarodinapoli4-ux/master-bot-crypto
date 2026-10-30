import os
import sys
import json
import subprocess

# Auto-installazione pacchetti se non presenti nel workflow YAML
try:
    import requests
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    print("📦 Installazione librerie necessarie in corso...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests", "yfinance", "pandas", "numpy"])
    import requests
    import yfinance as yf
    import pandas as pd
    import numpy as np

from datetime import datetime

# ==============================================================================
# ⚙️ CONFIGURAZIONE BOT MULTI-SETTORE (FIX COMPLETO DATI E STATO)
# ==============================================================================
STATE_FILE = "portafoglio_stato.json"
INITIAL_CAPITAL = 500.0

ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD", "NVDA", "AAPL", "MSFT", "^GSPC", "GC=F"]

# Scarichiamo 730 giorni (2 anni) per avere lo storico completo
DATA_PERIOD = "730d"
HOURS_PER_DAY = 24

SMA_DAYS = 100
MOM_DAYS = 30
DON_DAYS = 50
VOL_DAYS = 45

VOL_SCALE = 0.80
MAX_WEIGHT = 0.35
REB_THRESH = 0.08
COST_RATE = (0.05 + 0.02) / 100.0  # 0.07% tra fee e slippage

# Legge sia TELEGRAM_BOT_TOKEN (dal tuo workflow) sia TELEGRAM_TOKEN
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or ""
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or ""


def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ℹ️ Telegram non configurato (Secrets mancanti o vuoti su GitHub).")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=15)
        if r.status_code != 200:
            print(f"⚠️ Errore invio Telegram: {r.text}")
    except Exception as e:
        print(f"⚠️ Eccezione Telegram: {e}")


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                if "last_prices" not in state:
                    state["last_prices"] = {}
                return state
        except Exception as e:
            print(f"⚠️ Errore lettura stato, riparto da default: {e}")
    return {
        "capital": INITIAL_CAPITAL,
        "weights": {a: 0.0 for a in ASSETS},
        "last_prices": {},
        "history": []
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


def download_market_data():
    print(f"⏳ Scaricamento dati storici orari ({DATA_PERIOD}) con threads=False...")
    raw = yf.download(
        ASSETS,
        period=DATA_PERIOD,
        interval="1h",
        auto_adjust=True,
        group_by="ticker",
        threads=False,
        progress=False
    )
    if raw is None or raw.empty:
        raise RuntimeError("Download dati fallito o vuoto.")

    H = pd.DataFrame({a: raw[a]["High"] for a in ASSETS if a in raw.columns.levels[0]})
    L = pd.DataFrame({a: raw[a]["Low"] for a in ASSETS if a in raw.columns.levels[0]})
    C = pd.DataFrame({a: raw[a]["Close"] for a in ASSETS if a in raw.columns.levels[0]})

    for d in (H, L, C):
        d.index = d.index.tz_localize(None) if d.index.tz is None else d.index.tz_convert(None)
        d.ffill(inplace=True)

    if len(C) > 2:
        H = H.iloc[:-1]
        L = L.iloc[:-1]
        C = C.iloc[:-1]

    return H, L, C


def main():
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] 🚀 Avvio Bot Multi-Settore (Fix Completo)")
    print("=" * 70)

    state = load_state()
    capital = float(state["capital"])
    current_weights = pd.Series(state["weights"]).reindex(ASSETS).fillna(0.0)
    last_prices = state.get("last_prices", {})

    print(f"💼 Capitale Attuale in Gestione: {capital:.2f} $")

    try:
        H, L, C = download_market_data()
    except Exception as e:
        print(f"❌ Errore critico durante il download dati: {e}")
        sys.exit(1)

    latest_prices = C.iloc[-1]

    # 1. Aggiornamento preciso del Capitale in base ai prezzi reali salvati
    port_ret = 0.0
    if last_prices and current_weights.sum() > 0:
        prev_p = pd.Series(last_prices).reindex(ASSETS)
        valid = prev_p.notna() & (prev_p > 0) & latest_prices.notna()
        asset_rets = pd.Series(0.0, index=ASSETS)
        asset_rets[valid] = (latest_prices[valid] / prev_p[valid]) - 1.0

        port_ret = float((current_weights * asset_rets).sum())
        capital *= (1.0 + port_ret)

        if (1.0 + port_ret) > 0:
            current_weights = current_weights * (1.0 + asset_rets) / (1.0 + port_ret)

    # 2. Calcolo Indicatori (su 730 giorni)
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

    score = ((s1 + s2 + s3) / 3.0).clip(lower=0.0)
    vol = C.pct_change().rolling(n_vol).std() * np.sqrt(365 * HOURS_PER_DAY)

    avail = score.notna() & vol.notna() & (vol > 0)
    N_avail = avail.sum(axis=1).clip(lower=1)

    w_target_df = (score * (VOL_SCALE / vol)).div(N_avail, axis=0).where(avail, 0.0).clip(0.0, MAX_WEIGHT)
    target_weights = w_target_df.iloc[-1].reindex(ASSETS).fillna(0.0)

    if target_weights.sum() > 1.0:
        target_weights = target_weights / target_weights.sum()

    print("📊 Pesi Target calcolati dal modello in questo momento:")
    for a in ASSETS:
        print(f"   - {a:8s}: {target_weights[a]*100:5.1f}% (Attuale: {current_weights[a]*100:5.1f}%)")

    # 3. Verifica soglia di ribilanciamento
    diff = float(np.abs(target_weights - current_weights).sum())
    rebalanced = False

    if diff > REB_THRESH or (target_weights.sum() == 0 and current_weights.sum() > 0):
        fee_cost = diff * capital * COST_RATE
        capital -= fee_cost
        current_weights = target_weights.copy()
        rebalanced = True
        print(f"🔄 Ribilanciamento eseguito! Variazione: {diff:.3f} | Costo stimato: {fee_cost:.2f} $")

        alloc_lines = [
            f"• <b>{a}</b>: {w*100:.1f}% ({w*capital:.1f} $)"
            for a, w in current_weights.items() if w > 0.005
        ]
        alloc_str = "\n".join(alloc_lines) if alloc_lines else "• 100% Liquidità (Cash)"

        msg = (
            f"🔄 <b>RIBILANCIAMENTO PORTAFOGLIO</b>\n"
            f"📅 {now_str} UTC\n"
            f"💰 <b>Capitale:</b> {capital:.2f} $\n"
            f"📉 <b>Costi operazione:</b> -{fee_cost:.2f} $\n\n"
            f"📊 <b>Nuova Allocazione:</b>\n{alloc_str}"
        )
        send_telegram(msg)
    else:
        print(f"⏸️ Variazione pesi sotto soglia ({diff:.3f} <= {REB_THRESH}). Nessun ribilanciamento.")

    # 4. Salvataggio Stato e Prezzi Correnti
    state["capital"] = round(capital, 4)
    state["weights"] = {a: round(float(current_weights[a]), 4) for a in ASSETS}
    state["last_prices"] = {a: round(float(latest_prices[a]), 4) for a in ASSETS if pd.notna(latest_prices[a])}

    state["history"].append({
        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "capital": round(capital, 2),
        "port_return_pct": round(port_ret * 100, 4),
        "rebalanced": rebalanced
    })
    state["history"] = state["history"][-500:]

    save_state(state)
    print(f"💰 Capitale aggiornato: {capital:.2f} $")
    print("=" * 70)
    print(f"✅ Esecuzione completata con successo.")


if __name__ == "__main__":
    main()
