import os
import sys
import subprocess

# --- AUTO-INSTALLAZIONE DI SICUREZZA ---
required_packages = ['pandas', 'yfinance', 'requests', 'nltk', 'numpy']
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
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings
warnings.filterwarnings('ignore')

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ==========================================
# CONFIGURAZIONE PORTAFOGLIO & PESI DIFFERENZIALI
# ==========================================
INITIAL_CAPITAL = 500.0
NUM_SLOTS = 3
DEFAULT_SLOT_CAPITAL = round(INITIAL_CAPITAL / NUM_SLOTS, 2)
STATE_FILE = "portafoglio_stato.json"
EXCHANGE_FEE_RATE = 0.001  # 0.1% fee per lato (0.2% round-trip)

# Allocazione Differenziale sui 3 Slot (45% / 35% / 20%)
SLOT_WEIGHTS = {
    "slot_1": 0.45,
    "slot_2": 0.35,
    "slot_3": 0.20
}

# ==========================================
# PARAMETRI HIGH WIN-RATE (CONNORS RSI2 + SCUDO)
# ==========================================
BIG_CAP_LIST = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", 
    "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
    "LTC-USD", "BCH-USD", "TON-USD"
]

BIG_CAP_TP = 3.5          # Take Profit rapido (+3.5% netto)
BIG_CAP_SL = -5.5         # Stop Loss ampio (-5.5%) anti-rumore 5m
BIG_CAP_TRAIL_ACT = 1.6   # Attiva scudo Trailing a +1.6% netto
BIG_CAP_TRAIL_DIST = 0.5  # Chiude se scende dello 0.5% dal picco

ALT_TP = 5.0              # Take Profit rapido (+5.0% netto)
ALT_SL = -7.0             # Stop Loss protezione (-7.0% netto)
ALT_TRAIL_ACT = 2.2       # Attiva scudo Trailing a +2.2% netto
ALT_TRAIL_DIST = 0.7      # Chiude se scende dello 0.7% dal picco

INSTITUTIONAL_WHITELIST = [
    "CoinDesk", "Cointelegraph", "Bloomberg Crypto", "Glassnode", 
    "Vitalik Buterin", "CoinMetrics", "WatcherGuru", "Wu Blockchain"
]

CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", 
    "SUI-USD", "NEAR-USD", "RENDER-USD", "FET-USD", "DOT-USD", "UNI-USD", 
    "ICP-USD", "ARB-USD", "OP-USD", "INJ-USD", "TIA-USD", "SEI-USD", 
    "AAVE-USD", "ONDO-USD", "ALGO-USD", "FIL-USD", "ATOM-USD", 
    "JUP-USD", "CRV-USD", "LTC-USD", "BCH-USD", "HBAR-USD", "TON-USD"
]

# ==========================================
# NOTIFICHE TELEGRAM (SE CONFIGURATE SU GITHUB)
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
# GESTIONE STATO E LIBRO CONTABILE
# ==========================================
def load_state():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                if isinstance(state, dict) and "slot_1" in state:
                    if "stats" not in state:
                        state["stats"] = {"total_realized_usd": 0.0, "wins": 0, "losses": 0, "history": []}
                    for k in ["slot_1", "slot_2", "slot_3"]:
                        if "invested_capital" not in state[k] or state[k]["invested_capital"] == 0.0:
                            state[k]["invested_capital"] = DEFAULT_SLOT_CAPITAL if state[k].get("active") else 0.0
                        if "opened_at" not in state[k] or not state[k]["opened_at"]:
                            state[k]["opened_at"] = now_str if state[k].get("active") else None
                    return state
        except Exception as e:
            print(f"Errore lettura stato: {e}")
            
    initial_state = {
        "slot_1": {"symbol": None, "type": None, "entry_price": 0.0, "max_price": 0.0, "invested_capital": 0.0, "opened_at": None, "active": False},
        "slot_2": {"symbol": None, "type": None, "entry_price": 0.0, "max_price": 0.0, "invested_capital": 0.0, "opened_at": None, "active": False},
        "slot_3": {"symbol": None, "type": None, "entry_price": 0.0, "max_price": 0.0, "invested_capital": 0.0, "opened_at": None, "active": False},
        "stats": {"total_realized_usd": 0.0, "wins": 0, "losses": 0, "history": []}
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
# ANALISI SENTIMENT & CONNORS RSI(2)
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
                if any(s.lower() in str(publisher).lower() or s.lower() in title.lower() for s in INSTITUTIONAL_WHITELIST):
                    score *= 1.5
                sentiments.append(score)
        return sum(sentiments) / len(sentiments) if sentiments else 0.0
    except Exception:
        return 0.0

def analyze_market(symbol):
    try:
        df = yf.download(symbol, period="1mo", interval="1h", progress=False)
        if df.empty or len(df) < 210:
            return "NEUTRAL", 0.0, 0.0, 50.0, 0.0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close'].dropna()
        if len(close) < 205:
            return "NEUTRAL", 0.0, 0.0, 50.0, 0.0

        current_price = float(close.iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        sma5 = float(close.rolling(5).mean().iloc[-1])

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/2, min_periods=2, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/2, min_periods=2, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi2_series = (100 - (100 / (1 + rs))).fillna(50.0)
        current_rsi2 = float(rsi2_series.iloc[-1])

        inst_score = analyze_institutional_sentiment(symbol)
        score = 0.0
        signal = "NEUTRAL"

        if current_price > sma200 and current_rsi2 < 10.0 and inst_score >= -0.4:
            score = (10.0 - current_rsi2) + (inst_score * 1.5)
            signal = "LONG"
        elif current_price < sma200 and current_rsi2 > 90.0 and inst_score <= 0.4:
            score = (current_rsi2 - 90.0) + (abs(inst_score) * 1.5)
            signal = "SHORT"

        return signal, round(score, 2), current_price, round(current_rsi2, 1), sma5
    except Exception:
        return "NEUTRAL", 0.0, 0.0, 50.0, 0.0

# ==========================================
# ESECUZIONE PRINCIPALE
# ==========================================
def run_trading_bot():
    now_dt = datetime.now()
    print(f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Avvio Master Bot v21.0 (Connors RSI2 + Pesi Differenziali)")
    print("==================================================================")
    
    state = load_state()
    stats = state.get("stats", {"total_realized_usd": 0.0, "wins": 0, "losses": 0, "history": []})
    
    compounded_cap = max(50.0, INITIAL_CAPITAL + stats['total_realized_usd'])
    tot_trades = stats['wins'] + stats['losses']
    win_rate = (stats['wins'] / tot_trades * 100) if tot_trades > 0 else 0.0

    print(f"💼 CAPITALE COMPOSTO TOTALE: {compounded_cap:.2f} $ (Base: {INITIAL_CAPITAL:.0f}$ | Cassa: {stats['total_realized_usd']:+.2f}$)")
    print(f"🏆 Bilancio Trade Chiusi: {stats['wins']} Vinte | {stats['losses']} Perse (Win Rate: {win_rate:.1f}%)")
    print(f"⚖️ Pesi Slot: S1(45%): {compounded_cap*0.45:.1f}$ | S2(35%): {compounded_cap*0.35:.1f}$ | S3(20%): {compounded_cap*0.20:.1f}$")
    print("==================================================================")

    total_floating_usd = 0.0
    slot_keys = ["slot_1", "slot_2", "slot_3"]
    fee_pct = EXCHANGE_FEE_RATE * 2 * 100  # 0.2%
    
    for slot_key in slot_keys:
        slot = state[slot_key]
        if slot.get('active') and slot.get('symbol'):
            sym = slot['symbol']
            pos_type = slot['type']
            entry = float(slot['entry_price'])
            slot_cap = float(slot.get('invested_capital', DEFAULT_SLOT_CAPITAL))
            best_price = float(slot.get('max_price', entry)) or entry

            opened_str = slot.get('opened_at')
            hours_open = 0.0
            opened_dt = None
            if opened_str:
                try:
                    opened_dt = datetime.strptime(opened_str, "%Y-%m-%d %H:%M:%S")
                    hours_open = round((now_dt - opened_dt).total_seconds() / 3600.0, 1)
                except Exception:
                    hours_open = 0.0
            else:
                slot['opened_at'] = now_dt.strftime("%Y-%m-%d %H:%M:%S")

            is_big = sym in BIG_CAP_LIST
            cat_label = "BIG CAP" if is_big else "ALTCOIN"
            tp_target = BIG_CAP_TP if is_big else ALT_TP
            sl_target = BIG_CAP_SL if is_big else ALT_SL
            trail_act = BIG_CAP_TRAIL_ACT if is_big else ALT_TRAIL_ACT
            trail_dist = BIG_CAP_TRAIL_DIST if is_big else ALT_TRAIL_DIST

            if pos_type == "LONG":
                sl_price = entry * (1 + ((sl_target + fee_pct) / 100))
                tp_price = entry * (1 + ((tp_target + fee_pct) / 100))
            else:
                sl_price = entry * (1 - ((sl_target + fee_pct) / 100))
                tp_price = entry * (1 - ((tp_target + fee_pct) / 100))

            curr_price = entry
            close_pos = False
            close_reason = ""
            exit_price = entry

            try:
                df_5m = yf.download(sym, period="1d", interval="5m", progress=False)
                if not df_5m.empty:
                    if isinstance(df_5m.columns, pd.MultiIndex):
                        df_5m.columns = df_5m.columns.get_level_values(0)
                    curr_price = float(df_5m['Close'].iloc[-1])
                    for idx, row in df_5m.tail(14).iterrows():
                        if opened_dt is not None:
                            c_time = idx.replace(tzinfo=None) if hasattr(idx, 'tzinfo') else idx
                            if c_time < (opened_dt - timedelta(minutes=5)):
                                continue
                        c_high, c_low = float(row['High']), float(row['Low'])
                        if pos_type == "LONG":
                            low_net = (((c_low - entry) / entry) * 100) - fee_pct
                            if low_net <= sl_target:
                                close_pos, exit_price = True, sl_price
                                close_reason = f"🛑 STOP LOSS 5m ({cat_label} {sl_target}%)"
                                break
                            if c_high > best_price:
                                best_price = c_high
                            max_net = (((best_price - entry) / entry) * 100) - fee_pct
                            if max_net >= tp_target:
                                close_pos, exit_price = True, tp_price
                                close_reason = f"🎯 TAKE PROFIT 5m ({cat_label} +{tp_target}%)"
                                break
                            if max_net >= trail_act and (max_net - low_net) >= trail_dist:
                                close_pos = True
                                exit_price = entry * (1 + (((max_net - trail_dist) + fee_pct) / 100))
                                close_reason = f"🛡️ TRAILING STOP 5m (Picco +{max_net:.2f}%)"
                                break
                        else:
                            high_net = (((entry - c_high) / entry) * 100) - fee_pct
                            if high_net <= sl_target:
                                close_pos, exit_price = True, sl_price
                                close_reason = f"🛑 STOP LOSS 5m ({cat_label} {sl_target}%)"
                                break
                            if c_low < best_price or best_price == 0.0:
                                best_price = c_low
                            max_net = (((entry - best_price) / entry) * 100) - fee_pct
                            if max_net >= tp_target:
                                close_pos, exit_price = True, tp_price
                                close_reason = f"🎯 TAKE PROFIT 5m ({cat_label} +{tp_target}%)"
                                break
                            if max_net >= trail_act and (max_net - high_net) >= trail_dist:
                                close_pos = True
                                exit_price = entry * (1 - (((max_net - trail_dist) + fee_pct) / 100))
                                close_reason = f"🛡️ TRAILING STOP 5m (Picco +{max_net:.2f}%)"
                                break
            except Exception as e:
                print(f"Avviso 5m {sym}: {e}")

            _, _, _, live_rsi2, live_sma5 = analyze_market(sym)
            eff_price = exit_price if close_pos else curr_price

            if pos_type == "LONG":
                if not close_pos and curr_price > best_price:
                    best_price = curr_price
                pnl_gross = ((eff_price - entry) / entry) * 100
                max_gross = ((best_price - entry) / entry) * 100
            else:
                if not close_pos and (curr_price < best_price or best_price == 0.0):
                    best_price = curr_price
                pnl_gross = ((entry - eff_price) / entry) * 100
                max_gross = ((entry - best_price) / entry) * 100

            pnl_net = pnl_gross - fee_pct
            max_net = max_gross - fee_pct
            pnl_usd = slot_cap * (pnl_net / 100)
            qty = slot_cap / entry if entry > 0 else 0.0

            if not close_pos:
                if pnl_net >= tp_target:
                    close_pos, close_reason = True, f"🎯 TAKE PROFIT (+{tp_target}%)"
                elif pnl_net <= sl_target:
                    close_pos, close_reason = True, f"🛑 STOP LOSS ({sl_target}%)"
                elif max_net >= trail_act and (max_net - pnl_net) >= trail_dist:
                    close_pos, close_reason = True, f"🛡️ TRAILING STOP (Picco +{max_net:.2f}%)"
                elif pos_type == "LONG" and live_sma5 > 0 and curr_price > live_sma5 and pnl_net >= 0.35:
                    close_pos, close_reason = True, f"✅ RIMBALZO SMA5 (+{pnl_net:.2f}%)"
                elif pos_type == "SHORT" and live_sma5 > 0 and curr_price < live_sma5 and pnl_net >= 0.35:
                    close_pos, close_reason = True, f"✅ STORNO SMA5 (+{pnl_net:.2f}%)"

            if close_pos:
                msg_close = f"🔔 CHIUSO [{slot_key.upper()}]: {sym} ({pos_type}) | { pnl_usd:+.2f}$ ({pnl_net:+.2f}%) | {close_reason}"
                print(msg_close)
                print("------------------------------------------------------------------")
                send_telegram(msg_close)

                stats["total_realized_usd"] = round(stats["total_realized_usd"] + pnl_usd, 2)
                if pnl_usd >= 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                
                stats["history"].append({
                    "date": now_dt.strftime("%Y-%m-%d %H:%M"), "symbol": sym, "type": pos_type,
                    "invested_usd": round(slot_cap, 2), "pnl_usd": round(pnl_usd, 2),
                    "pnl_pct": round(pnl_net, 2), "reason": close_reason
                })
                state[slot_key] = {
                    "symbol": None, "type": None, "entry_price": 0.0, "max_price": 0.0,
                    "current_price": 0.0, "invested_capital": 0.0, "opened_at": None,
                    "pnl_net_percent": 0.0, "active": False
                }
            else:
                total_floating_usd += pnl_usd
                slot["max_price"] = round(best_price, 4)
                slot["current_price"] = round(curr_price, 4)
                slot["invested_capital"] = round(slot_cap, 2)
                slot["pnl_net_percent"] = round(pnl_net, 2)
                t_stat = f"Attivo (Picco +{max_net:.2f}%)" if max_net >= trail_act else f"Attiva a +{trail_act}%"

                print(f"⏳ ATTIVO [{slot_key.upper()}]: {sym} ({pos_type}) [{cat_label}] da {hours_open}h")
                print(f"   Cap: {slot_cap:.2f}$ | Qta: {qty:.5f} | Entrata: {entry:.4f}$ | Attuale: {curr_price:.4f}$ | RSI2: {live_rsi2:.1f}")
                print(f"   TP: {tp_price:.4f}$ | SL: {sl_price:.4f}$ | Trail: {t_stat} | SMA5: {live_sma5:.4f}$")
                print(f"   💵 P&L Netto: {pnl_usd:+.2f}$ ({pnl_net:+.2f}%)")
                print("------------------------------------------------------------------")

    free_slots = [k for k in slot_keys if not state[k].get('active')]
    active_syms = [state[k].get('symbol') for k in slot_keys if state[k].get('active')]

    if free_slots:
        print(f"🔍 Scansione Connors RSI(2) per {len(free_slots)} slot liberi...")
        candidates = []
        for sym in CRYPTO_LIST:
            if sym in active_syms:
                continue
            sig, sc, price, rsi2_val, _ = analyze_market(sym)
            if sig in ["LONG", "SHORT"] and sc > 0:
                candidates.append({"symbol": sym, "signal": sig, "score": sc, "price": price, "rsi2": rsi2_val})

        candidates.sort(key=lambda x: x["score"], reverse=True)
        updated_cap = max(50.0, INITIAL_CAPITAL + stats['total_realized_usd'])

        for idx, slot_key in enumerate(free_slots):
            if idx >= len(candidates):
                print(f"⏸️ [{slot_key.upper()}] Nessun eccesso RSI(2) trovato. Slot in attesa.")
                continue
            best = candidates[idx]
            w = SLOT_WEIGHTS.get(slot_key, 0.33)
            alloc_cap = round(updated_cap * w, 2)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            state[slot_key] = {
                "symbol": best["symbol"], "type": best["signal"],
                "entry_price": round(best["price"], 4), "max_price": round(best["price"], 4),
                "current_price": round(best["price"], 4), "invested_capital": alloc_cap,
                "opened_at": now_str, "pnl_net_percent": -fee_pct, "active": True
            }
            msg_open = f"🚀 APERTO [{slot_key.upper()}] ({int(w*100)}%): {best['symbol']} ({best['signal']}) a {best['price']:.4f}$ | Cap: {alloc_cap:.2f}$ | RSI2: {best['rsi2']:.1f}"
            print(msg_open)
            print("------------------------------------------------------------------")
            send_telegram(msg_open)

    final_cap = INITIAL_CAPITAL + stats['total_realized_usd']
    print(f"📊 P&L APERTE: {total_floating_usd:+.2f}$ | 🏦 CAPITALE COMPOSTO: {final_cap:.2f}$")
    print("==================================================================")
    print("📜 LIBRO CONTABILE (Ultime 5 operazioni):")
    for h in stats.get("history", [])[-5:]:
        icon = "✅" if h["pnl_usd"] >= 0 else "❌"
        print(f"   {icon} [{h['date']}] {h['symbol']} ({h['type']}) | {h['pnl_usd']:+.2f}$ ({h['pnl_pct']:+.2f}%) | {h['reason']}")
    print("========