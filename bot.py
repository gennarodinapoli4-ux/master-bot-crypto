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
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings
warnings.filterwarnings('ignore')

# Inizializzazione VADER
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ==========================================
# CONFIGURAZIONE PORTAFOGLIO & PESI DIFFERENZIALI
# ==========================================
INITIAL_CAPITAL = 500.0  # Capitale di partenza (base per interesse composto)
NUM_SLOTS = 3
DEFAULT_SLOT_CAPITAL = round(INITIAL_CAPITAL / NUM_SLOTS, 2)
STATE_FILE = "portafoglio_stato.json"
EXCHANGE_FEE_RATE = 0.001  # 0.1% fee per operazione (0.2% totale round-trip)

# Allocazione Differenziale sui 3 Slot in base alla forza del segnale (45% / 35% / 20%)
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

# Parametri Big Cap (Tarati per alto Win Rate: incasso sul rimbalzo e respiro contro gli spike 5m)
BIG_CAP_TP = 3.5          # Take Profit rapido se fa spike (+3.5% netto)
BIG_CAP_SL = -5.5         # Stop Loss ampio (-5.5%) per evitare caccia agli stop sui 5m
BIG_CAP_TRAIL_ACT = 1.6   # Attiva scudo Trailing già a +1.6% netto
BIG_CAP_TRAIL_DIST = 0.5  # Chiude se ritraccia dello 0.5% dal picco

# Parametri Altcoin Selezionate
ALT_TP = 5.0              # Take Profit rapido (+5.0% netto)
ALT_SL = -7.0             # Stop Loss di protezione (-7.0% netto)
ALT_TRAIL_ACT = 2.2       # Attiva scudo Trailing a +2.2% netto
ALT_TRAIL_DIST = 0.7      # Chiude se ritraccia dello 0.7% dal picco

# ==========================================
# WHITELIST ISTITUZIONALE & TOP ESPERTI
# ==========================================
INSTITUTIONAL_WHITELIST = [
    "CoinDesk", "Cointelegraph", "Bloomberg Crypto", "Glassnode", 
    "Vitalik Buterin", "CoinMetrics", "WatcherGuru", "Wu Blockchain"
]

# ==========================================
# PANIERE CRIPTO AD ALTA LIQUIDITÀ (Rimossi Meme/Shitcoin)
# ==========================================
CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", 
    "SUI-USD", "NEAR-USD", "RENDER-USD", "FET-USD", "DOT-USD", "UNI-USD", 
    "ICP-USD", "ARB-USD", "OP-USD", "INJ-USD", "TIA-USD", "SEI-USD", 
    "AAVE-USD", "ONDO-USD", "ALGO-USD", "FIL-USD", "ATOM-USD", 
    "JUP-USD", "CRV-USD", "LTC-USD", "BCH-USD", "HBAR-USD", "TON-USD"
]

# ==========================================
# GESTIONE STATO SLOT E LIBRO CONTABILE
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
            print(f"Errore lettura stato esistente: {e}")
            
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
    except Exception:
        return 0.0

# ==========================================
# 2. MOTORE ISTITUZIONALE CONNORS RSI(2) (EWM)
# ==========================================
def analyze_market(symbol):
    """
    Ritorna: (signal, conviction_score, current_price, current_rsi2, sma5_price)
    """
    try:
        # Scarichiamo 1 mese a candele orarie (1h) per avere una SMA 200 solida e reattiva
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

        # Calcolo Originale Connors RSI a 2 periodi con metodo Wilder (EWM alpha=1/2)
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

        # REGOLA 1: LONG SU IPERVENDUTO ESTREMO IN TREND RIALZISTA (Prezzo > SMA200 e RSI2 < 10)
        if current_price > sma200 and current_rsi2 < 10.0:
            if inst_score >= -0.4:  # Evita solo se c'è un hack/crollo catastrofico nelle news
                # Più l'RSI2 è vicino a 0, più il punteggio di convinzione è alto (fino a 10+)
                score = (10.0 - current_rsi2) + (inst_score * 1.5)
                signal = "LONG"
                
        # REGOLA 2: SHORT SU EUFORIA ESTREMA IN TREND RIBASSISTA (Prezzo < SMA200 e RSI2 > 90)
        elif current_price < sma200 and current_rsi2 > 90.0:
            if inst_score <= 0.4:
                # Più l'RSI2 è vicino a 100, più il punteggio di convinzione è alto
                score = (current_rsi2 - 90.0) + (abs(inst_score) * 1.5)
                signal = "SHORT"

        return signal, round(score, 2), current_price, round(current_rsi2, 1), sma5
    except Exception:
        return "NEUTRAL", 0.0, 0.0, 50.0, 0.0

# ==========================================
# 3. ESECUZIONE PRINCIPALE (FULL AUTONOMOUS)
# ==========================================
def run_trading_bot():
    now_dt = datetime.now()
    print(f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Avvio Master Bot v21.0 (Connors RSI2 High-WinRate + Pesi Differenziali)")
    print("==================================================================")
    
    state = load_state()
    stats = state.get("stats", {"total_realized_usd": 0.0, "wins": 0, "losses": 0, "history": []})
    
    compounded_total_capital = max(50.0, INITIAL_CAPITAL + stats['total_realized_usd'])
    total_closed = stats['wins'] + stats['losses']
    win_rate_pct = (stats['wins'] / total_closed * 100) if total_closed > 0 else 0.0

    print(f"💼 CAPITALE COMPOSTO TOTALE: {compounded_total_capital:.2f} $ (Base: {INITIAL_CAPITAL:.0f}$ | Cassa: {stats['total_realized_usd']:+.2f}$)")
    print(f"🏆 Bilancio Trade Chiusi: {stats['wins']} Vinte | {stats['losses']} Perse (Win Rate: {win_rate_pct:.1f}%)")
    print(f"⚖️ Allocazione Differenziale Slot: S1 (45%): {compounded_total_capital*0.45:.1f}$ | S2 (35%): {compounded_total_capital*0.35:.1f}$ | S3 (20%): {compounded_total_capital*0.20:.1f}$")
    print("==================================================================")

    total_floating_usd = 0.0
    slot_keys = ["slot_1", "slot_2", "slot_3"]
    fee_pct = EXCHANGE_FEE_RATE * 2 * 100  # 0.2%
    
    # ----------------------------------------------------
    # FASE 1: GESTIONE E MONITORAGGIO POSIZIONI APERTE
    # ----------------------------------------------------
    for slot_key in slot_keys:
        slot = state[slot_key]
        if slot.get('active') and slot.get('symbol'):
            sym = slot['symbol']
            pos_type = slot['type']
            entry = float(slot['entry_price'])
            slot_cap = float(slot.get('invested_capital', DEFAULT_SLOT_CAPITAL))
            best_price = float(slot.get('max_price', entry))
            if best_price == 0.0:
                best_price = entry

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

            is_big_cap = sym in BIG_CAP_LIST
            cat_label = "BIG CAP" if is_big_cap else "ALTCOIN"
            tp_target = BIG_CAP_TP if is_big_cap else ALT_TP
            sl_target = BIG_CAP_SL if is_big_cap else ALT_SL
            trail_act = BIG_CAP_TRAIL_ACT if is_big_cap else ALT_TRAIL_ACT
            trail_dist = BIG_CAP_TRAIL_DIST if is_big_cap else ALT_TRAIL_DIST

            if pos_type == "LONG":
                sl_price = entry * (1 + ((sl_target + fee_pct) / 100))
                tp_price = entry * (1 + ((tp_target + fee_pct) / 100))
            else:
                sl_price = entry * (1 - ((sl_target + fee_pct) / 100))
                tp_price = entry * (1 - ((tp_target + fee_pct) / 100))

            # --- SCANSIONE RETROATTIVA A 5 MINUTI ---
            curr_price = entry
            close_position = False
            close_reason = ""
            exit_price = entry

            try:
                df_5m = yf.download(sym, period="1d", interval="5m", progress=False)
                if not df_5m.empty:
                    if isinstance(df_5m.columns, pd.MultiIndex):
                        df_5m.columns = df_5m.columns.get_level_values(0)
                    
                    curr_price = float(df_5m['Close'].iloc[-1])
                    recent_candles = df_5m.tail(14)

                    for idx, row in recent_candles.iterrows():
                        if opened_dt is not None:
                            candle_time = idx.replace(tzinfo=None) if hasattr(idx, 'tzinfo') else idx
                            if candle_time < (opened_dt - timedelta(minutes=5)):
                                continue

                        c_high = float(row['High'])
                        c_low = float(row['Low'])

                        if pos_type == "LONG":
                            low_pnl_net = (((c_low - entry) / entry) * 100) - fee_pct
                            if low_pnl_net <= sl_target:
                                close_position = True
                                exit_price = sl_price
                                close_reason = f"🛑 STOP LOSS INTRA-ORA ({cat_label} {sl_target}% toccato sui 5m)"
                                break

                            if c_high > best_price:
                                best_price = c_high
                            max_pnl_net = (((best_price - entry) / entry) * 100) - fee_pct

                            if max_pnl_net >= tp_target:
                                close_position = True
                                exit_price = tp_price
                                close_reason = f"🎯 TAKE PROFIT INTRA-ORA ({cat_label} +{tp_target}% toccato sui 5m)"
                                break

                            if max_pnl_net >= trail_act and (max_pnl_net - low_pnl_net) >= trail_dist:
                                close_position = True
                                trail_exit_net = max_pnl_net - trail_dist
                                exit_price = entry * (1 + ((trail_exit_net + fee_pct) / 100))
                                close_reason = f"🛡️ TRAILING STOP INTRA-ORA (Picco +{max_pnl_net:.2f}%, incasso protetto sui 5m!)"
                                break

                        else: # SHORT
                            high_pnl_net = (((entry - c_high) / entry) * 100) - fee_pct
                            if high_pnl_net <= sl_target:
                                close_position = True
                                exit_price = sl_price
                                close_reason = f"🛑 STOP LOSS INTRA-ORA ({cat_label} {sl_target}% toccato sui 5m)"
                                break

                            if c_low < best_price or best_price == 0.0:
                                best_price = c_low
                            max_pnl_net = (((entry - best_price) / entry) * 100) - fee_pct

                            if max_pnl_net >= tp_target:
                                close_position = True
                                exit_price = tp_price
                                close_reason = f"🎯 TAKE PROFIT INTRA-ORA ({cat_label} +{tp_target}% toccato sui 5m)"
                                break

                            if max_pnl_net >= trail_act and (max_pnl_net - high_pnl_net) >= trail_dist:
                                close_position = True
                                trail_exit_net = max_pnl_net - trail_dist
                                exit_price = entry * (1 - ((trail_exit_net + fee_pct) / 100))
                                close_reason = f"🛡️ TRAILING STOP INTRA-ORA (Picco +{max_pnl_net:.2f}%, incasso protetto sui 5m!)"
                                break

            except Exception as e:
                print(f"Impossibile aggiornare candele 5m per {sym}: {e}")

            live_signal, live_score, _, live_rsi2, live_sma5 = analyze_market(sym)
            effective_price = exit_price if close_position else curr_price

            if pos_type == "LONG":
                if not close_position and curr_price > best_price:
                    best_price = curr_price
                pnl_gross = ((effective_price - entry) / entry) * 100
                max_pnl_gross = ((best_price - entry) / entry) * 100
            else: # SHORT
                if not close_position and (curr_price < best_price or best_price == 0.0):
                    best_price = curr_price
                pnl_gross = ((entry - effective_price) / entry) * 100
                max_pnl_gross = ((entry - best_price) / entry) * 100

            pnl_net_pct = pnl_gross - fee_pct
            max_pnl_net_pct = max_pnl_gross - fee_pct
            pnl_usd = slot_cap * (pnl_net_pct / 100)
            coin_qty = slot_cap / entry if entry > 0 else 0.0
            coin_name = sym.replace("-USD", "")

            # --- USCITA CONNORS SMA 5 (MEAN REVERSION) & FALLBACK ---
            if not close_position:
                if pnl_net_pct >= tp_target:
                    close_position = True
                    close_reason = f"🎯 TAKE PROFIT ({cat_label} +{tp_target}%)"
                elif pnl_net_pct <= sl_target:
                    close_position = True
                    close_reason = f"🛑 STOP LOSS ({cat_label} {sl_target}%)"
                elif max_pnl_net_pct >= trail_act and (max_pnl_net_pct - pnl_net_pct) >= trail_dist:
                    close_position = True
                    close_reason = f"🛡️ TRAILING STOP (Picco era +{max_pnl_net_pct:.2f}%, incasso protetto!)"
                # Regola aurea Connors: se siamo in profitto netto (> +0.35%) e il prezzo ha rimbalzato oltre la SMA 5, incassiamo subito!
                elif pos_type == "LONG" and live_sma5 > 0 and curr_price > live_sma5 and pnl_net_pct >= 0.35:
                    close_position = True
                    close_reason = f"✅ RIMBALZO CONNORS SMA5 COMPLETATO (Prezzo > SMA5 in profitto +{pnl_net_pct:.2f}%)"
                elif pos_type == "SHORT" and live_sma5 > 0 and curr_price < live_sma5 and pnl_net_pct >= 0.35:
                    close_position = True
                    close_reason = f"✅ STORNO CONNORS SMA5 COMPLETATO (Prezzo < SMA5 in profitto +{pnl_net_pct:.2f}%)"

            if close_position:
                print(f"🔔 CHIUSURA AUTOMATICA [{slot_key.upper()}]: {sym} ({pos_type}) dopo {hours_open} ore")
                print(f"   Motivo: {close_reason}")
                print(f"   Capitale Slot: {slot_cap:.2f} $ | Entrata: {entry:.4f} $ | Uscita: {effective_price:.4f} $")
                print(f"   💰 Profitto/Perdita incassata (netto fee): {pnl_usd:+.2f} $ ({pnl_net_pct:+.2f}%)")
                print("------------------------------------------------------------------")

                stats["total_realized_usd"] = round(stats["total_realized_usd"] + pnl_usd, 2)
                if pnl_usd >= 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                
                stats["history"].append({
                    "date": now_dt.strftime("%Y-%m-%d %H:%M"),
                    "symbol": sym,
                    "type": pos_type,
                    "invested_usd": round(slot_cap, 2),
                    "pnl_usd": round(pnl_usd, 2),
                    "pnl_pct": round(pnl_net_pct, 2),
                    "reason": close_reason
                })

                state[slot_key] = {
                    "symbol": None, "type": None, "entry_price": 0.0, 
                    "max_price": 0.0, "current_price": 0.0, "invested_capital": 0.0,
                    "opened_at": None, "pnl_net_percent": 0.0, "active": False
                }
            else:
                total_floating_usd += pnl_usd
                slot['max_price'] = round(best_price, 4)
                slot['current_price'] = round(curr_price, 4)
                slot['invested_capital'] = round(slot
