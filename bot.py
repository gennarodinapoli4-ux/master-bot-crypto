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
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Inizializzazione VADER
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ==========================================
# CONFIGURAZIONE PORTAFOGLIO & FEE
# ==========================================
INITIAL_CAPITAL = 500.0  # Capitale di partenza (base per interesse composto)
NUM_SLOTS = 3
DEFAULT_SLOT_CAPITAL = round(INITIAL_CAPITAL / NUM_SLOTS, 2)  # ~166.67 $ iniziali
STATE_FILE = "portafoglio_stato.json"
EXCHANGE_FEE_RATE = 0.001  # 0.1% fee per operazione (0.2% totale round-trip)

# ==========================================
# PARAMETRI STOP WIN / STOP LOSS DIFFERENZIATI
# ==========================================
BIG_CAP_LIST = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", 
    "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
    "LTC-USD", "BCH-USD", "TON-USD"
]

# Parametri Big Cap (più stretti e realistici)
BIG_CAP_TP = 6.0          # Stop Win secco a +6% netto
BIG_CAP_SL = -3.0         # Stop Loss a -3% netto
BIG_CAP_TRAIL_ACT = 2.5   # Attiva Trailing Stop sopra +2.5% di profitto
BIG_CAP_TRAIL_DIST = 1.0  # Vende se scende dell'1.0% dal picco massimo

# Parametri Altcoin (più ampi per cavalcare la volatilità)
ALT_TP = 12.0             # Stop Win secco a +12% netto
ALT_SL = -4.0             # Stop Loss a -4% netto
ALT_TRAIL_ACT = 4.0       # Attiva Trailing Stop sopra +4.0% di profitto
ALT_TRAIL_DIST = 1.5      # Vende se scende dell'1.5% dal picco massimo

# ==========================================
# WHITELIST ISTITUZIONALE & TOP ESPERTI
# ==========================================
INSTITUTIONAL_WHITELIST = [
    "CoinDesk", "Cointelegraph", "Bloomberg Crypto", "Glassnode", 
    "Vitalik Buterin", "CoinMetrics", "WatcherGuru", "Wu Blockchain"
]

# ==========================================
# PANIERE TOP CRIPTO PULITO (Zero errori Yahoo)
# ==========================================
CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", 
    "SUI-USD", "NEAR-USD", "RENDER-USD", "FET-USD", "PEPE-USD", "SHIB-USD", "WIF-USD",
    "DOGE-USD", "DOT-USD", "UNI-USD", "ICP-USD", "ARB-USD", "OP-USD",
    "INJ-USD", "TIA-USD", "SEI-USD", "AAVE-USD", "ONDO-USD",
    "ALGO-USD", "FIL-USD", "ATOM-USD", "KAS-USD", "JUP-USD", "CRV-USD",
    "LTC-USD", "BCH-USD", "HBAR-USD", "TON-USD"
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
        "slot_1": {"symbol": "ETH-USD", "type": "SHORT", "entry_price": 2478.01, "max_price": 2478.01, "invested_capital": DEFAULT_SLOT_CAPITAL, "opened_at": now_str, "active": True},
        "slot_2": {"symbol": "TIA-USD", "type": "SHORT", "entry_price": 0.5526, "max_price": 0.5526, "invested_capital": DEFAULT_SLOT_CAPITAL, "opened_at": now_str, "active": True},
        "slot_3": {"symbol": "CRV-USD", "type": "SHORT", "entry_price": 0.3278, "max_price": 0.3278, "invested_capital": DEFAULT_SLOT_CAPITAL, "opened_at": now_str, "active": True},
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
# 2. MOTORE TECNICO + WHITELIST SENTIMENT
# ==========================================
def analyze_market(symbol):
    try:
        df = yf.download(symbol, period="5d", interval="1h", progress=False)
        if df.empty or len(df) < 50:
            return None, 0.0, 0.0, 50.0
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close']
        volume = df['Volume']

        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        current_price = float(close.iloc[-1])
        vol_mean = float(volume.rolling(20).mean().iloc[-1])
        current_vol = float(volume.iloc[-1])

        # RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])

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

        return signal, score, current_price, current_rsi
    except Exception:
        return None, 0.0, 0.0, 50.0

# ==========================================
# 3. ESECUZIONE PRINCIPALE (FULL AUTONOMOUS)
# ==========================================
def run_trading_bot():
    now_dt = datetime.now()
    print(f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Avvio Master Bot v20.3 (Full Autonomous + 5m High/Low Real-Sim)")
    print("==================================================================")
    
    state = load_state()
    stats = state.get("stats", {"total_realized_usd": 0.0, "wins": 0, "losses": 0, "history": []})
    
    compounded_total_capital = INITIAL_CAPITAL + stats['total_realized_usd']
    next_slot_capital = round(compounded_total_capital / NUM_SLOTS, 2)

    print(f"💼 CAPITALE COMPOSTO TOTALE: {compounded_total_capital:.2f} $ (Base: {INITIAL_CAPITAL:.0f}$ | Cassa: {stats['total_realized_usd']:+.2f}$)")
    print(f"🏆 Bilancio Trade Chiusi: {stats['wins']} Vinte | {stats['losses']} Perse | Prossima Puntata Slot: {next_slot_capital:.2f} $")
    print("==================================================================")

    total_floating_usd = 0.0
    slot_keys = ["slot_1", "slot_2", "slot_3"]
    fee_pct = EXCHANGE_FEE_RATE * 2 * 100  # 0.2%
    
    for slot_key in slot_keys:
        slot = state[slot_key]
        if slot['active'] and slot['symbol']:
            sym = slot['symbol']
            pos_type = slot['type']
            entry = float(slot['entry_price'])
            slot_cap = float(slot.get('invested_capital', DEFAULT_SLOT_CAPITAL))
            best_price = float(slot.get('max_price', entry))
            if best_price == 0.0:
                best_price = entry

            # Calcolo ore di apertura
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

            # Imposta i paletti in base alla Market Cap
            is_big_cap = sym in BIG_CAP_LIST
            cat_label = "BIG CAP" if is_big_cap else "ALTCOIN"
            tp_target = BIG_CAP_TP if is_big_cap else ALT_TP
            sl_target = BIG_CAP_SL if is_big_cap else ALT_SL
            trail_act = BIG_CAP_TRAIL_ACT if is_big_cap else ALT_TRAIL_ACT
            trail_dist = BIG_CAP_TRAIL_DIST if is_big_cap else ALT_TRAIL_DIST

            # Calcolo prezzi soglia esatti (includendo compensazione fee 0.2% per netto reale)
            if pos_type == "LONG":
                sl_price = entry * (1 + ((sl_target + fee_pct) / 100))
                tp_price = entry * (1 + ((tp_target + fee_pct) / 100))
            else:
                sl_price = entry * (1 - ((sl_target + fee_pct) / 100))
                tp_price = entry * (1 - ((tp_target + fee_pct) / 100))

            # --- SCANSIONE RETROATTIVA A 5 MINUTI (HIGH / LOW DELL'ULTIMA ORA) ---
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
                    
                    # Prende le ultime 14 candele da 5m (~70 minuti) per coprire l'intera ora trascorsa
                    recent_candles = df_5m.tail(14)

                    for idx, row in recent_candles.iterrows():
                        # Ignora candele precedenti all'apertura del trade (se aperto da meno di 1 ora)
                        if opened_dt is not None:
                            candle_time = idx.replace(tzinfo=None) if hasattr(idx, 'tzinfo') else idx
                            if candle_time < (opened_dt - timedelta(minutes=5)):
                                continue

                        c_high = float(row['High'])
                        c_low = float(row['Low'])

                        if pos_type == "LONG":
                            # 1. Controllo Stop Loss intra-candela (sul Minimo toccato)
                            low_pnl_net = (((c_low - entry) / entry) * 100) - fee_pct
                            if low_pnl_net <= sl_target:
                                close_position = True
                                exit_price = sl_price
                                close_reason = f"🛑 STOP LOSS INTRA-ORA ({cat_label} {sl_target}% toccato sui 5m)"
                                break

                            # 2. Aggiorna Picco Massimo (sul Massimo toccato) e controlla Take Profit
                            if c_high > best_price:
                                best_price = c_high
                            max_pnl_net = (((best_price - entry) / entry) * 100) - fee_pct

                            if max_pnl_net >= tp_target:
                                close_position = True
                                exit_price = tp_price
                                close_reason = f"🎯 TAKE PROFIT INTRA-ORA ({cat_label} +{tp_target}% toccato sui 5m)"
                                break

                            # 3. Controllo Trailing Stop intra-candela
                            if max_pnl_net >= trail_act and (max_pnl_net - low_pnl_net) >= trail_dist:
                                close_position = True
                                trail_exit_net = max_pnl_net - trail_dist
                                exit_price = entry * (1 + ((trail_exit_net + fee_pct) / 100))
                                close_reason = f"🛡️ TRAILING STOP INTRA-ORA (Picco +{max_pnl_net:.2f}%, incasso protetto sui 5m!)"
                                break

                        else: # SHORT
                            # 1. Controllo Stop Loss intra-candela (sul Massimo toccato)
                            high_pnl_net = (((entry - c_high) / entry) * 100) - fee_pct
                            if high_pnl_net <= sl_target:
                                close_position = True
                                exit_price = sl_price
                                close_reason = f"🛑 STOP LOSS INTRA-ORA ({cat_label} {sl_target}% toccato sui 5m)"
                                break

                            # 2. Aggiorna Picco Minimo (sul Minimo toccato) e controlla Take Profit
                            if c_low < best_price or best_price == 0.0:
                                best_price = c_low
                            max_pnl_net = (((entry - best_price) / entry) * 100) - fee_pct

                            if max_pnl_net >= tp_target:
                                close_position = True
                                exit_price = tp_price
                                close_reason = f"🎯 TAKE PROFIT INTRA-ORA ({cat_label} +{tp_target}% toccato sui 5m)"
                                break

                            # 3. Controllo Trailing Stop intra-candela
                            if max_pnl_net >= trail_act and (max_pnl_net - high_pnl_net) >= trail_dist:
                                close_position = True
                                trail_exit_net = max_pnl_net - trail_dist
                                exit_price = entry * (1 - ((trail_exit_net + fee_pct) / 100))
                                close_reason = f"🛡️ TRAILING STOP INTRA-ORA (Picco +{max_pnl_net:.2f}%, incasso protetto sui 5m!)"
                                break

            except Exception as e:
                print(f"Impossibile aggiornare candele 5m per {sym}: {e}")

            # Analisi tecnica e sentiment attuale per verificare inversioni di trend
            live_signal, live_score, _, live_rsi = analyze_market(sym)

            # Se i 5m hanno fatto scattare un ordine, usa il prezzo di esecuzione dello stop/target
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

            # --- CONTROLLO FALLBACK E INVERSIONE TREND A FINE ORA ---
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
                elif ((pos_type == "LONG" and live_signal == "SHORT") or (pos_type == "SHORT" and live_signal == "LONG")) and live_score >= 2.2:
                    close_position = True
                    close_reason = f"🔄 INVERSIONE TREND/NOTIZIE (Segnale opposto {live_signal} forte)"

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

                # Libera lo slot
                state[slot_key] = {
                    "symbol": None, "type": None, "entry_price": 0.0, 
                    "max_price": 0.0, "current_price": 0.0, "invested_capital": 0.0,
                    "opened_at": None, "pnl_net_percent": 0.0, "active": False
                }
            else:
                total_floating_usd += pnl_usd
                slot['max_price'] = round(best_price, 4)
                slot['current_price'] = round(curr_price, 4)
                slot['invested_capital'] = round(slot_cap, 2)
                slot['pnl_net_percent'] = round(pnl_net_pct, 2)

                trail_status = f"Attivo (Picco: +{max_pnl_net_pct:.2f}%)" if max_pnl_net_pct >= trail_act else f"Si attiva a +{trail_act}%"

                print(f"⏳ OPERAZIONE ATTIVA SU [{slot_key.upper()}]: {sym} ({pos_type}) [{cat_label}] da {hours_open} ore")
                print(f"   Capitale Puntato: {slot_cap:.2f} $ | Quantità: {coin_qty:.5f} {coin_name}")
                print(f"   Entrata: {entry:.4f} $ | Attuale: {curr_price:.4f} $ | RSI: {live_rsi:.1f}")
                print(f"   Target TP (+{tp_target}%): {tp_price:.4f} $ | Stop Loss ({sl_target}%): {sl_price:.4f} $")
                print(f"   🛡️ Trailing Stop: {trail_status} | Scansione 5m High/Low OK, mantengo!")
                print(f"   💵 Guadagno/Perdita netta attuale (già tolte le fee): {pnl_usd:+.2f} $ ({pnl_net_pct:+.2f}%)")
                print("------------------------------------------------------------------")
        else:
            print(f"💤 [{slot_key.upper()}]: Slot Libero (In cerca di opportunità...)")
            print("------------------------------------------------------------------")

    state["stats"] = stats
    compounded_total_capital = INITIAL_CAPITAL + stats['total_realized_usd']
    
    print(f"📊 P&L POSIZIONI APERTE: {total_floating_usd:+.2f} $ | 🏦 NUOVO CAPITALE COMPOSTO: {compounded_total_capital:.2f} $")
    print("==================================================================")

    # --- STAMPA LIBRO CONTABILE (ULTIME 5 OPERAZIONI CHIUSE) ---
    if stats["history"]:
        print("📜 LIBRO CONTABILE (Ultime operazioni concluse):")
        for trade in stats["history"][-5:]:
            esito_icon = "✅" if trade["pnl_usd"] >= 0 else "❌"
            print(f"   {esito_icon} [{trade['date']}] {trade['symbol']} ({trade['type']}) | Netto: {trade['pnl_usd']:+.2f}$ ({trade['pnl_pct']:+.2f}%) | {trade['reason']}")
    else:
        print("📜 LIBRO CONTABILE: Nessuna operazione ancora conclusa.")
    print("==================================================================")

    save_state(state)

    # --- RICERCA NUOVE OPPORTUNITÀ MULTI-SLOT ---
    free_slots = [k for k in slot_keys if not state[k]['active']]

    if free_slots:
        new_slot_cap = round((INITIAL_CAPITAL + stats['total_realized_usd']) / NUM_SLOTS, 2)
        print(f"🔍 Ricerca segnali per {len(free_slots)} slot liberi (Budget composto per slot: {new_slot_cap:.2f} $)...")
        
        candidates = []
        active_symbols = [state[k]['symbol'] for k in slot_keys if state[k]['active']]

        for crypto in CRYPTO_LIST:
            if crypto in active_symbols:
                continue
                
            signal, score, price, _ = analyze_market(crypto)
            if score >= 1.8 and signal in ["LONG", "SHORT"]:
                candidates.append((score, crypto, signal, price))

        # Ordina i candidati dal punteggio più alto al più basso
        candidates.sort(key=lambda x: x[0], reverse=True)

        opened_count = 0
        for target_slot in free_slots:
            if opened_count < len(candidates):
                score, crypto, signal, price = candidates[opened_count]
                state[target_slot] = {
                    "symbol": crypto,
                    "type": signal,
                    "entry_price": round(float(price), 4),
                    "max_price": round(float(price), 4),
                    "current_price": round(float(price), 4),
                    "invested_capital": new_slot_cap,
                    "opened_at": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "pnl_net_percent": 0.0,
                    "active": True
                }
                opened_count += 1
                print(f"🚀 NUOVA APERTURA [{target_slot.upper()}] | Asset: {crypto} | Direzione: {signal} (Score: {score:.2f}) | Entrata: {price:.4f} $ | Puntata: {new_slot_cap:.2f} $")

        if opened_count > 0:
            save_state(state)
        else:
            print("Nessun nuovo asset ha superato i filtri rigorosi in questo ciclo.")
    else:
        print("Tutti i 3 slot sono operativi. Monitoraggio e protezione attivi.")

if __name__ == "__main__":
    run_trading_bot()
