import yfinance as yf
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
import datetime
import json
import os

warnings.filterwarnings('ignore')

# =====================================================================
# MASTER BOT v16.1 - DYNAMIC PROFIT RUNNER (+15% Target + Live News)
# =====================================================================

STATE_FILE = "stato_portafoglio.json"
HISTORY_FILE = "storico_operazioni.csv"
CAPITALE_INIZIALE = 150.0

FEE_TOTALE_PERC = 0.003  # 0.30% totale A/R
MAX_ORE_OPERAZIONE = 72  # Più tempo (72h) per permettere di raggiungere il +15%

UNIVERSE_CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", 
    "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD", 
    "NEAR-USD", "ATOM-USD", "LTC-USD", "ETC-USD", "XLM-USD", 
    "ALGO-USD", "BCH-USD", "ICP-USD", "FIL-USD", "ARB-USD", "HBAR-USD"
]

analyzer = SentimentIntensityAnalyzer()
now_dt = datetime.datetime.now(datetime.timezone.utc)
now_str = now_dt.strftime("%Y-%m-%d %H:%M UTC")

if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
else:
    state = {
        "capitale": CAPITALE_INIZIALE,
        "vinte": 0,
        "perse": 0,
        "commissioni_pagate": 0.0,
        "posizione_aperta": None
    }

def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=4)

def append_history(record):
    df_rec = pd.DataFrame([record])
    if os.path.exists(HISTORY_FILE):
        df_rec.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        df_rec.to_csv(HISTORY_FILE, mode='w', header=True, index=False)

def get_eur_usd_rate():
    try:
        df_fx = yf.download("EURUSD=X", period="5d", interval="1h", progress=False)
        if isinstance(df_fx.columns, pd.MultiIndex):
            df_fx.columns = df_fx.columns.droplevel(1)
        return float(df_fx['Close'].iloc[-1])
    except Exception:
        return 1.08

def calcola_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_market_sentiment():
    news_urls = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outbound/rss/"
    ]
    sentiments = []
    for url in news_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                text = entry.title + " " + getattr(entry, 'summary', '')
                vs = analyzer.polarity_scores(text)
                sentiments.append(vs['compound'])
        except Exception:
            continue
    return sum(sentiments) / len(sentiments) if sentiments else 0.0

eur_usd = get_eur_usd_rate()
market_sentiment = get_market_sentiment()  # Legge SEMPRE le notizie, anche a trade aperto!
diff_cap = state['capitale'] - CAPITALE_INIZIALE
perc_cap = (diff_cap / CAPITALE_INIZIALE) * 100

print(f"🚀 Avvio Master Bot v16.1 (Dynamic Runner +15%) - {now_str}")
print("==============================================================")
print(f"💰 CAPITALE NETTO DISPONIBILE: {state['capitale']:.2f} € (Cambio EUR/USD: {eur_usd:.4f})")
print(f"📈 Profitto/Perdita Totale: {diff_cap:+.2f} € ({perc_cap:+.2f}%)")
print(f"🏆 Bilancio Trade: {state.get('vinte', 0)} Vinte | {state.get('perse', 0)} Perse | Fee pagate: {state.get('commissioni_pagate', 0.0):.2f} €")
print(f"🌐 Sentiment Globale Notizie Attuale: {market_sentiment:+.3f}")
print("==============================================================")

# =====================================================================
# FASE 1: GESTIONE POSIZIONE APERTA (Con Analisi News + Trailing al 15%)
# =====================================================================
if state["posizione_aperta"] is not None:
    pos = state["posizione_aperta"]
    ticker = pos["ticker"]
    coin_name = ticker.replace("-USD", "")
    entry = pos["entry_price"]
    
    # Aggiorna automaticamente il Take Profit al 15% se era rimasto all'8% della v16.0
    target_15 = entry * 1.15 if pos["tipo"] == "LONG" else entry * 0.85
    if abs(pos["tp"] - target_15) > 1.0:
        pos["tp"] = target_15
    
    open_dt = datetime.datetime.strptime(pos["data_apertura"], "%Y-%m-%d %H:%M UTC").replace(tzinfo=datetime.timezone.utc)
    ore_aperte = (now_dt - open_dt).total_seconds() / 3600.0
    
    print(f"⏳ OPERAZIONE ATTIVA SU: {ticker} ({pos['tipo']}) da {ore_aperte:.1f} ore")
    print(f"   Capitale Puntato: {pos['capitale_investito']:.2f} € ({pos['dollari_investiti']:.2f} $)")
    print(f"   Quantità in tasca: {pos['quantita']:.6f} {coin_name}")
    print(f"   Entrata: {entry:.4f} $ | SL Attuale: {pos['sl']:.4f} $ | Super TP (+15%): {pos['tp']:.4f} $")
    
    try:
        df_15m = yf.download(ticker, period="1d", interval="15m", progress=False)
        df_1h = yf.download(ticker, period="30d", interval="1h", progress=False)
        if isinstance(df_15m.columns, pd.MultiIndex):
            df_15m.columns = df_15m.columns.droplevel(1)
        if isinstance(df_1h.columns, pd.MultiIndex):
            df_1h.columns = df_1h.columns.droplevel(1)
            
        ultime_candele = df_15m.tail(4)
        current_price = float(df_15m['Close'].iloc[-1])
        high_ora = float(ultime_candele['High'].max())
        low_ora = float(ultime_candele['Low'].min())
        
        df_1h['SMA_20'] = df_1h['Close'].rolling(window=20).mean()
        df_1h['SMA_50'] = df_1h['Close'].rolling(window=50).mean()
        df_1h['RSI'] = calcola_rsi(df_1h['Close'], 14)
        sma_20 = float(df_1h['SMA_20'].iloc[-1])
        sma_50 = float(df_1h['SMA_50'].iloc[-1])
        rsi_attuale = float(df_1h['RSI'].iloc[-1])
        
        var_attuale_perc = ((current_price - entry) / entry) if pos["tipo"] == "LONG" else ((entry - current_price) / entry)
        max_raggiunto_perc = ((high_ora - entry) / entry) if pos["tipo"] == "LONG" else ((entry - low_ora) / entry)
        
        print(f"📊 Prezzo Attuale: {current_price:.4f} $ ({var_attuale_perc*100:+.2f}%) | RSI: {rsi_attuale:.1f}")
        
        chiusa = False
        esito = ""
        prezzo_uscita = current_price
        
        # 1. Verifica se nell'ultima ora ha toccato lo Stop Loss o il Take Profit (+15%)
        if pos["tipo"] == "LONG":
            if low_ora <= pos["sl"]:
                chiusa = True
                prezzo_uscita = pos["sl"]
                esito = "PROFITTO PROTETTO (Trailing Stop) 🛡️" if pos["sl"] > entry else "STOP LOSS ❌"
            elif high_ora >= pos["tp"]:
                chiusa = True
                prezzo_uscita = pos["tp"]
                esito = "SUPER TAKE PROFIT (+15%) 🎯🚀"
            else:
                # SCALINO DINAMICO: Se supera +6%, insegue il prezzo a -2.5% di distanza!
                if max_raggiunto_perc >= 0.06:
                    nuovo_sl = high_ora * 0.975
                    if nuovo_sl > pos["sl"]:
                        pos["sl"] = nuovo_sl
                        pos["breakeven_attivo"] = True
                        print(f"🚀 CORSA DEL PROFITTO! Prezzo salito a +{max_raggiunto_perc*100:.1f}%. Alzo lo Stop Win a {pos['sl']:.4f} $!")
                # Se supera +3.5%, mette lo Stop in pareggio (+0.4% per coprire le fee)
                elif max_raggiunto_perc >= 0.035 and not pos.get("breakeven_attivo", False):
                    pos["sl"] = entry * 1.004
                    pos["breakeven_attivo"] = True
                    print(f"🔒 STOP A PAREGGIO ATTIVATO! Superato +3.5%, Stop Loss spostato a {pos['sl']:.4f} $ (Zero Rischio!)")
        else:  # SHORT
            if high_ora >= pos["sl"]:
                chiusa = True
                prezzo_uscita = pos["sl"]
                esito = "PROFITTO PROTETTO (Trailing Stop) 🛡️" if pos["sl"] < entry else "STOP LOSS ❌"
            elif low_ora <= pos["tp"]:
                chiusa = True
                prezzo_uscita = pos["tp"]
                esito = "SUPER TAKE PROFIT (+15%) 🎯🚀"
            else:
                if max_raggiunto_perc >= 0.06:
                    nuovo_sl = low_ora * 1.025
                    if nuovo_sl < pos["sl"]:
                        pos["sl"] = nuovo_sl
                        pos["breakeven_attivo"] = True
                        print(f"🚀 CORSA DEL PROFITTO! Prezzo sceso di +{max_raggiunto_perc*100:.1f}%. Abbasso lo Stop Win a {pos['sl']:.4f} $!")
                elif max_raggiunto_perc >= 0.035 and not pos.get("breakeven_attivo", False):
                    pos["sl"] = entry * 0.996
                    pos["breakeven_attivo"] = True
                    print(f"🔒 STOP A PAREGGIO ATTIVATO! Stop Loss spostato a {pos['sl']:.4f} $ (Zero Rischio!)")

        # 2. CONTROLLO INTELLIGENTE NOTIZIE + ANDAMENTO (Se l'operazione è ancora aperta)
        if not chiusa:
            # A) Se siamo in guadagno (> +2%) ma escono NOTIZIE MOLTO NEGATIVE o RSI esplode (> 78), incassa subito!
            if pos["tipo"] == "LONG" and var_attuale_perc >= 0.02 and (market_sentiment < -0.10 or rsi_attuale >= 78):
                chiusa = True
                prezzo_uscita = current_price
                esito = "INCASSO INTELLIGENTE (News peggiorate / Ipercomprato) 🧠💵"
            elif pos["tipo"] == "SHORT" and var_attuale_perc >= 0.02 and (market_sentiment > 0.15 or rsi_attuale <= 22):
                chiusa = True
                prezzo_uscita = current_price
                esito = "INCASSO INTELLIGENTE (News cambiate / Ipervenduto) 🧠💵"
            # B) Se il trend grafico su base oraria si inverte del tutto
            elif pos["tipo"] == "LONG" and current_price < sma_20 < sma_50:
                chiusa = True
                prezzo_uscita = current_price
                esito = "USCITA PER INVERSIONE TREND 🔄"
            elif pos["tipo"] == "SHORT" and current_price > sma_20 > sma_50:
                chiusa = True
                prezzo_uscita = current_price
                esito = "USCITA PER INVERSIONE TREND 🔄"
            # C) Scadenza massima 72 ore se resta impantanata
            elif ore_aperte >= MAX_ORE_OPERAZIONE:
                chiusa = True
                prezzo_uscita = current_price
                esito = "CHIUSURA PER SCADENZA 72H ⏰"

        # 3. Esecuzione della Vendita se si è attivata una condizione di uscita
        if chiusa:
            var_perc_lorda = ((prezzo_uscita - entry) / entry) if pos["tipo"] == "LONG" else ((entry - prezzo_uscita) / entry)
            profitto_lordo_euro = pos["capitale_investito"] * var_perc_lorda
            commissioni_euro = pos["capitale_investito"] * FEE_TOTALE_PERC
            profitto_netto_euro = round(profitto_lordo_euro - commissioni_euro, 2)
            
            state["capitale"] = round(state["capitale"] + profitto_netto_euro, 2)
            state["commissioni_pagate"] = round(state.get("commissioni_pagate", 0.0) + commissioni_euro, 2)
            if profitto_netto_euro > 0:
                state["vinte"] = state.get("vinte", 0) + 1
            else:
                state["perse"] = state.get("perse", 0) + 1
                
            print("==============================================================")
            print(f"🔔 VENDITA ESEGUITA SU {ticker} -> {esito}")
            print(f"🪙 Quantità Venduta: {pos['quantita']:.6f} {coin_name} a {prezzo_uscita:.4f} $")
            print(f"🧾 Risultato Lordo: {profitto_lordo_euro:+.2f} € | Commissioni Exchange: -{commissioni_euro:.2f} €")
            print(f"💵 PROFITTO/PERDITA NETTA REALE: {profitto_netto_euro:+.2f} €")
            print(f"🏦 NUOVO CAPITALE NEL PORTAFOGLIO: {state['capitale']:.2f} €")
            
            append_history({
                "Apertura": pos["data_apertura"],
                "Chiusura": now_str,
                "Asset": ticker,
                "Tipo": pos["tipo"],
                "Euro_Investiti": pos["capitale_investito"],
                "Quantita": round(pos["quantita"], 5),
                "Prezzo_Entrata": round(entry, 4),
                "Prezzo_Uscita": round(prezzo_uscita, 4),
                "Motivo_Chiusura": esito,
                "Fee_Euro": round(commissioni_euro, 2),
                "Netto_Euro": profitto_netto_euro,
                "Capitale_Finale": state["capitale"]
            })
            state["posizione_aperta"] = None
            save_state(state)
        else:
            pnl_netto_stimato = (pos["capitale_investito"] * var_attuale_perc) - (pos["capitale_investito"] * FEE_TOTALE_PERC)
            print(f"🛡️ Trend e Notizie favorevoli: mantengo la posizione e lascio correre verso il +15%!")
            print(f"💵 Guadagno/Perdita netta attuale (già tolte le fee): {pnl_netto_stimato:+.2f} €")
            save_state(state)
            
    except Exception as e:
        print(f"⚠️ Errore temporaneo nella lettura dati di {ticker}: {e}")

# =====================================================================
# FASE 2: RICERCA NUOVA PUNTATA (Solo se il capitale è libero)
# =====================================================================
if state["posizione_aperta"] is None:
    print("🔍 Scansione dei 21 asset (Trend + Volumi + RSI + News)...")
    
    best_crypto = None
    best_score = -999.0
    selected_data = {}

    for ticker in UNIVERSE_CRYPTO:
        try:
            df = yf.download(ticker, period="30d", interval="1h", progress=False)
            if df.empty or len(df) < 50:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = calcola_rsi(df['Close'], 14)
            
            current_price = float(df['Close'].iloc[-1])
            sma_20 = float(df['SMA_20'].iloc[-1])
            sma_50 = float(df['SMA_50'].iloc[-1])
            rsi = float(df['RSI'].iloc[-1])
            volume_medio = float(df['Volume'].rolling(window=24).mean().iloc[-1])
            volume_attuale = float(df['Volume'].iloc[-1])
            
            if current_price > sma_20 > sma_50 and 45 <= rsi <= 68:
                trend_score = 1.0
                direzione = "LONG"
            elif current_price < sma_20 < sma_50 and 32 <= rsi <= 55:
                trend_score = 1.0
                direzione = "SHORT"
            else:
                continue
                
            volume_multiplier = 1.3 if volume_attuale > volume_medio else 0.85
            sentiment_align = market_sentiment if direzione == "LONG" else -market_sentiment
            
            total_score = (trend_score * 2.0 * volume_multiplier) + (sentiment_align * 1.5)
            
            if total_score > best_score:
                best_score = total_score
                best_crypto = ticker
                selected_data = {
                    'price': current_price,
                    'trend': direzione,
                    'rsi': rsi,
                    'score': total_score
                }
        except Exception:
            continue

    if best_crypto and best_score >= 1.8:
        capitale_investito = state["capitale"]
        dollari_investiti = round(capitale_investito * eur_usd, 2)
        price = selected_data['price']
        quantita_acquistata = (dollari_investiti * (1 - (FEE_TOTALE_PERC / 2))) / price
        coin_name = best_crypto.replace("-USD", "")
        
        if selected_data['trend'] == 'LONG':
            sl = price * 0.96   # Stop Loss iniziale al -4%
            tp = price * 1.15   # Super Take Profit al +15%
        else:
            sl = price * 1.04
            tp = price * 0.85
            
        state["posizione_aperta"] = {
            "ticker": best_crypto,
            "tipo": selected_data['trend'],
            "capitale_investito": capitale_investito,
            "dollari_investiti": dollari_investiti,
            "quantita": quantita_acquistata,
            "entry_price": price,
            "sl": sl,
            "tp": tp,
            "breakeven_attivo": False,
            "data_apertura": now_str
        }
        save_state(state)
        
        print("==============================================================")
        print(f"🎯 NUOVO ORDINE ESEGUITO (PUNTATA EFFETTUATA)!")
        print(f"⭐ Asset Scelto: {best_crypto} | Direzione: {selected_data['trend']} | Score: {best_score:.2f} | RSI: {selected_data['rsi']:.1f}")
        print(f"💶 Capitale Impegnato: {capitale_investito:.2f} € (convertiti in {dollari_investiti:.2f} $)")
        print(f"🪙 Quantità Acquistata: {quantita_acquistata:.6f} {coin_name} (al prezzo di {price:.4f} $)")
        print(f"🛡️ Stop Loss: {sl:.4f} $ (-4%) | 🎯 Super Take Profit: {tp:.4f} $ (+15%)")
        print(f"🔒 Gestione Dinamica: A +3.5% Stop a pareggio; oltre +6% inseguimento automatico del profitto e controllo News orario!")
    else:
        print(f"⏸️ NESSUNA PUNTATA: Miglior score rilevato ({best_score:.2f}) sotto la soglia 1.80 o RSI fuori zona sicura.")
        print(f"🏦 I tuoi {state['capitale']:.2f} € restano al sicuro in attesa della prossima ora.")
        save_state(state)

# =====================================================================
# FASE 3: STAMPA TABELLA STORICO
# =====================================================================
print("==============================================================")
if os.path.exists(HISTORY_FILE):
    df_hist = pd.read_csv(HISTORY_FILE)
    print(f"📜 LIBRO CONTABILE - ULTIME OPERAZIONI CHIUSE ({len(df_hist)} totali):")
    print(df_hist.tail(10).to_string(index=False))
else:
    print("📜 Libro Contabile: Nessuna operazione ancora conclusa.")
print("==============================================================")
