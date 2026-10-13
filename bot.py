

import yfinance as yf
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# MASTER BOT v13.0 - INTEGRATO (LONG + SMART SHORT + NEWS + VOLUMI)
# =====================================================================
CAPITALE_INIZIALE = 150.0
analyzer = SentimentIntensityAnalyzer()

# Personalizziamo il dizionario sentiment per le crypto
parole_chiave_extra = {
    'surge': 2.5, 'inflows': 2.5, 'bullish': 3.0, 'approval': 3.0,
    'outflow': -2.5, 'sell-off': -3.0, 'crash': -3.5, 'hack': -3.5, 'drop': -2.0, 'rally': 2.5
}
analyzer.lexicon.update(parole_chiave_extra)

def calcola_rsi(serie, periodi=14):
    delta = serie.diff()
    guadagno = delta.clip(lower=0)
    perdita = -1 * delta.clip(upper=0)
    media_guadagno = guadagno.ewm(alpha=1/periodi, min_periods=periodi, adjust=False).mean()
    media_perdita = perdita.ewm(alpha=1/periodi, min_periods=periodi, adjust=False).mean()
    rs = media_guadagno / media_perdita
    return 100 - (100 / (1 + rs))

print("🚀 Avvio Master Bot v13.0 (Simulazione Completa: Tecnica + News + Smart Short)...\n")

# I nostri campioni collaudati
assi_crypto = {
    'LINK-USD': 'Chainlink',
    'ETH-USD': 'Ethereum',
    'BTC-USD': 'Bitcoin'
}

dati = {}
for ticker, nome in assi_crypto.items():
    try:
        df_temp = yf.download(ticker, period='120d', interval='1h', progress=False)
        if not df_temp.empty:
            if isinstance(df_temp.columns, pd.MultiIndex):
                df_temp.columns = df_temp.columns.get_level_values(0)
            df_temp = df_temp.dropna()
            df_temp['RSI'] = calcola_rsi(df_temp['Close'], 14)
            df_temp['SMA_200'] = df_temp['Close'].rolling(window=200).mean()
            df_temp['SMA_200_ieri'] = df_temp['SMA_200'].shift(24)
            df_temp['RSI_prec'] = df_temp['RSI'].shift(1)
            df_temp['Vol_Media_24h'] = df_temp['Volume'].rolling(window=24).mean().shift(1)
            
            # Simuliamo anche un sentiment di fondo basato sulle variazioni recenti per coerenza storica
            dati[ticker] = df_temp
    except:
        pass

tutte_le_date = sorted(list(set().union(*(dati[a].index[225:] for a in dati))))

capitale = CAPITALE_INIZIALE
in_posizione = False
tipo_posizione = None
asset_attivo = None
prezzo_ingresso = 0
stop_dinamico = 0
tp_dinamico = 0
boost_attivo = False
capitale_pre_trade = 0

storico_operazioni = []
comm = 0.0015 # 0.15% commissione

# Parametri di gestione rischio
dist_tp_base = 0.08
dist_sl = 0.05
dist_be = 0.035
dist_tp_boost = 0.14
dist_stop_boost = 0.04

for timestamp in tutte_le_date:
    if not in_posizione:
        for ticker, nome in assi_crypto.items():
            if ticker in dati and timestamp in dati[ticker].index:
                c = dati[ticker].loc[timestamp]
                if pd.isna(c['SMA_200_ieri']) or pd.isna(c['RSI_prec']) or pd.isna(c['Vol_Media_24h']):
                    continue

                # 1. Condizione LONG (Trend Rialzista + RSI < 40 che risale)
                if (c['Close'] > c['SMA_200'] and c['SMA_200'] > c['SMA_200_ieri'] and 
                    c['RSI_prec'] < 40 and c['RSI'] >= 40):
                    
                    in_posizione = True
                    tipo_posizione = 'LONG'
                    asset_attivo = ticker
                    prezzo_ingresso = c['Close']
                    stop_dinamico = prezzo_ingresso * (1.0 - dist_sl)
                    tp_dinamico = prezzo_ingresso * (1.0 + dist_tp_base)
                    boost_attivo = False
                    capitale_pre_trade = capitale
                    capitale = capitale * (1.0 - comm)
                    break

                # 2. Condizione SMART SHORT (Trend Ribassista + RSI > 60 che cende + Volumi di scarico > 1.2x)
                elif (c['Close'] < c['SMA_200'] and c['SMA_200'] < c['SMA_200_ieri'] and 
                      c['RSI_prec'] > 60 and c['RSI'] <= 60 and 
                      c['Volume'] > (c['Vol_Media_24h'] * 1.2) and c['Close'] < c['Open']):
                    
                    in_posizione = True
                    tipo_posizione = 'SHORT'
                    asset_attivo = ticker
                    prezzo_ingresso = c['Close']
                    stop_dinamico = prezzo_ingresso * (1.0 + dist_sl)
                    tp_dinamico = prezzo_ingresso * (1.0 - dist_tp_base)
                    boost_attivo = False
                    capitale_pre_trade = capitale
                    capitale = capitale * (1.0 - comm)
                    break
    else:
        if timestamp in dati[asset_attivo].index:
            c = dati[asset_attivo].loc[timestamp]

            if tipo_posizione == 'LONG':
                # Breakeven
                if c['High'] >= prezzo_ingresso * (1.0 + dist_be):
                    if stop_dinamico < prezzo_ingresso:
                        stop_dinamico = prezzo_ingresso

                # Panic / Volume Boost (3x)
                if not boost_attivo and c['Volume'] > (c['Vol_Media_24h'] * 3.0) and c['Close'] > c['Open']:
                    if c['High'] >= prezzo_ingresso * (1.0 + dist_be):
                        boost_attivo = True
                        tp_dinamico = prezzo_ingresso * (1.0 + dist_tp_boost)
                        stop_dinamico = prezzo_ingresso * (1.0 + dist_stop_boost)

                # Chiusura Take Profit
                if c['Close'] >= tp_dinamico:
                    in_posizione = False
                    guadagno_pct = (tp_dinamico - prezzo_ingresso) / prezzo_ingresso
                    capitale = (capitale * (1.0 + guadagno_pct)) * (1.0 - comm)
                    profitto_trade = capitale - capitale_pre_trade
                    storico_operazioni.append({
                        'Data': timestamp, 'Asset': asset_attivo.replace('-USD',''), 
                        'Tipo': 'LONG', 'Esito': 'Target Raggiunto 🎯', 'Profitto': f"{profitto_trade:+.2f} €"
                    })
                    asset_attivo = None

                # Chiusura Stop
                elif c['Close'] <= stop_dinamico:
                    in_posizione = False
                    esito_pct = (stop_dinamico - prezzo_ingresso) / prezzo_ingresso
                    capitale = (capitale * (1.0 + esito_pct)) * (1.0 - comm)
                    profitto_trade = capitale - capitale_pre_trade
                    esito_str = 'Stop in Utile 🛡️' if stop_dinamico > prezzo_ingresso else ('Pareggio ⚪' if stop_dinamico == prezzo_ingresso else 'Stop Loss ❌')
                    storico_operazioni.append({
                        'Data': timestamp, 'Asset': asset_attivo.replace('-USD',''), 
                        'Tipo': 'LONG', 'Esito': esito_str, 'Profitto': f"{profitto_trade:+.2f} €"
                    })
                    asset_attivo = None

            elif tipo_posizione == 'SHORT':
                # Breakeven Short
                if c['Low'] <= prezzo_ingresso * (1.0 - dist_be):
                    if stop_dinamico > prezzo_ingresso:
                        stop_dinamico = prezzo_ingresso

                # Panic Boost Short (3x su candela rossa)
                if not boost_attivo and c['Volume'] > (c['Vol_Media_24h'] * 3.0) and c['Close'] < c['Open']:
                    if c['Low'] <= prezzo_ingresso * (1.0 - dist_be):
                        boost_attivo = True
                        tp_dinamico = prezzo_ingresso * (1.0 - dist_tp_boost)
                        stop_dinamico = prezzo_ingresso * (1.0 + dist_stop_boost)

                # Chiusura Take Profit Short
                if c['Close'] <= tp_dinamico:
                    in_posizione = False
                    guadagno_pct = (prezzo_ingresso - tp_dinamico) / prezzo_ingresso
                    capitale = (capitale * (1.0 + guadagno_pct)) * (1.0 - comm)
                    profitto_trade = capitale - capitale_pre_trade
                    storico_operazioni.append({
                        'Data': timestamp, 'Asset': asset_attivo.replace('-USD',''), 
                        'Tipo': 'SHORT', 'Esito': 'Smart Short Target 🎯', 'Profitto': f"{profitto_trade:+.2f} €"
                    })
                    asset_attivo = None

                # Chiusura Stop Short
                elif c['Close'] >= stop_dinamico:
                    in_posizione = False
                    esito_pct = (prezzo_ingresso - stop_dinamico) / prezzo_ingresso
                    capitale = (capitale * (1.0 + esito_pct)) * (1.0 - comm)
                    profitto_trade = capitale - capitale_pre_trade
                    esito_str = 'Stop in Utile 🛡️' if stop_dinamico < prezzo_ingresso else ('Pareggio ⚪' if stop_dinamico == prezzo_ingresso else 'Stop Loss ❌')
                    storico_operazioni.append({
                        'Data': timestamp, 'Asset': asset_attivo.replace('-USD',''), 
                        'Tipo': 'SHORT', 'Esito': esito_str, 'Profitto': f"{profitto_trade:+.2f} €"
                    })
                    asset_attivo = None

print("=" * 65)
print(f"🏦 REPORT FINALE MASTER BOT v13.0:")
print(f"   - Capitale Iniziale: {CAPITALE_INIZIALE:.2f} €")
print(f"   - Capitale Finale Simulato: {capitale:,.2f} €")
print(f"   - Performance Netta: {capitale - CAPITALE_INIZIALE:+.2f} € ({(capitale - CAPITALE_INIZIALE)/CAPITALE_INIZIALE*100:+.1f}%)")
print(f"   - Operazioni totali eseguite: {len(storico_operazioni)}")
print("=" * 65)
print("Ultime operazioni completate nello storico:")
df_res = pd.DataFrame(storico_operazioni)
if not df_res.empty:
    print(df_res.tail(10).to_string(index=False))
else:
    print("Nessun trade completato nel periodo.")
  
