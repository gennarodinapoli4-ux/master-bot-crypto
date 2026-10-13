import yfinance as yf
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
import datetime

warnings.filterwarnings('ignore')

# =====================================================================
# MASTER BOT v14.2 - DYNAMIC QUANT BOT (Clean Universe + Risk + Short/Long)
# =====================================================================

CAPITALE_INIZIALE = 150.0
analyzer = SentimentIntensityAnalyzer()

# Paniere pulito e stabile delle principali criptovalute
UNIVERSE_CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", 
    "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD", 
    "UNI-USD", "NEAR-USD", "ATOM-USD", "LTC-USD", "ETC-USD", 
    "XLM-USD", "ALGO-USD", "BCH-USD", "ICP-USD", "FIL-USD", 
    "APT-USD", "ARB-USD", "FTM-USD", "HBAR-USD"
]

print(f"🚀 Avvio Master Bot v14.2 (Analisi Dinamica su {len(UNIVERSE_CRYPTO)} Asset)...")
print("==============================================================")

def get_market_sentiment():
    """Analizza il sentiment globale delle notizie crypto tramite RSS e VADER"""
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
    
    if not sentiments:
        return 0.0
    return sum(sentiments) / len(sentiments)

# 1. Rilevamento Sentiment Globale
market_sentiment = get_market_sentiment()
print(f"🌐 Sentiment di Mercato Globale: {market_sentiment:.3f}")

# 2. Scansione di mercato e scelta autonoma dell'asset migliore
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

        # Indicatori tecnici di base
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        current_price = float(df['Close'].iloc[-1])
        sma_20 = float(df['SMA_20'].iloc[-1])
        sma_50 = float(df['SMA_50'].iloc[-1])
        volume_medio = float(df['Volume'].rolling(window=24).mean().iloc[-1])
        volume_attuale = float(df['Volume'].iloc[-1])
        
        # Regole di punteggio per la scelta autonoma
        trend_score = 1.0 if current_price > sma_20 > sma_50 else (-1.0 if current_price < sma_20 < sma_50 else 0.0)
        volume_multiplier = 1.2 if volume_attuale > volume_medio else 0.8
        
        # Punteggio combinato (Tecnica + Sentiment)
        total_score = (trend_score * 2.0) + (market_sentiment * 1.5) * volume_multiplier
        
        if total_score > best_score:
            best_score = total_score
            best_crypto = ticker
            selected_data = {
                'price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'trend': 'LONG' if current_price > sma_20 else 'SHORT'
            }
    except Exception:
        continue

print("==============================================================")
print(f"🔍 SCELTA AUTONOMA DEL BOT:")
if best_crypto:
    print(f"⭐ Asset Selezionato: {best_crypto}")
    print(f"📊 Prezzo Rilevato: {selected_data['price']:.4f} $")
    print(f"📈 Direzione Strategia: {selected_data['trend']}")
    print(f"⚖️ Score di Mercato: {best_score:.2f}")

    # Applicazione delle regole di gestione del rischio e simulazione operatività
    print("--------------------------------------------------------------")
    print("🛡️ APPLICAZIONE REGOLE DI RISCHIO E STOP LOSS:")
    
    capitale_disponibile = CAPITALE_INIZIALE
    rischio_operazione = capitale_disponibile * 0.04 
    
    if selected_data['trend'] == 'LONG':
        stop_loss_prezzo = selected_data['price'] * 0.96
        take_profit_prezzo = selected_data['price'] * 1.08
    else:
        stop_loss_prezzo = selected_data['price'] * 1.04
        take_profit_prezzo = selected_data['price'] * 0.92

    print(f"   - Capitale Gestito: {capitale_disponibile:.2f} €")
    print(f"   - Stop Loss Fissato a: {stop_loss_prezzo:.4f} $")
    print(f"   - Take Profit Fissato a: {take_profit_prezzo:.4f} $")
    print(f"   - Rischio Massimo Consentito: {rischio_operazione:.2f} €")

else:
    print("⚠️ Nessun asset ha soddisfatto i parametri di volatilità in questa ora.")

print("==============================================================")
print("✅ Esecuzione oraria con regole complete completata nel Cloud.")
