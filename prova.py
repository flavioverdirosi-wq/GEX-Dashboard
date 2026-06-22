"""
VERSIONE 4.0 - EOGA GEX/DEX ENGINE (PREDICTIVE ORDER FLOW)
-------------------------------------------------------------------------
"""

import os
import calendar
import datetime
import pytz
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as si
import streamlit as st
import yfinance as yf
import requests
from scipy.signal import find_peaks, savgol_filter
from streamlit_autorefresh import st_autorefresh

# =====================================================================
# CONFIGURAZIONE PAGINA E STATO
# =====================================================================
st.set_page_config(page_title="EOGA GEX/DEX Engine", layout="wide")

if "memoria_dati" not in st.session_state:
    st.session_state.memoria_dati = {}

# =====================================================================
# --- V 4.0: DATABASE LOCALE PERSISTENTE (GEX, VELOCITY E DRIFT) ---
# =====================================================================
NOME_FILE_DB = "storico_heatmap_intraday.csv"
tz_roma_db = pytz.timezone('Europe/Rome')
data_oggi_str = datetime.datetime.now(tz_roma_db).strftime("%Y-%m-%d")

if "memoria_heatmap" not in st.session_state:
    if os.path.exists(NOME_FILE_DB):
        df_salvato = pd.read_csv(NOME_FILE_DB)
        if not df_salvato.empty and 'Data_Rilevamento' in df_salvato.columns and df_salvato['Data_Rilevamento'].iloc[0] == data_oggi_str:
            st.session_state.memoria_heatmap = df_salvato
        else:
            st.session_state.memoria_heatmap = pd.DataFrame(columns=['Data_Rilevamento', 'Orario', 'Strike_ETF', 'GEX_Netto', 'DEX_Netto', 'Drift_C', 'Drift_P'])
    else:
        st.session_state.memoria_heatmap = pd.DataFrame(columns=['Data_Rilevamento', 'Orario', 'Strike_ETF', 'GEX_Netto', 'DEX_Netto', 'Drift_C', 'Drift_P'])

# =====================================================================
# FUNZIONI MATEMATICHE E CALENDARIO
# =====================================================================
@st.cache_data(ttl=60)
def get_vix_data():
    try:
        vix = yf.Ticker("^VIX").fast_info['last_price']
        vxn = yf.Ticker("^VXN").fast_info['last_price']
        return vix, vxn
    except:
        return 15.0, 15.0

def trova_prossimo_opex_mensile():
    oggi = datetime.date.today()
    mese, anno = oggi.month, oggi.year
    c = calendar.monthcalendar(anno, mese)
    venerdi_mese = [settimana[4] for settimana in c if settimana[4] != 0]
    terzo_venerdi = datetime.date(anno, mese, venerdi_mese[2])
    
    if oggi > terzo_venerdi:
        mese += 1
        if mese > 12: mese, anno = 1, anno + 1
        c = calendar.monthcalendar(anno, mese)
        venerdi_mese = [settimana[4] for settimana in c if settimana[4] != 0]
        terzo_venerdi = datetime.date(anno, mese, venerdi_mese[2])
    return str(terzo_venerdi)

def calcola_greche_base(S, K, t, sigma, r=0.045):
    if sigma <= 0 or t <= 0 or S <= 0 or K <= 0: return 0, 0, 0, 0, 0, 0
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    n_prime_d1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d1**2))
    
    delta_call = si.norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = n_prime_d1 / (S * sigma * np.sqrt(t))
    
    # Calcolo Strutturale di Vanna e Charm
    vanna = -n_prime_d1 * d2 / sigma
    charm_call = -n_prime_d1 * (r / (sigma * np.sqrt(t)) - d2 / (2 * t))
    charm_put = charm_call + r
    
    return delta_call, delta_put, gamma, vanna, charm_call, charm_put

def pulisci_numero(val):
    if pd.isna(val) or val == '--' or val == '': return 0.0
    try:
        val_str = str(val).replace(',', '').replace('+', '').replace('▲', '').replace('▼', '').strip()
        return float(val_str)
    except:
        return 0.0

def verifica_stato_mercato():
    tz = pytz.timezone('Europe/Rome')
    now = datetime.datetime.now(tz)
    if now.weekday() >= 5: return "CHIUSO 🔴"
    if datetime.time(15, 30) <= now.time() <= datetime.time(22, 0): return "APERTO 🟢"
    return "CHIUSO 🔴"

def calcola_zone_flipper(df, colonna_strike='Strike_ETF', colonna_gex='GEX', usa_filtro=True):
    df_zone = df.sort_values(by=colonna_strike).reset_index(drop=True)
    if len(df_zone) < 3:
        df_zone['Zona_Tattica'] = '⚪ Neutro'
        return df_zone

    if usa_filtro:
        window = 5 if len(df_zone) >= 5 else 3
        df_zone['GEX_Smooth'] = savgol_filter(df_zone[colonna_gex], window_length=window, polyorder=2)
        max_gex_assoluto = df_zone['GEX_Smooth'].abs().max()
        soglia_minima = max_gex_assoluto * 0.05 
        distanza_picchi = 2 
    else:
        df_zone['GEX_Smooth'] = df_zone[colonna_gex]
        max_gex_assoluto = df_zone['GEX_Smooth'].abs().max()
        soglia_minima = max_gex_assoluto * 0.01
        distanza_picchi = 1

    muri_idx, _ = find_peaks(df_zone['GEX_Smooth'], distance=distanza_picchi, prominence=soglia_minima)
    tasche_idx, _ = find_peaks(-df_zone['GEX_Smooth'], distance=distanza_picchi, prominence=soglia_minima)
    
    df_zone['Zona_Tattica'] = '⚪ Neutro'
    df_zone.loc[muri_idx, 'Zona_Tattica'] = '🟢 Muro (Assorbimento)'
    df_zone.loc[tasche_idx, 'Zona_Tattica'] = '🔴 Tasca (Accelerazione)'
    return df_zone

# =====================================================================
# MOTORE DI ESTRAZIONE DATI E GESTIONE CACHE
# =====================================================================
@st.cache_resource
def inizializza_ticker(symbol):
    return yf.Ticker(symbol)

@st.cache_data(ttl=10)
def scarica_prezzo_spot(symbol):
    try:
        tk_obj = yf.Ticker(symbol)
        storico = tk_obj.history(period="1d", interval="1m", prepost=True)
        if storico.empty: 
            storico_giornaliero = tk_obj.history(period="1d")
            return storico_giornaliero["Close"].iloc[-1]
        return storico["Close"].iloc[-1]
    except Exception:
        return None

@st.cache_data(ttl=10)
def scarica_quote_nasdaq(ticker):
    url = f"https://api.nasdaq.com/api/quote/{ticker}/info?assetclass=etf"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/"
    }
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json().get("data", {})
            if data:
                ext_data = data.get("extendedMarketData", {})
                prim_data = data.get("primaryData", {})
                
                ext_price = ext_data.get("lastSalePrice", "") if ext_data else ""
                if ext_price: return float(ext_price.replace("$", "").replace(",", ""))
                
                prim_price = prim_data.get("lastSalePrice", "") if prim_data else ""
                if prim_price: return float(prim_price.replace("$", "").replace(",", ""))
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def scarica_chain_nasdaq_pura(ticker, data_scadenza, asset_class="etf"):
    url = f"https://api.nasdaq.com/api/quote/{ticker}/option-chain?assetclass={asset_class}&limit=5000&date={data_scadenza}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/"
    }
    try:
        risposta = requests.get(url, headers=headers, timeout=10)
        if risposta.status_code == 200:
            dati = risposta.json()
            righe = dati.get('data', {}).get('table', {}).get('rows', [])
            if righe:
                df = pd.DataFrame(righe)
                colonne_da_pulire = [
                    'strike', 'c_Last', 'c_Change', 'c_Bid', 'c_Ask', 'c_Volume', 'c_Openinterest',
                    'p_Last', 'p_Change', 'p_Bid', 'p_Ask', 'p_Volume', 'p_Openinterest'
                ]
                for col in colonne_da_pulire:
                    if col in df.columns:
                        df[col] = df[col].apply(pulisci_numero)
                return df
    except Exception as e:
        st.error(f"Errore API Nasdaq: {e}")
    return pd.DataFrame()

def ottieni_dati_intelligenti(ticker, scadenza):
    df_nuova = scarica_chain_nasdaq_pura(ticker, scadenza)
    chiave = f"{ticker}_{scadenza}"
    ora_attuale = pd.Timestamp.now(tz='Europe/Rome').strftime('%H:%M:%S')
    
    if chiave not in st.session_state.memoria_dati:
        if not df_nuova.empty:
            st.session_state.memoria_dati[chiave] = {"df": df_nuova.copy(), "ora": ora_attuale}
            return st.session_state.memoria_dati[chiave]
        return {"df": pd.DataFrame(), "ora": "Mai"}
        
    df_vecchia = st.session_state.memoria_dati[chiave]["df"]
    
    if not df_nuova.empty:
        if not df_nuova.equals(df_vecchia):
            st.session_state.memoria_dati[chiave] = {"df": df_nuova.copy(), "ora": ora_attuale}
            
    return st.session_state.memoria_dati[chiave]

# =====================================================================
# SIDEBAR E SINCRONIZZAZIONE
# =====================================================================
st.sidebar.title("🧭 Navigazione App")
pagina = st.sidebar.radio("Seleziona la vista:", ["📊 Dashboard Grafica (GEX)", "🗄️ Database Ufficiale Nasdaq"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Sincronizzazione Dati")

col_t1, col_t2 = st.sidebar.columns([3, 1])
with col_t1:
    ticker_input = st.text_input("Ticker Sottostante (es. QQQ, SPY)", "QQQ").upper()
with col_t2:
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    btn_vai = st.button("Vai 🚀", use_container_width=True)

if "ticker_attivo" not in st.session_state:
    st.session_state.ticker_attivo = "QQQ"

if btn_vai:
    st.session_state.ticker_attivo = ticker_input
    st.cache_data.clear()

ticker = st.session_state.ticker_attivo

MAPPATURA_FUTURES = {
    "QQQ": {"simbolo": "NQU26.CME", "nome": "NQ"},
    "SPY": {"simbolo": "ESU26.CME", "nome": "ES"},
    "IWM": {"simbolo": "RTYU26.CME", "nome": "RTY"},
    "DIA": {"simbolo": "YMU26.CME", "nome": "YM"}
}
info_future = MAPPATURA_FUTURES.get(ticker, {"simbolo": ticker, "nome": ticker})
ticker_future = info_future["simbolo"]
nome_future = info_future["nome"]

tk = inizializza_ticker(ticker)
spot_price_reale = scarica_prezzo_spot(ticker)

if spot_price_reale is None:
    st.sidebar.error(f"Impossibile scaricare il prezzo per {ticker}.")
    st.stop()

try:
    storico_daily = tk.history(period="5d", auto_adjust=False)
    prezzo_chiusura_api = storico_daily['Close'].iloc[-2] if "APERTO" in verifica_stato_mercato() else storico_daily['Close'].iloc[-1]
except:
    prezzo_chiusura_api = spot_price_reale

usa_chiusura_manuale = st.sidebar.checkbox(f"Modifica Chiusura {ticker} a mano", value=False)
if usa_chiusura_manuale:
    prezzo_chiusura = st.sidebar.number_input(f"Prezzo {ticker} (Chiusura)", value=float(prezzo_chiusura_api), step=0.10)
else:
    prezzo_chiusura = prezzo_chiusura_api
    st.sidebar.text_input(f"Prezzo {ticker} (Chiusura)", value=f"{prezzo_chiusura:.2f} $", disabled=True)

tk_future = yf.Ticker(ticker_future)
future_spot_reale = scarica_prezzo_spot(ticker_future)
try:
    storico_fut_1m = tk_future.history(period="5d", interval="1m", prepost=True)
    if not storico_fut_1m.empty:
        storico_fut_1m.index = storico_fut_1m.index.tz_convert('Europe/Rome')
        candele_pre_chiusura = storico_fut_1m[(storico_fut_1m.index.hour == 21) & (storico_fut_1m.index.minute >= 50)]
        prezzo_fut_api = candele_pre_chiusura.groupby(candele_pre_chiusura.index.date).last()['Close'].iloc[-1] if not candele_pre_chiusura.empty else future_spot_reale
    else:
        prezzo_fut_api = future_spot_reale
except:
    prezzo_fut_api = future_spot_reale if future_spot_reale is not None else spot_price_reale

usa_fut_manuale = st.sidebar.checkbox(f"Modifica Chiusura {nome_future} a mano", value=False)
if usa_fut_manuale:
    prezzo_future_man = st.sidebar.number_input(f"Prezzo Future {nome_future}", value=float(prezzo_fut_api), step=10.0)
else:
    prezzo_future_man = prezzo_fut_api
    st.sidebar.text_input(f"Prezzo Future {nome_future}", value=f"{prezzo_future_man:.2f}", disabled=True)

ratio_esatto = prezzo_future_man / prezzo_chiusura if prezzo_chiusura > 0 else 1.0
st.sidebar.metric("Moltiplicatore Calcolato", value=f"{ratio_esatto:.4f}x")

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Controllo Dati")
col_r1, col_r2 = st.sidebar.columns([3, 1])
with col_r1:
    auto_refresh = st.checkbox("⏱️ Auto-Refresh (1 min)", value=True)
with col_r2:
    if st.button("Forza", use_container_width=True):
        st.cache_data.clear() 
        st.rerun()

if auto_refresh:
    st_autorefresh(interval=60000, key="data_refresh_1m")

stato_mercato = verifica_stato_mercato()

# Variabili Globali per il Live Spot 
etf_realtime_nasdaq = scarica_quote_nasdaq(ticker)
if etf_realtime_nasdaq is None: etf_realtime_nasdaq = spot_price_reale 

future_realtime_yf = scarica_prezzo_spot(ticker_future)
if future_realtime_yf is None: future_realtime_yf = prezzo_future_man

scadenze_disponibili = list(tk.options)
if not scadenze_disponibili:
    st.error(f"Nessuna data trovata per {ticker}.")
    st.stop()


# =====================================================================
# PAGINA 1: DASHBOARD GRAFICA
# =====================================================================
if pagina == "📊 Dashboard Grafica (GEX)":
    
    # --- MOTORE ASINCRONO: TOP BAR ---
    @st.fragment(run_every=10)
    def renderizza_top_bar_live():
        live_etf = scarica_quote_nasdaq(ticker)
        if live_etf is None: live_etf = spot_price_reale
        
        live_fut = scarica_prezzo_spot(ticker_future) or prezzo_future_man
        
        colore_stato = "#00E676" if "APERTO" in stato_mercato else "#FF3B30"
        stato_testo = stato_mercato.replace('🟢', '').replace('🔴', '').strip()
        
        if prezzo_chiusura > 0:
            var_pct_etf = ((live_etf - prezzo_chiusura) / prezzo_chiusura) * 100
        else:
            var_pct_etf = 0.0
            
        colore_var = "#00E676" if var_pct_etf >= 0 else "#FF3B30"
        segno_var = "+" if var_pct_etf >= 0 else ""
        testo_var = f"{segno_var}{var_pct_etf:.2f}%"

        tz_it_live = pytz.timezone('Europe/Rome')
        ora_it_live = datetime.datetime.now(tz_it_live).strftime("%d %b %Y - %H:%M:%S IT")
        tz_est_live = pytz.timezone('America/New_York')
        ora_est_live = datetime.datetime.now(tz_est_live).strftime("%d %b %Y - %H:%M:%S NY")

        top_bar_html = f"""<div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(145deg, #1A1D24 0%, #131722 100%); padding: 20px 30px; border-radius: 12px; border: 1px solid #2B3139; box-shadow: 0px 8px 20px rgba(0,0,0,0.4); margin-bottom: 25px;">
        <div style="flex: 1;">
            <h1 style="margin: 0; color: #E0E3EB; font-size: 28px; font-family: 'Arial Black', sans-serif; text-transform: uppercase; letter-spacing: 1.5px;">🎯 EOGA <span style="color: #FFD700;">GEX</span></h1>
            <span style="color: #8C92A4; font-size: 13px; text-transform: uppercase; letter-spacing: 2px;">Advanced Order Book</span>
        </div>
        <div style="display: flex; gap: 35px; align-items: center; justify-content: center; flex: 2.5;">
            <div style="text-align: right;">
                <span style="color: #8C92A4; font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;">{ticker} (Spot)</span><br>
                <span style="color: #FFFFFF; font-size: 34px; font-weight: 900; font-family: 'Courier New', monospace; text-shadow: 0 0 10px rgba(255,255,255,0.1);">${live_etf:,.2f}</span>
            </div>
            <div style="height: 50px; width: 2px; background: linear-gradient(to bottom, transparent, #3A414D, transparent);"></div>
            <div style="text-align: left;">
                <span style="color: #8C92A4; font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;">{nome_future} (Future)</span><br>
                <span style="color: #FFFFFF; font-size: 34px; font-weight: 900; font-family: 'Courier New', monospace; text-shadow: 0 0 10px rgba(255,255,255,0.1);">{live_fut:,.2f}</span>
            </div>
            <div style="height: 50px; width: 2px; background: linear-gradient(to bottom, transparent, #3A414D, transparent);"></div>
            <div style="text-align: left;">
                <span style="color: #8C92A4; font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;">{ticker} Chg %</span><br>
                <span style="color: {colore_var}; font-size: 34px; font-weight: 900; font-family: 'Courier New', monospace; text-shadow: 0 0 10px {colore_var}20;">{testo_var}</span>
            </div>
        </div>
        <div style="flex: 1; text-align: right;">
            <div style="display: inline-flex; align-items: center; gap: 10px; background-color: rgba(255,255,255,0.03); padding: 8px 15px; border-radius: 30px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 4px;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {colore_stato}; box-shadow: 0 0 8px {colore_stato};"></div>
                <span style="color: #E0E3EB; font-weight: bold; font-size: 14px; letter-spacing: 1px;">{stato_testo}</span>
            </div>
            <div style="color: #8C92A4; font-size: 12px; font-family: monospace; line-height: 1.3;">{ora_it_live}</div>
            <div style="color: #53B9EA; font-size: 12px; font-family: monospace; font-weight: bold; line-height: 1.3;">{ora_est_live}</div>
        </div>
    </div>"""
        st.markdown(top_bar_html, unsafe_allow_html=True)

    renderizza_top_bar_live()

    # --- FILTRI E OPZIONI ---
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        usa_total_gex = st.checkbox("🌐 Calcola Total GEX (Aggregato)", value=False)
        scadenza_sel = st.selectbox("Scadenza Singola:", scadenze_disponibili, disabled=usa_total_gex)
    with col_f2:
        filtro_percentuale = st.slider("Zoom Grafico (+/- % dal prezzo)", min_value=1, max_value=20, value=1)
    with col_f3:
        tipo_visualizzazione = st.radio("Visualizza Istogramma:", ["GEX (Gamma)", "DEX (Delta)"], horizontal=True)
    with col_f4:
        st.markdown("<br>", unsafe_allow_html=True)
        mostra_etf = st.checkbox(f"🔄 Mostra livelli in {ticker}", value=False)

    # --- PANNELLO INFORMATIVO E TOGGLE MOTORE HVL ---
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 Guida Strategica: Il Mercato Invisibile (Luca Giusti)"):
        st.markdown("""
        La meccanica di mercato è guidata dall'hedging dei Market Maker per restare neutrali al rischio. 
        [Approfondimento Teoria: Il Mercato Invisibile](https://www.lucagiusti.it/2026/04/29/il-mercato-invisibile-gex-e-dex/)
        """)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            **🟢 CALL WALL (Resistenza):** Soffitto meccanico. I Dealer vendono sui rialzi.
            * *Tattica:* Resistenza forte. Valuta Take-Profit sui Long o ingressi Short.
            
            **🟡 HVL (FLIP POINT):** Lo "Zero Gamma". Separa il regime stabilizzatore da quello amplificatore.
            * *Tattica:* Sotto=Volatilità alta (Short Gamma) Ogni movimento direzionale viene amplificato dal flusso di coperture dei Market Maker. Qui il "pavimento" (Put Wall) diventa l'unico livello di difesa prima di potenziali crash direzionali.
            Sopra=Trend direzionale (Long Gamma) Il mercato è "intrappolato" in una morsa. La volatilità viene compressa e i movimenti diventano più prevedibili. Il Call Wall agisce come un magnete che attrae il prezzo.
            """)
        with col_t2:
            st.markdown("""
            **🔴 PUT WALL (Supporto):** Pavimento meccanico. I Dealer comprano sui ribassi.
            * *Tattica:* Supporto forte. Area ideale per Buy-the-Dip o chiusura Short.
            
            **⚖️ P/C RATIO (OI):** Bilancia del sentiment.
            * *Tattica:* >1.2 Pessimismo (rischio squeeze); <0.8 Ottimismo (ipercomprato).
            """)

        st.info("""
        **REGIMI GEX:** - **Positivo (Sopra HVL):** I Dealer stabilizzano. "Compra basso, vende alto" -> **Volatilità compressa**.
        - **Negativo (Sotto HVL):** I Dealer amplificano. "Vende basso, compra alto" -> **Volatilità esplosiva**.
        """)
        
    with st.expander("🔬 Info Motore HVL (Gamma Flip) & Attivazione Metodo Istituzionale"):
        st.markdown("""
        **1. Metodo Base (Approssimazione Cumulativa):** È un calcolo "statico". Stima il baricentro dell'Order Book facendo una somma cumulativa dell'esposizione GEX limitatamente al prezzo *attuale*. È rapido per il processore, ma ignora la convessità delle opzioni se il prezzo dovesse spostarsi.
        
        **2. Metodo Istituzionale (Simulazione Dinamica Vettoriale):** Crea una matrice di 200 scenari di prezzo simulati da -10% a +10%. Attraverso il *broadcasting* di NumPy, ricalcola simultaneamente l'equazione di Black-Scholes per *tutte* le migliaia di opzioni a mercato ad ogni tick simulato. Identifica il vero *Zero Gamma Level* cercando l'esatto punto in cui la curva dinamica incrocia lo zero:
        Total GEX(S) = Somma( OI_Call * Gamma_Call(S) - OI_Put * Gamma_Put(S) ) * S^2 * 0.01 = 0
        """)
        usa_hvl_istituzionale = st.checkbox("⚙️ Attiva Calcolo HVL Istituzionale (Richiede maggiore potenza di calcolo)", value=False)

    vix_live, vxn_live = get_vix_data()
    iv_stimata = vix_live / 100.0
    st.info(f"⚙️ Motore Greche alimentato da VIX Live: {vix_live:.2f}% (Rif. VXN: {vxn_live:.2f}%)")
        
    oggi = datetime.date.today()
    struttura = []
    
    # --- ESTRAZIONE DATI ---
    scadenze_da_analizzare = scadenze_disponibili[:12] if usa_total_gex else [scadenza_sel]
    if usa_total_gex: st.warning(f"⏳ Elaborazione Total GEX in corso ({len(scadenze_da_analizzare)} scadenze)...")

    for scad in scadenze_da_analizzare:
        dati_estratti = ottieni_dati_intelligenti(ticker, scad)
        df_chain_temp = dati_estratti["df"]
        if df_chain_temp.empty: continue
            
        data_scad = datetime.datetime.strptime(scad, "%Y-%m-%d").date()
        giorni = (data_scad - oggi).days
        t_anno = (0.5 / 365.0) if giorni <= 0 else (giorni / 365.0)

        for _, riga in df_chain_temp.iterrows():
            K = riga["strike"]
            oi_call = riga.get("c_Openinterest", 0)
            oi_put = riga.get("p_Openinterest", 0)
            vol_call = riga.get("c_Volume", 0)
            vol_put = riga.get("p_Volume", 0)
            
            if oi_call > 0 or vol_call > 0:
                d_c, _, gamma_c, vanna_c, charm_c, _ = calcola_greche_base(spot_price_reale, K, t_anno, iv_stimata)
                struttura.append({
                    "Strike_ETF": K, "Strike_Future": K * ratio_esatto, 
                    "GEX": gamma_c * oi_call * 100 * (spot_price_reale**2) * 0.01, 
                    "DEX": d_c * oi_call * 100 * spot_price_reale * 0.01,
                    "Vanna": vanna_c * oi_call * 100 * spot_price_reale,
                    "Charm": charm_c * oi_call * 100 * spot_price_reale,
                    "Drift_C": d_c * vol_call * 100 * spot_price_reale * 0.01,
                    "Drift_P": 0.0,
                    "Call_OI": oi_call, "Put_OI": 0,
                    "t_anno": t_anno 
                })
                
            if oi_put > 0 or vol_put > 0:
                _, d_p, gamma_p, vanna_p, _, charm_p = calcola_greche_base(spot_price_reale, K, t_anno, iv_stimata)
                struttura.append({
                    "Strike_ETF": K, "Strike_Future": K * ratio_esatto, 
                    "GEX": -gamma_p * oi_put * 100 * (spot_price_reale**2) * 0.01, 
                    "DEX": d_p * oi_put * 100 * spot_price_reale * 0.01,
                    "Vanna": -vanna_p * oi_put * 100 * spot_price_reale,
                    "Charm": -charm_p * oi_put * 100 * spot_price_reale,
                    "Drift_C": 0.0,
                    "Drift_P": abs(d_p) * vol_put * 100 * spot_price_reale * 0.01,
                    "Call_OI": 0, "Put_OI": oi_put,
                    "t_anno": t_anno 
                })

    df_raw = pd.DataFrame(struttura)
    if df_raw.empty: 
        st.error("Nessun dato estrapolato.")
        st.stop()

    df = df_raw.groupby(["Strike_ETF", "Strike_Future"]).sum().reset_index()

    # --- SALVATAGGIO DISCO ---
    tz_roma = pytz.timezone('Europe/Rome')
    ora_corrente = datetime.datetime.now(tz_roma)
    orario_snapshot = ora_corrente.strftime("%H:%M")
    data_snapshot = ora_corrente.strftime("%Y-%m-%d")

    df_foto = df[['Strike_ETF', 'GEX', 'DEX', 'Drift_C', 'Drift_P']].copy()
    df_foto['Data_Rilevamento'] = data_snapshot
    df_foto['Orario'] = orario_snapshot
    df_foto.rename(columns={'GEX': 'GEX_Netto', 'DEX': 'DEX_Netto'}, inplace=True)

    if st.session_state.memoria_heatmap.empty or orario_snapshot not in st.session_state.memoria_heatmap['Orario'].values:
        st.session_state.memoria_heatmap = pd.concat([st.session_state.memoria_heatmap, df_foto], ignore_index=True)
        st.session_state.memoria_heatmap.to_csv(NOME_FILE_DB, index=False)

    # --- LOGICA VISUALIZZAZIONE (CORREZIONE GRAFICO SQUASHED) ---
    colonna_y = "Strike_ETF" if mostra_etf else "Strike_Future"
    spot_riferimento = etf_realtime_nasdaq if mostra_etf else future_realtime_yf
    nome_asset = ticker if mostra_etf else nome_future

    limite_inf = spot_riferimento * (1 - (filtro_percentuale / 100.0))
    limite_sup = spot_riferimento * (1 + (filtro_percentuale / 100.0))

    df_utile = df[(df[colonna_y] >= limite_inf) & (df[colonna_y] <= limite_sup)].copy()
    if df_utile.empty: df_utile = df.copy()

    df_utile = df_utile.sort_values(colonna_y).reset_index(drop=True)

    call_wall = df.loc[df["GEX"].idxmax()][colonna_y]
    put_wall = df.loc[df["GEX"].idxmin()][colonna_y]

    if usa_hvl_istituzionale:
        spot_simulati = np.linspace(spot_riferimento * 0.9, spot_riferimento * 1.1, 200)
        strikes = df_raw[colonna_y].values
        oi_calls = df_raw["Call_OI"].values
        oi_puts = df_raw["Put_OI"].values
        t_annos = df_raw["t_anno"].values
        
        S = spot_simulati[:, np.newaxis]
        K = strikes[np.newaxis, :]
        T = t_annos[np.newaxis, :]
        
        sigma_sqrt_t = iv_stimata * np.sqrt(T)
        sigma_sqrt_t = np.where(sigma_sqrt_t == 0, 1e-9, sigma_sqrt_t) 
        
        d1 = (np.log(S / K) + (0.045 + (iv_stimata**2) / 2) * T) / sigma_sqrt_t
        n_prime_d1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d1**2))
        gamma_matrice = n_prime_d1 / (S * sigma_sqrt_t)
        
        gex_totale_simulato = np.sum(gamma_matrice * oi_calls * (S**2) - gamma_matrice * oi_puts * (S**2), axis=1)
        
        cambi_segno = np.where(np.diff(np.sign(gex_totale_simulato)))[0]
        if len(cambi_segno) > 0:
            idx = cambi_segno[0]
            s1, s2 = spot_simulati[idx], spot_simulati[idx+1]
            g1, g2 = gex_totale_simulato[idx], gex_totale_simulato[idx+1]
            gamma_flip = s1 - g1 * ((s2 - s1) / (g2 - g1)) 
        else:
            gamma_flip = spot_simulati[np.argmin(np.abs(gex_totale_simulato))]
    else:
        df_utile["GEX_Cum"] = df_utile["GEX"].cumsum()
        idx_flip = np.where(np.diff(np.sign(df_utile["GEX_Cum"])) != 0)[0]
        
        if len(idx_flip) > 0:
            indice_sotto = idx_flip[0]
            indice_sopra = indice_sotto + 1
            if indice_sopra < len(df_utile):
                gamma_flip = (df_utile.iloc[indice_sotto][colonna_y] + df_utile.iloc[indice_sopra][colonna_y]) / 2.0
            else:
                gamma_flip = df_utile.iloc[indice_sotto][colonna_y]
        else:
            gamma_flip = df_utile.loc[df_utile["GEX_Cum"].abs().idxmin()][colonna_y]

    metric_col = "GEX" if tipo_visualizzazione == "GEX (Gamma)" else "DEX"
    df_utile["Colore"] = np.where(df_utile[metric_col] >= 0, "#32CD32", "#FF3B30")

    tot_call_oi = df_raw['Call_OI'].sum()
    tot_put_oi = df_raw['Put_OI'].sum()
    pcr_oi = tot_put_oi / tot_call_oi if tot_call_oi > 0 else 0.0

    # --- METRICHE CON TOOLTIP (Ripristinate dal tuo file originario) ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label=f"🟢 CALL WALL {nome_asset}", value=f"{call_wall:.0f}", help="CALL WALL: Soffitto meccanico. Operatività: Resistenza forte, valutazione take-profit sui long o apertura short.")
    c2.metric(label=f"🟡 HVL (FLIP POINT) {nome_asset}", value=f"{gamma_flip:.2f}", help="HVL (Zero Gamma): Punto di equilibrio. Operatività: Sotto=Volatilità (Short Gamma); Sopra=Trend (Long Gamma).")
    c3.metric(label=f"🔴 PUT WALL {nome_asset}", value=f"{put_wall:.0f}", help="PUT WALL: Pavimento meccanico. Operatività: Supporto forte, zona ideale per buy-the-dip o chiusura short.")
    c4.metric(label="⚖️ P/C RATIO (OI)", value=f"{pcr_oi:.2f}", help="P/C RATIO: Misura dell'eccesso di posizionamento. Operatività: >1.2 pessimismo (rischio squeeze), <0.8 ottimismo (ipercomprato).")
    
    # =====================================================================
    # --- TABS E STRUTTURA GRAFICA ---
    # =====================================================================
    tab_grafico, tab_xray, tab_heatmap, tab_oi = st.tabs(["📊 Istogramma Classico", "🩻 X-Ray Tattico", "🔥 Gamma Heatmap", "🐋 Muri Latenti (OI)"])
    
    with tab_grafico:
        soglia_visibilita = df_utile[metric_col].abs().max() * 0.15
        testi_barre = []
        dimensioni_testo = []
        colori_testo = []

        for index, row in df_utile.iterrows():
            val_y = row[colonna_y]
            val_x = row[metric_col]
            if val_y in [call_wall, put_wall] or abs(val_x) >= soglia_visibilita:
                testi_barre.append(f"<b>{int(round(val_y))}</b>")
                dimensioni_testo.append(14)
                colori_testo.append("black")
            else:
                testi_barre.append(f"{int(round(val_y))}")
                dimensioni_testo.append(11) 
                colori_testo.append("#555555")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_utile[metric_col], y=df_utile[colonna_y], orientation='h',
            marker_color=df_utile["Colore"], text=testi_barre, textposition='outside',
            textfont=dict(size=dimensioni_testo, color=colori_testo, family="Arial Black"), cliponaxis=False
        ))

        fig.add_hline(y=gamma_flip, line_dash="solid", line_color="#FFB300", line_width=3, layer="below")
        fig.add_hline(y=call_wall, line_dash="dash", line_color="#32CD32", line_width=2, layer="below")
        fig.add_hline(y=put_wall, line_dash="dash", line_color="#FF3B30", line_width=2, layer="below")
        fig.add_hline(y=spot_riferimento, line_color="#00FFFF", line_width=2, layer="below")

        fig.add_annotation(x=1, y=call_wall, xref="paper", yref="y", text=f" 🟢 <b>CALL WALL: {int(round(call_wall))}</b> ", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=12, color="black", family="Arial Black"), bgcolor="rgba(245, 255, 245, 0.95)", bordercolor="#32CD32", borderwidth=1, borderpad=4)
        fig.add_annotation(x=1, y=put_wall, xref="paper", yref="y", text=f" 🔴 <b>PUT WALL: {int(round(put_wall))}</b> ", showarrow=False, xanchor="right", yanchor="top", font=dict(size=12, color="black", family="Arial Black"), bgcolor="rgba(255, 245, 245, 0.95)", bordercolor="#FF3B30", borderwidth=1, borderpad=4)
        fig.add_annotation(x=0, y=gamma_flip, xref="paper", yref="y", text=f" 🟡 <b>HVL FLIP: {gamma_flip:.2f}</b> ", showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=12, color="black", family="Arial Black"), bgcolor="rgba(255, 255, 240, 0.95)", bordercolor="#FFB300", borderwidth=1, borderpad=4)
        fig.add_annotation(x=0, y=spot_riferimento, xref="paper", yref="y", text=f" 🔵 <b>SPOT LIVE: {spot_riferimento:.2f}</b> ", showarrow=False, xanchor="left", yanchor="top", font=dict(size=12, color="black", family="Arial Black"), bgcolor="rgba(240, 255, 255, 0.95)", bordercolor="#00FFFF", borderwidth=1, borderpad=4)

        fig.update_layout(height=800, template="plotly_white", xaxis_title=f"<b>Esposizione Monetaria ({metric_col})</b>", yaxis_title=f"<b>Prezzo ({nome_asset})</b>", yaxis=dict(showgrid=True, gridcolor="#EEEEEE"), margin=dict(l=100, r=100, t=20, b=50), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_xray:
        st.markdown("### 🎛️ Centro di Comando Istituzionale")
        usa_savgol = st.checkbox("🔬 Filtro Segnali Savitzky-Golay (Isola i Muri Reali)", value=True)
        df_zone = calcola_zone_flipper(df_utile, colonna_strike=colonna_y, colonna_gex=metric_col, usa_filtro=usa_savgol)
        
        regime_convexity = "🟢 CONVEX (Stabilizzatore)" if spot_riferimento >= gamma_flip else "🔴 CONCAVE (Acceleratore)"
        distanza_flip = spot_riferimento - gamma_flip
        
        col_x1, col_x2, col_x3 = st.columns([1.5, 2, 1.5])
        
        with col_x1:
            st.subheader("📊 Market Regime")
            st.metric("Convexity Status", regime_convexity)
            st.metric("Distanza da HVL (Flip)", f"{distanza_flip:+.2f} pts")
            # --- BIAS STRUTTURALE (VANNA E CHARM) ---
            st.markdown("---")
            st.subheader("🌪️ Bias Strutturale (Dealer)")
            
            with st.expander("📖 Guida Rapida: Come leggere il Bias"):
                st.markdown("""
                Queste metriche svelano le *forze passive* dei Market Maker, che muovono il mercato a prescindere dal prezzo:
                
                * **🌪️ Vanna (Sensibilità al VIX):** Come si coprono i Dealer quando cambia la volatilità. 
                  * **🟢 Long:** Se il VIX scende, i Dealer **comprano** future. Ottimo alleato per assorbire i ritracciamenti (*buy the dip*).
                  * **🔴 Short:** Se il VIX scende, i Dealer **vendono**. Vento contrario per i rialzi.
                  
                * **⏳ Charm (Decadimento Temporale):** L'effetto del tempo che passa verso la scadenza.
                  * **🟢 Rialzista:** Il semplice passare delle ore forza i Dealer a comprare. È la causa delle lente salite pomeridiane o del venerdì (*grind up*). Mai shortare questo regime.
                  * **🔴 Ribassista:** Il tempo forza i Dealer a vendere. Causa sanguinamenti a fine sessione (*bleed*).
                
                💡 **Setup Perfetto:** Regime **CONVEX** + Vanna **🟢** + Charm **🟢** = Trend rialzista blindato. I cali vengono comprati dai Dealer.
                """)
            
            total_vanna = df_utile['Vanna'].sum()
            total_charm = df_utile['Charm'].sum()
            
            status_vanna = "🟢 Long (Compra su IV Crush)" if total_vanna > 0 else "🔴 Short (Vende su IV Crush)"
            status_charm = "🟢 Rialzista (Drift a fine giornata)" if total_charm > 0 else "🔴 Ribassista (Sanguinamento)"
            
            st.metric("Vanna (Effetto Volatilità VIX)", status_vanna)
            st.metric("Charm (Decadimento Tempo)", status_charm)
            
            st.markdown("---")
            st.subheader("⚡ Flusso Intraday")
            df_storico = st.session_state.memoria_heatmap
            orari_unici = df_storico['Orario'].unique() if not df_storico.empty else []
            
            if len(orari_unici) >= 2:
                dex_adesso = df_storico[df_storico['Orario'] == orari_unici[-1]]['DEX_Netto'].sum()
                dex_prima = df_storico[df_storico['Orario'] == orari_unici[-2]]['DEX_Netto'].sum()
                dex_velocity = dex_adesso - dex_prima
                
                status_flow = "⚪ Neutro"
                if dex_velocity > 0: status_flow = "🟢 Dealer Comprano"
                elif dex_velocity < 0: status_flow = "🔴 Dealer Vendono"
                    
                st.metric("⚡ DEX Velocity (Flusso Hedging)", f"${dex_adesso:,.0f}", f"{dex_velocity:+,.0f}")
                st.caption(f"**Azione Real-Time:** {status_flow}")
            else:
                st.metric("⚡ DEX Velocity", "In calcolo...", "In attesa 2° snapshot...")
                
            if spot_riferimento >= gamma_flip: st.info("**Tattica:** Regime Long Gamma. Cerca assorbimenti sui supporti.")
            else: st.error("**Tattica:** Regime Short Gamma. Momentum puro. Attenzione accelerazioni.")
                
        with col_x2:
            st.subheader("🗺️ Mappa Operativa (Zone)")
            df_muri = df_zone[df_zone['Zona_Tattica'] == '🟢 Muro (Assorbimento)'].sort_values(by=colonna_y, ascending=False).reset_index(drop=True)
            mappa_operativa = []
            for i in range(len(df_muri)):
                muro_attuale = df_muri.iloc[i][colonna_y]
                gex_attuale = df_muri.iloc[i][metric_col]
                mappa_operativa.append({
                    "Zona / Livello": f"🧱 MURO: {int(round(muro_attuale))}",
                    "Struttura": f"GEX: ${gex_attuale:,.0f}",
                    "Tattica Consigliata": "🟢 Assorbimento (Mean-Reversion)"
                })
                if i < len(df_muri) - 1:
                    muro_sotto = df_muri.iloc[i+1][colonna_y]
                    mappa_operativa.append({
                        "Zona / Livello": f"💨 TASCA: {int(round(muro_sotto))} ↔ {int(round(muro_attuale))}",
                        "Struttura": "Vuoto di Liquidità",
                        "Tattica Consigliata": "🔴 Accelerazione / Breakout"
                    })
            df_radar_zone = pd.DataFrame(mappa_operativa)
            st.dataframe(df_radar_zone, use_container_width=True, hide_index=True)
            
        with col_x3:
            st.subheader("🎯 Target Dinamici")
            st.metric("Muro Superiore (Call Wall)", f"{call_wall:.0f}", f"{call_wall - spot_riferimento:+.1f} pts dallo spot", delta_color="off")
            st.metric("Muro Inferiore (Put Wall)", f"{put_wall:.0f}", f"{put_wall - spot_riferimento:+.1f} pts dallo spot", delta_color="off")

        st.markdown("---")
        st.subheader("📈 Net Premium Drift (Crossover dei Flussi Intraday)")
        df_storico_drift = st.session_state.memoria_heatmap
        
        if not df_storico_drift.empty and 'Drift_C' in df_storico_drift.columns:
            df_drift = df_storico_drift.groupby('Orario')[['Drift_C', 'Drift_P']].sum().reset_index()
            df_drift = df_drift[(df_drift['Drift_C'] > 0) | (df_drift['Drift_P'] > 0)].reset_index(drop=True)
            
            if not df_drift.empty:
                ultimo_c = df_drift['Drift_C'].iloc[-1]
                ultimo_p = df_drift['Drift_P'].iloc[-1]
                col_d1, col_d2 = st.columns([1, 4])
                with col_d1: st.metric("Dominanza Attuale", "🟢 CALL" if ultimo_c > ultimo_p else "🔴 PUT")
                with col_d2:
                    fig_drift = go.Figure()
                    fig_drift.add_trace(go.Scatter(x=df_drift['Orario'], y=df_drift['Drift_C'], mode='lines+markers', name='Pressione CALL (Acquisti)', line=dict(color='#00E676', width=3)))
                    fig_drift.add_trace(go.Scatter(x=df_drift['Orario'], y=df_drift['Drift_P'], mode='lines+markers', name='Pressione PUT (Vendite)', line=dict(color='#FF3B30', width=3)))
                    fig_drift.update_layout(height=300, template="plotly_dark", hovermode="x unified", margin=dict(l=20, r=20, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig_drift, use_container_width=True)
            else:
                st.warning("⏳ Elaborazione volumi in corso, attendere...")
        else:
            st.warning("⏳ In attesa del secondo snapshot dati per tracciare i flussi...")

    with tab_heatmap:
        st.markdown("### 🔥 Mappa Termica GEX (Evoluzione Intraday)")
        if st.session_state.memoria_heatmap.empty:
            st.warning("⏳ In attesa del primo salvataggio dati per generare la mappa termica...")
        else:
            df_storico_filtrato = st.session_state.memoria_heatmap[(st.session_state.memoria_heatmap['Strike_ETF'] >= limite_inf) & (st.session_state.memoria_heatmap['Strike_ETF'] <= limite_sup)]
            fig_heat = go.Figure(data=go.Heatmap(x=df_storico_filtrato['Orario'], y=df_storico_filtrato['Strike_ETF'], z=df_storico_filtrato['GEX_Netto'], colorscale='RdBu', zmid=0))
            fig_heat.update_layout(height=650, template="plotly_dark", xaxis_title="<b>Orario di Rilevamento</b>", yaxis_title=f"<b>Prezzo ({nome_asset})</b>", margin=dict(l=50, r=50, t=30, b=50))
            fig_heat.add_hline(y=spot_riferimento, line_color="#00FFFF", line_width=2, line_dash="dash", annotation_text=f" Spot: {spot_riferimento:.2f} ", annotation_position="top left")
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab_oi:
        st.markdown("### 🐋 Muri Latenti (Struttura dell'Open Interest)")
        zoom_oi = filtro_percentuale * 3.0
        limite_inf_oi = spot_riferimento * (1 - (zoom_oi / 100.0))
        limite_sup_oi = spot_riferimento * (1 + (zoom_oi / 100.0))
        df_oi = df[(df[colonna_y] >= limite_inf_oi) & (df[colonna_y] <= limite_sup_oi)].copy()
        df_oi = df_oi.sort_values(colonna_y).reset_index(drop=True)
        df_oi['Put_OI_Neg'] = df_oi['Put_OI'] * -1
        
        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(y=df_oi[colonna_y], x=df_oi['Call_OI'], orientation='h', name='Call OI', marker_color='#32CD32', text=df_oi['Call_OI'].apply(lambda x: f"{x:,.0f}" if x > df_oi['Call_OI'].max()*0.1 else ""), textposition='outside', textfont=dict(color="#32CD32")))
        fig_oi.add_trace(go.Bar(y=df_oi[colonna_y], x=df_oi['Put_OI_Neg'], orientation='h', name='Put OI', marker_color='#FF3B30', text=df_oi['Put_OI'].apply(lambda x: f"{x:,.0f}" if x > df_oi['Put_OI'].max()*0.1 else ""), textposition='outside', textfont=dict(color="#FF3B30")))
        fig_oi.add_hline(y=spot_riferimento, line_color="#00FFFF", line_width=2, line_dash="solid", annotation_text=f" Spot: {spot_riferimento:.2f} ", annotation_position="top left")
        fig_oi.update_layout(barmode='relative', height=800, template="plotly_dark", xaxis_title="<b>Open Interest Strutturale</b>", yaxis_title=f"<b>Prezzo ({nome_asset})</b>", margin=dict(l=50, r=50, t=30, b=50))
        st.plotly_chart(fig_oi, use_container_width=True)

# =====================================================================
# PAGINA 2: REPLICA SITO NASDAQ
# =====================================================================
elif pagina == "🗄️ Database Ufficiale Nasdaq":
    NOMI_COMPLETI = {
        "QQQ": "Invesco QQQ Trust, Series 1",
        "SPY": "SPDR S&P 500 ETF Trust",
        "IWM": "iShares Russell 2000 ETF",
        "DIA": "SPDR Dow Jones Industrial Average ETF"
    }
    titolo_esteso = NOMI_COMPLETI.get(ticker, f"Asset: {ticker}")
    st.title(f"{titolo_esteso} ({ticker}) Option Chain")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: filtro_scadenza = st.selectbox("Expiration Dates", scadenze_disponibili)
    with col2: filtro_strategy = st.selectbox("Strategy", ["Calls & Puts", "Calls", "Puts"])
    with col3: filtro_moneyness = st.selectbox("Moneyness", ["All", "Near the Money", "In the Money", "Out of the Money"], index=1)
    with col4: filtro_type = st.selectbox("Type", ["All (Types)", "Weekly", "Monthly", "Quarterly", "CEBO"])

    dati_estratti = ottieni_dati_intelligenti(ticker, filtro_scadenza)
    df_replica = dati_estratti["df"]
    orario_fetch = dati_estratti["ora"]
    
    st.markdown("---")
    st.caption(f"⏱️ **Ultimo cambiamento registrato nei dati della Chain:** {orario_fetch} (Ora Italiana)")
    
    if not df_replica.empty:
        if filtro_moneyness == "Near the Money":
            limite_inf = spot_price_reale * 0.95
            limite_sup = spot_price_reale * 1.05
            df_replica = df_replica[(df_replica['strike'] >= limite_inf) & (df_replica['strike'] <= limite_sup)]
        
        colonne_call = {'c_Last': 'Call Last', 'c_Change': 'Call Change', 'c_Bid': 'Call Bid', 'c_Ask': 'Call Ask', 'c_Volume': 'Call Volume', 'c_Openinterest': 'Call Open Int.'}
        colonne_put = {'p_Last': 'Put Last', 'p_Change': 'Put Change', 'p_Bid': 'Put Bid', 'p_Ask': 'Put Ask', 'p_Volume': 'Put Volume', 'p_Openinterest': 'Put Open Int.'}
        
        df_replica.rename(columns=colonne_call, inplace=True)
        df_replica.rename(columns=colonne_put, inplace=True)
        df_replica.rename(columns={'strike': 'Strike'}, inplace=True)
        df_replica['Strike NQ'] = (df_replica['Strike'] * ratio_esatto).round(1)

        if filtro_strategy == "Calls & Puts":
            colonne_finali = ['Call Last', 'Call Change', 'Call Bid', 'Call Ask', 'Call Volume', 'Call Open Int.', 'Strike', 'Strike NQ', 'Put Last', 'Put Change', 'Put Bid', 'Put Ask', 'Put Volume', 'Put Open Int.']
        elif filtro_strategy == "Calls":
            colonne_finali = ['Call Last', 'Call Change', 'Call Bid', 'Call Ask', 'Call Volume', 'Call Open Int.', 'Strike', 'Strike NQ']
        elif filtro_strategy == "Puts":
            colonne_finali = ['Strike', 'Strike NQ', 'Put Last', 'Put Change', 'Put Bid', 'Put Ask', 'Put Volume', 'Put Open Int.']
            
        df_tabella = df_replica[colonne_finali].copy().replace(0.0, "--")
        
        if not df_tabella.empty:
            strike_piu_vicino = df_tabella.iloc[(df_tabella['Strike'] - spot_price_reale).abs().argsort()[:1]]['Strike'].values[0]
            
            def formatta_base(val): return f"{float(val):.2f}" if not pd.isna(val) and val != "--" else "--"
            def formatta_interi(val): return f"{int(float(val))}" if not pd.isna(val) and val != "--" else "--"
            def formatta_change(val):
                if pd.isna(val) or val == "--": return "--"
                v = float(val)
                return f"▲ {v:.2f}" if v > 0 else f"▼ {abs(v):.2f}" if v < 0 else "--"
            
            colonne_change = [c for c in ['Call Change', 'Put Change'] if c in df_tabella.columns]
            colonne_intere = [c for c in ['Call Volume', 'Call Open Int.', 'Put Volume', 'Put Open Int.'] if c in df_tabella.columns]
            
            dict_formattazione = {col: formatta_base for col in df_tabella.columns if col not in colonne_change and col not in colonne_intere}
            for col in colonne_change: dict_formattazione[col] = formatta_change
            for col in colonne_intere: dict_formattazione[col] = formatta_interi

            def colora_celle_change(s):
                stili = []
                for val in s:
                    if pd.isna(val) or val == "--": stili.append("")
                    else:
                        v = float(val)
                        if v > 0: stili.append("color: #00C853; font-weight: bold;")
                        elif v < 0: stili.append("color: #FF3B30; font-weight: bold;")
                        else: stili.append("")
                return stili
            
            def evidenzia_spot(row):
                return ['background-color: rgba(0, 255, 255, 0.15)'] * len(row) if row['Strike'] == strike_piu_vicino else [''] * len(row)
            
            df_styled = (df_tabella.style
                         .format(dict_formattazione)
                         .apply(colora_celle_change, subset=colonne_change, axis=0)
                         .apply(evidenzia_spot, axis=1))
            
            st.markdown(f"📍 *La riga azzurra indica l'area At-The-Money (Strike più vicino allo spot attuale di {spot_price_reale:.2f}$)*")
            st.dataframe(df_styled, use_container_width=True, height=800)
        else:
            st.dataframe(df_tabella, use_container_width=True, height=800)
    else:
        st.warning("Dati non trovati per la data selezionata. Il mercato potrebbe non aver ancora popolato i volumi.")