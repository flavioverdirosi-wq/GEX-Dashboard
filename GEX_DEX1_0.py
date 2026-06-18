"""
VERSIONE 1.0 - EOGA GEX/DEX ENGINE
-------------------------------------------------------------------------
MOTORE DI ANALISI:
1. GEX (Gamma Exposure) Dinamico: Implementa un motore vettoriale 
   basato su matrici NumPy per la simulazione Black-Scholes a 200 tick 
   di prezzo. Identifica l'HVL (Gamma Flip) reale attraverso il root-finding 
   sullo zero matematico (0.0), superando l'approssimazione statica.
   
2. ARCHITETTURA: 
   - Utilizzo di calcolo parallelo (broadcasting) per simulare 
     dinamicamente la sensibilità delle opzioni (Greeks) su 5.000+ strike.
   - UI ottimizzata con visualizzazione gerarchica per la leggibilità dei livelli.

3. DEX (Delta Exposure): 
   - Attualmente in modalità di visualizzazione statica (Versione 1.0). 
   - Pianificata integrazione dinamica (DEX Velocity & Vanna) per la V 2.0.
-------------------------------------------------------------------------
"""

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

# =====================================================================
# CONFIGURAZIONE PAGINA E STATO
# =====================================================================
st.set_page_config(page_title="EOGA GEX/DEX Engine", layout="wide")

if "memoria_dati" not in st.session_state:
    st.session_state.memoria_dati = {}

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
    if sigma <= 0 or t <= 0 or S <= 0 or K <= 0: return 0, 0, 0
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * t) / (sigma * np.sqrt(t))
    n_prime_d1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d1**2))
    delta_call = si.norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = n_prime_d1 / (S * sigma * np.sqrt(t))
    return delta_call, delta_put, gamma

def pulisci_numero(val):
    import pandas as pd # Sicurezza anti-crash
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

# =====================================================================
# MOTORE DI ESTRAZIONE DATI E GESTIONE CACHE
# =====================================================================
@st.cache_resource
def inizializza_ticker(symbol):
    return yf.Ticker(symbol)

@st.cache_data(ttl=30)
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

@st.cache_data(ttl=30)
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

@st.cache_data(ttl=299)
def scarica_chain_nasdaq_pura(ticker, data_scadenza, asset_class="etf"):
    import pandas as pd # <-- SOLUZIONE BOMB-PROOF PER L'ERRORE PD
    import requests
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
    import pandas as pd # Sicurezza aggiuntiva
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

# Input Ticker
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

# Mappatura Futures
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

# Calcolo Chiusure e Moltiplicatore
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

# Refresh e Stato Mercato
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Controllo Dati")
col_r1, col_r2 = st.sidebar.columns([3, 1])
with col_r1:
    auto_refresh = st.checkbox("⏱️ Auto-Refresh (2 min)", value=True)
with col_r2:
    if st.button("Forza", use_container_width=True):
        st.cache_data.clear() 
        st.rerun()

if auto_refresh:
    import streamlit.components.v1 as components
    components.html("<script>setTimeout(function(){window.parent.location.reload();}, 120000);</script>", height=0, width=0)

stato_mercato = verifica_stato_mercato()
tz_it = pytz.timezone('Europe/Rome')
ora_it = datetime.datetime.now(tz_it).strftime("%d %b %Y - %H:%M IT")

# 1. Calcolo dell'orario di New York (EST/EDT) per la zona evidenziata in blu
tz_est = pytz.timezone('America/New_York')
ora_est = datetime.datetime.now(tz_est).strftime("%d %b %Y - %H:%M NY")

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
    
    valore_etf = etf_realtime_nasdaq if etf_realtime_nasdaq is not None else 0.0
    valore_fut = future_realtime_yf if future_realtime_yf is not None else 0.0
    status_color = "#00E676" if "APERTO" in stato_mercato else "#FF3B30"
    stato_pulito = stato_mercato.replace('🟢', '').replace('🔴', '').strip()
    
    # Calcolo dinamico della variazione % rispetto alla chiusura per la zona evidenziata in rosso
    if prezzo_chiusura > 0:
        var_pct_etf = ((valore_etf - prezzo_chiusura) / prezzo_chiusura) * 100
    else:
        var_pct_etf = 0.0
        
    colore_var = "#00E676" if var_pct_etf >= 0 else "#FF3B30"
    segno_var = "+" if var_pct_etf >= 0 else ""
    testo_var = f"{segno_var}{var_pct_etf:.2f}%"

    # HTML TOP BAR - AGGIORNATO CON CHG % E DOPPIO ORARIO
    top_bar_html = f"""<div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(145deg, #1A1D24 0%, #131722 100%); padding: 20px 30px; border-radius: 12px; border: 1px solid #2B3139; box-shadow: 0px 8px 20px rgba(0,0,0,0.4); margin-bottom: 25px;">
    <div style="flex: 1;">
        <h1 style="margin: 0; color: #E0E3EB; font-size: 28px; font-family: 'Arial Black', sans-serif; text-transform: uppercase; letter-spacing: 1.5px;">🎯 EOGA <span style="color: #FFD700;">GEX</span></h1>
        <span style="color: #8C92A4; font-size: 13px; text-transform: uppercase; letter-spacing: 2px;">Advanced Order Book</span>
    </div>
    <div style="display: flex; gap: 35px; align-items: center; justify-content: center; flex: 2.5;">
        <div style="text-align: right;">
            <span style="color: #8C92A4; font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;">{ticker} (Spot)</span><br>
            <span style="color: #FFFFFF; font-size: 34px; font-weight: 900; font-family: 'Courier New', monospace; text-shadow: 0 0 10px rgba(255,255,255,0.1);">${valore_etf:,.2f}</span>
        </div>
        <div style="height: 50px; width: 2px; background: linear-gradient(to bottom, transparent, #3A414D, transparent);"></div>
        <div style="text-align: left;">
            <span style="color: #8C92A4; font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;">{nome_future} (Future)</span><br>
            <span style="color: #FFFFFF; font-size: 34px; font-weight: 900; font-family: 'Courier New', monospace; text-shadow: 0 0 10px rgba(255,255,255,0.1);">{valore_fut:,.2f}</span>
        </div>
        <div style="height: 50px; width: 2px; background: linear-gradient(to bottom, transparent, #3A414D, transparent);"></div>
        <div style="text-align: left;">
            <span style="color: #8C92A4; font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;">{ticker} Chg %</span><br>
            <span style="color: {colore_var}; font-size: 34px; font-weight: 900; font-family: 'Courier New', monospace; text-shadow: 0 0 10px {colore_var}20;">{testo_var}</span>
        </div>
    </div>
    <div style="flex: 1; text-align: right;">
        <div style="display: inline-flex; align-items: center; gap: 10px; background-color: rgba(255,255,255,0.03); padding: 8px 15px; border-radius: 30px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 4px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {status_color}; box-shadow: 0 0 8px {status_color};"></div>
            <span style="color: #E0E3EB; font-weight: bold; font-size: 14px; letter-spacing: 1px;">{stato_pulito}</span>
        </div>
        <div style="color: #8C92A4; font-size: 12px; font-family: monospace; line-height: 1.3;">{ora_it}</div>
        <div style="color: #53B9EA; font-size: 12px; font-family: monospace; font-weight: bold; line-height: 1.3;">{ora_est}</div>
    </div>
</div>"""
    st.markdown(top_bar_html, unsafe_allow_html=True)

    # --- FILTRI ---
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
    if usa_total_gex:
        st.warning(f"⏳ Elaborazione Total GEX in corso ({len(scadenze_da_analizzare)} scadenze). Potrebbe richiedere 5-10 secondi...")

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
            
            if oi_call > 0:
                d_c, _, gamma_c = calcola_greche_base(spot_price_reale, K, t_anno, iv_stimata)
                struttura.append({
                    "Strike_ETF": K, "Strike_Future": K * ratio_esatto, 
                    "GEX": gamma_c * oi_call * 100 * (spot_price_reale**2) * 0.01, 
                    "DEX": d_c * oi_call * 100 * spot_price_reale * 0.01,
                    "Call_OI": oi_call, "Put_OI": 0,
                    "t_anno": t_anno # <--- DATO AGGIUNTO PER LA MATRICE
                })
                
            if oi_put > 0:
                _, d_p, gamma_p = calcola_greche_base(spot_price_reale, K, t_anno, iv_stimata)
                struttura.append({
                    "Strike_ETF": K, "Strike_Future": K * ratio_esatto, 
                    "GEX": -gamma_p * oi_put * 100 * (spot_price_reale**2) * 0.01, 
                    "DEX": d_p * oi_put * 100 * spot_price_reale * 0.01,
                    "Call_OI": 0, "Put_OI": oi_put,
                    "t_anno": t_anno # <--- DATO AGGIUNTO PER LA MATRICE
                })

    import pandas as pd
    df_raw = pd.DataFrame(struttura)
    if df_raw.empty: 
        st.error("Nessun dato estrapolato.")
        st.stop()

    df = df_raw.groupby(["Strike_ETF", "Strike_Future"]).sum().reset_index()

    # --- LOGICA DI VISUALIZZAZIONE ---
    colonna_y = "Strike_ETF" if mostra_etf else "Strike_Future"
    spot_riferimento = etf_realtime_nasdaq if mostra_etf else future_realtime_yf
    nome_asset = ticker if mostra_etf else nome_future

    limite_inf = spot_riferimento * (1 - (filtro_percentuale / 100.0))
    limite_sup = spot_riferimento * (1 + (filtro_percentuale / 100.0))

    df_utile = df[(df[colonna_y] >= limite_inf) & (df[colonna_y] <= limite_sup)].copy()
    if df_utile.empty: df_utile = df.copy()

    df_utile = df_utile.sort_values(colonna_y).reset_index(drop=True)

    call_wall = df_utile.loc[df_utile["GEX"].idxmax()][colonna_y]
    put_wall = df_utile.loc[df_utile["GEX"].idxmin()][colonna_y]

    # ==========================================
    # CALCOLO HVL (FLIP POINT) - DOPPIO MOTORE
    # ==========================================
    if usa_hvl_istituzionale:
        # MOTORE VETTORIALE NUMPY (Simulazione Dinamica Istituzionale)
        spot_simulati = np.linspace(spot_riferimento * 0.9, spot_riferimento * 1.1, 200)
        strikes = df_raw[colonna_y].values
        oi_calls = df_raw["Call_OI"].values
        oi_puts = df_raw["Put_OI"].values
        t_annos = df_raw["t_anno"].values
        
        # 1. Creazione Matrici per Broadcasting
        S = spot_simulati[:, np.newaxis]
        K = strikes[np.newaxis, :]
        T = t_annos[np.newaxis, :]
        
        # 2. Ricalcolo Vettorializzato di Black-Scholes
        sigma_sqrt_t = iv_stimata * np.sqrt(T)
        sigma_sqrt_t = np.where(sigma_sqrt_t == 0, 1e-9, sigma_sqrt_t) # Previene errori di divisione
        
        d1 = (np.log(S / K) + (0.045 + (iv_stimata**2) / 2) * T) / sigma_sqrt_t
        n_prime_d1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d1**2))
        gamma_matrice = n_prime_d1 / (S * sigma_sqrt_t)
        
        # 3. Bilancio totale simulato per ogni tick di prezzo
        gex_totale_simulato = np.sum(gamma_matrice * oi_calls * (S**2) - gamma_matrice * oi_puts * (S**2), axis=1)
        
        # 4. Ricerca del Root (attraversamento dello zero matematico)
        cambi_segno = np.where(np.diff(np.sign(gex_totale_simulato)))[0]
        if len(cambi_segno) > 0:
            idx = cambi_segno[0]
            # Interpolazione lineare tra i due tick per precisione sub-decimale
            s1, s2 = spot_simulati[idx], spot_simulati[idx+1]
            g1, g2 = gex_totale_simulato[idx], gex_totale_simulato[idx+1]
            gamma_flip = s1 - g1 * ((s2 - s1) / (g2 - g1)) 
        else:
            # Fallback se la curva non incrocia mai lo zero
            gamma_flip = spot_simulati[np.argmin(np.abs(gex_totale_simulato))]
    else:
        # MOTORE STATICO (Approssimazione Cumulativa Base)
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

    # --- METRICHE ---
    tot_call_oi = df_raw['Call_OI'].sum()
    tot_put_oi = df_raw['Put_OI'].sum()
    pcr_oi = tot_put_oi / tot_call_oi if tot_call_oi > 0 else 0.0

   # --- RENDERIZZAZIONE METRICHE TATTICHE (Versione 1.0) ---
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric(
        label=f"🟢 CALL WALL {nome_asset}", 
        value=f"{call_wall:.0f}",
        help="""CALL WALL: Soffitto meccanico. 
        Operatività: Resistenza forte, valutazione take-profit sui long o apertura short."""
    )
    
    c2.metric(
        label=f"🟡 HVL (FLIP POINT) {nome_asset}", 
        value=f"{gamma_flip:.2f}",
        help="""HVL (Zero Gamma): Punto di equilibrio. 
        Operatività: Sotto=Volatilità (Short Gamma); Sopra=Trend (Long Gamma)."""
    )
    
    c3.metric(
        label=f"🔴 PUT WALL {nome_asset}", 
        value=f"{put_wall:.0f}",
        help="""PUT WALL: Pavimento meccanico. 
        Operatività: Supporto forte, zona ideale per buy-the-dip o chiusura short."""
    )
    
    c4.metric(
        label="⚖️ P/C RATIO (OI)", 
        value=f"{pcr_oi:.2f}",
        help="""P/C RATIO: Misura dell'eccesso di posizionamento. 
        Operatività: >1.2 pessimismo (rischio squeeze), <0.8 ottimismo (ipercomprato)."""
    )
    
    
    # --- RENDERIZZAZIONE GRAFICO PLOTLY PULITO ---
    soglia_visibilita = df_utile[metric_col].abs().max() * 0.15
    
    testi_barre = []
    dimensioni_testo = []
    colori_testo = []

    for index, row in df_utile.iterrows():
        val_y = row[colonna_y]
        val_x = row[metric_col]
        
        # Livelli Master e barre con grandi volumi (> 15% del picco): GRASSETTO NERO
        if val_y in [call_wall, put_wall] or abs(val_x) >= soglia_visibilita:
            testi_barre.append(f"<b>{int(round(val_y))}</b>")
            dimensioni_testo.append(14)
            colori_testo.append("black")
            
        # Livelli Minori: TESTO NORMALE E GRIGIO (così li vedi ma non creano confusione)
        else:
            testi_barre.append(f"{int(round(val_y))}")
            dimensioni_testo.append(11) 
            colori_testo.append("#555555") # Grigio scuro

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_utile[metric_col], 
        y=df_utile[colonna_y], 
        orientation='h',
        marker_color=df_utile["Colore"], 
        text=testi_barre, 
        textposition='outside',
        textfont=dict(size=dimensioni_testo, color=colori_testo, family="Arial Black"), 
        cliponaxis=False
    ))

# =====================================================================
    # 3. LINEE ORIZZONTALI E ETICHETTE DI SICUREZZA ANTI-SOVRAPPOSIZIONE
    # =====================================================================
    
    # Tracciamento linee base (sotto il grafico)
    fig.add_hline(y=gamma_flip, line_dash="solid", line_color="#FFB300", line_width=3, layer="below")
    fig.add_hline(y=call_wall, line_dash="dash", line_color="#32CD32", line_width=2, layer="below")
    fig.add_hline(y=put_wall, line_dash="dash", line_color="#FF3B30", line_width=2, layer="below")
    fig.add_hline(y=spot_riferimento, line_color="#00FFFF", line_width=2, layer="below")

    # --- ETICHETTE LATO DESTRO (Ancorate al margine destro del grafico) ---
    
    # CALL WALL - Scritta sopra la linea tratteggiata verde
    fig.add_annotation(
        x=1, y=call_wall, xref="paper", yref="y",
        text=f" 🟢 <b>CALL WALL: {int(round(call_wall))}</b> ",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=12, color="black", family="Arial Black"),
        bgcolor="rgba(245, 255, 245, 0.95)", bordercolor="#32CD32", borderwidth=1, borderpad=4
    )

    # PUT WALL - Scritta sotto la linea tratteggiata rossa
    fig.add_annotation(
        x=1, y=put_wall, xref="paper", yref="y",
        text=f" 🔴 <b>PUT WALL: {int(round(put_wall))}</b> ",
        showarrow=False, xanchor="right", yanchor="top",
        font=dict(size=12, color="black", family="Arial Black"),
        bgcolor="rgba(255, 245, 245, 0.95)", bordercolor="#FF3B30", borderwidth=1, borderpad=4
    )

    # --- ETICHETTE LATO SINISTRO (Ancorate al margine sinistro del grafico) ---
    
    # HVL FLIP POINT - Scritta sopra la linea continua gialla
    fig.add_annotation(
        x=0, y=gamma_flip, xref="paper", yref="y",
        text=f" 🟡 <b>HVL FLIP: {gamma_flip:.2f}</b> ",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=12, color="black", family="Arial Black"),
        bgcolor="rgba(255, 255, 240, 0.95)", bordercolor="#FFB300", borderwidth=1, borderpad=4
    )

    # PREZZO SPOT LIVE - Scritta sotto la linea continua azzurra
    fig.add_annotation(
        x=0, y=spot_riferimento, xref="paper", yref="y",
        text=f" 🔵 <b>SPOT LIVE: {spot_riferimento:.2f}</b> ",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=12, color="black", family="Arial Black"),
        bgcolor="rgba(240, 255, 255, 0.95)", bordercolor="#00FFFF", borderwidth=1, borderpad=4
    )

    fig.update_layout(
        height=800, 
        template="plotly_white", 
        xaxis_title=f"<b>Esposizione Monetaria ({metric_col})</b>", 
        yaxis_title=f"<b>Prezzo ({nome_asset})</b>", 
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE"),
        margin=dict(l=100, r=100, t=20, b=50),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

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
        
        import pandas as pd
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
