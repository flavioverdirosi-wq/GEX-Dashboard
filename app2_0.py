import calendar
import datetime
import pytz
import numpy as np
import pandas as pd  # <--- QUESTA È LA RIGA MANCANTE
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
# --- FUNZIONI DI SUPPORTO ---
@st.cache_data(ttl=60)
def get_vix_data():
    try:
        vix = yf.Ticker("^VIX").fast_info['last_price']
        vxn = yf.Ticker("^VXN").fast_info['last_price']
        return vix, vxn
    except:
        return 15.0, 15.0

# --- SEGUONO LE ALTRE FUNZIONI (scarica_prezzo_spot, ecc.) ---
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

@st.cache_data(ttl=30) # Aggiorna ogni 30 secondi per mantenere il real-time
def scarica_prezzo_spot(symbol):
    try:
        tk_obj = yf.Ticker(symbol)
        # prepost=True permette di scaricare i dati anche a mercato chiuso
        storico = tk_obj.history(period="1d", interval="1m", prepost=True)
        if storico.empty: 
            storico_giornaliero = tk_obj.history(period="1d")
            return storico_giornaliero["Close"].iloc[-1]
        return storico["Close"].iloc[-1]
    except Exception:
        return None

@st.cache_data(ttl=30)
def scarica_quote_nasdaq(ticker):
    """Estrae il prezzo in tempo reale / after-hours dall'API del Nasdaq"""
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
                # Cerca prima i dati estesi (pre/post market), altrimenti il mercato primario
                ext_data = data.get("extendedMarketData", {})
                prim_data = data.get("primaryData", {})
                
                ext_price = ext_data.get("lastSalePrice", "") if ext_data else ""
                if ext_price:
                    return float(ext_price.replace("$", "").replace(",", ""))
                
                prim_price = prim_data.get("lastSalePrice", "") if prim_data else ""
                if prim_price:
                    return float(prim_price.replace("$", "").replace(",", ""))
    except Exception:
        pass
    return None # Ritorna None se l'API fallisce, attivando il fallback su Yahoo

@st.cache_data(ttl=300) 
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
# SIDEBAR E BOTTONE AGGIORNA
# =====================================================================
st.sidebar.title("🧭 Navigazione App")
pagina = st.sidebar.radio("Seleziona la vista:", ["📊 Dashboard Grafica (GEX)", "🗄️ Database Ufficiale Nasdaq"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Sincronizzazione Dati")

# ==========================================
# 1. GESTIONE INPUT TICKER CON BOTTONE
# ==========================================
col_t1, col_t2 = st.sidebar.columns([3, 1])
with col_t1:
    ticker_input = st.text_input("Ticker Sottostante (es. QQQ, SPY)", "QQQ").upper()
with col_t2:
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    btn_vai = st.button("Vai 🚀", use_container_width=True)

# Gestione session_state per non perdere il ticker ai ricaricamenti
if "ticker_attivo" not in st.session_state:
    st.session_state.ticker_attivo = "QQQ"

if btn_vai:
    st.session_state.ticker_attivo = ticker_input
    st.cache_data.clear() # Pulisce la cache per forzare i nuovi dati

ticker = st.session_state.ticker_attivo

# ==========================================
# 2. MAPPATURA DINAMICA ETF -> FUTURE
# ==========================================
MAPPATURA_FUTURES = {
    "QQQ": {"simbolo": "NQU26.CME", "nome": "NQ"},
    "SPY": {"simbolo": "ESU26.CME", "nome": "ES"},
    "IWM": {"simbolo": "RTYU26.CME", "nome": "RTY"},
    "DIA": {"simbolo": "YMU26.CME", "nome": "YM"}
}

# Fallback se metti un ticker non mappato (usa l'ETF stesso per entrambi)
info_future = MAPPATURA_FUTURES.get(ticker, {"simbolo": ticker, "nome": ticker})
ticker_future = info_future["simbolo"]
nome_future = info_future["nome"]

tk = inizializza_ticker(ticker)
spot_price_reale = scarica_prezzo_spot(ticker)

if spot_price_reale is None:
    st.sidebar.error(f"Impossibile scaricare il prezzo per {ticker}.")
    st.stop()

# Recupera la chiusura ufficiale (ore 22:00 IT)
try:
    storico_daily = tk.history(period="5d", auto_adjust=False)
    if "APERTO" in verifica_stato_mercato():
        prezzo_chiusura_api = storico_daily['Close'].iloc[-2]
    else:
        prezzo_chiusura_api = storico_daily['Close'].iloc[-1]
except:
    prezzo_chiusura_api = spot_price_reale

# ==========================================
# OVERRIDE MANUALE CHIUSURA ETF
# ==========================================
usa_chiusura_manuale = st.sidebar.checkbox(f"Modifica Chiusura {ticker} a mano", value=False)

if usa_chiusura_manuale:
    prezzo_chiusura = st.sidebar.number_input(f"Prezzo {ticker} (Chiusura 22:00 IT)", value=float(prezzo_chiusura_api), step=0.10)
else:
    prezzo_chiusura = prezzo_chiusura_api
    st.sidebar.text_input(f"Prezzo {ticker} (Chiusura 22:00 IT)", value=f"{prezzo_chiusura:.2f} $", disabled=True)

# ==========================================
# ESTRAZIONE E OVERRIDE MANUALE FUTURE (DINAMICO)
# ==========================================
tk_future = yf.Ticker(ticker_future)
future_spot_reale = scarica_prezzo_spot(ticker_future)

try:
    storico_fut_1m = tk_future.history(period="5d", interval="1m", prepost=True)
    if not storico_fut_1m.empty:
        storico_fut_1m.index = storico_fut_1m.index.tz_convert('Europe/Rome')
        candele_pre_chiusura = storico_fut_1m[(storico_fut_1m.index.hour == 21) & (storico_fut_1m.index.minute >= 50)]
        if not candele_pre_chiusura.empty:
            chiusure_giornaliere_22 = candele_pre_chiusura.groupby(candele_pre_chiusura.index.date).last()
            prezzo_fut_api = chiusure_giornaliere_22['Close'].iloc[-1]
        else:
            prezzo_fut_api = future_spot_reale
    else:
        prezzo_fut_api = future_spot_reale
except:
    prezzo_fut_api = future_spot_reale if future_spot_reale is not None else spot_price_reale

usa_fut_manuale = st.sidebar.checkbox(f"Modifica Chiusura {nome_future} a mano", value=False)

if usa_fut_manuale:
    prezzo_future_man = st.sidebar.number_input(f"Prezzo Future {nome_future} (Riferimento per Greche)", value=float(prezzo_fut_api), step=10.0)
else:
    prezzo_future_man = prezzo_fut_api
    st.sidebar.text_input(f"Prezzo Future {nome_future} (Riferimento per Greche)", value=f"{prezzo_future_man:.2f}", disabled=True)

# Moltiplicatore dinamico
ratio_esatto = prezzo_future_man / prezzo_chiusura if prezzo_chiusura > 0 else 1.0
st.sidebar.metric("Moltiplicatore Calcolato", value=f"{ratio_esatto:.4f}x")

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Controllo Dati")
if st.sidebar.button("Forza Aggiornamento Ora", use_container_width=True):
    st.cache_data.clear() 
    st.sidebar.success("✅ Richiesta nuovi dati inviata!")
    st.rerun()

stato_mercato = verifica_stato_mercato()
tz_it = pytz.timezone('Europe/Rome')
ora_it = datetime.datetime.now(tz_it).strftime("%d %b %Y - %H:%M IT")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Valori di Mercato")

# =====================================================================
# RECUPERO DATI IN TEMPO REALE E HTML BOX DINAMICO
# =====================================================================
etf_realtime_nasdaq = scarica_quote_nasdaq(ticker)
if etf_realtime_nasdaq is None:
    etf_realtime_nasdaq = spot_price_reale 

future_realtime_yf = scarica_prezzo_spot(ticker_future)
if future_realtime_yf is None:
    future_realtime_yf = prezzo_future_man

colore_bg = "#00C853" if "APERTO" in stato_mercato else "#131722"
colore_text = "#FFFFFF" if "APERTO" in stato_mercato else "#B2B5BE"

html_box = f"""
<div style="background-color: {colore_bg}; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
    <h4 style="margin: 0; color: {colore_text}; font-size: 14px; opacity: 0.8;">{ticker} (Real-Time Nasdaq)</h4>
    <h1 style="margin: 0; color: {colore_text}; font-size: 32px;">${etf_realtime_nasdaq:.2f}</h1>
    <hr style="border-color: {colore_text}; opacity: 0.2; margin: 10px 0;">
    <h4 style="margin: 0; color: {colore_text}; font-size: 14px; opacity: 0.8;">{nome_future}=F (Real-Time Future)</h4>
    <h1 style="margin: 0; color: {colore_text}; font-size: 28px;">{future_realtime_yf:.2f}</h1>
    <p style="margin: 10px 0 0 0; color: {colore_text}; font-size: 16px; font-weight: bold;">{stato_mercato}</p>
    <p style="margin: 2px 0 0 0; color: {colore_text}; font-size: 13px; opacity: 0.9;">{ora_it}</p>
</div>
"""
st.sidebar.markdown(html_box, unsafe_allow_html=True)
scadenze_disponibili = list(tk.options)
if not scadenze_disponibili:
    st.error(f"Nessuna data trovata per {ticker}.")
    st.stop()

opex_mensile_calcolato = trova_prossimo_opex_mensile()

# =====================================================================
# PAGINA 1: DASHBOARD GRAFICA
# =====================================================================
if pagina == "📊 Dashboard Grafica (GEX)":
    st.title("🎯 EOGA GEX & DEX Order Book")
    
    # =====================================================================
    # DASHBOARD GRAFICA - FILTRI E TOGGLE DINAMICO ETF/FUTURE
    # =====================================================================
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        scadenza_sel = st.selectbox("Scadenza Analisi:", scadenze_disponibili)
    with col_f2:
        filtro_percentuale = st.slider("Zoom Grafico (+/- % dal prezzo)", min_value=1, max_value=20, value=1)
    with col_f3:
        tipo_visualizzazione = st.radio("Visualizza Istogramma:", ["GEX (Gamma)", "DEX (Delta)"], horizontal=True)
    with col_f4:
        st.markdown("<br>", unsafe_allow_html=True)
        mostra_etf = st.checkbox(f"🔄 Mostra livelli in {ticker}", value=False, help=f"Visualizza asse Y e livelli sul prezzo dell'ETF {ticker} invece che sul Future {nome_future}.")
    
    # Integrazione VIX Live come motore per le Greche
    vix_live, vxn_live = get_vix_data()
    iv_stimata = vix_live / 100.0
    st.info(f"⚙️ Motore Greche alimentato da VIX Live: {vix_live:.2f}% (Rif. VXN: {vxn_live:.2f}%)")
        
    dati_estratti = ottieni_dati_intelligenti(ticker, scadenza_sel)
    df_nasdaq_grafico = dati_estratti["df"]
    orario_fetch = dati_estratti["ora"]
    
    if df_nasdaq_grafico.empty:
        st.warning("⚠️ Dati non disponibili per questa scadenza dal Nasdaq.")
        st.stop()
        
    st.caption(f"⏱️ **Ultimo cambiamento registrato nei dati della Chain:** {orario_fetch} (Ora Italiana)")

    oggi = datetime.date.today()
    data_scad = datetime.datetime.strptime(scadenza_sel, "%Y-%m-%d").date()
    giorni = (data_scad - oggi).days
    t_anno = (0.5 / 365.0) if giorni <= 0 else (giorni / 365.0)

    # ==========================================
    # ELABORAZIONE STRUTTURA GEX E DEX (Doppio Strike)
    # ==========================================
    struttura = []
    for _, riga in df_nasdaq_grafico.iterrows():
        K = riga["strike"]
        oi_call = riga.get("c_Openinterest", 0)
        oi_put = riga.get("p_Openinterest", 0)
        
        if oi_call > 0:
            d_c, _, gamma_c = calcola_greche_base(spot_price_reale, K, t_anno, iv_stimata)
            struttura.append({
                "Strike_ETF": K, 
                "Strike_Future": K * ratio_esatto, 
                "GEX": gamma_c * oi_call * 100 * (spot_price_reale**2) * 0.01, 
                "DEX": d_c * oi_call * 100 * spot_price_reale * 0.01
            })
            
        if oi_put > 0:
            _, d_p, gamma_p = calcola_greche_base(spot_price_reale, K, t_anno, iv_stimata)
            struttura.append({
                "Strike_ETF": K, 
                "Strike_Future": K * ratio_esatto, 
                "GEX": -gamma_p * oi_put * 100 * (spot_price_reale**2) * 0.01, 
                "DEX": d_p * oi_put * 100 * spot_price_reale * 0.01
            })

    df_raw = pd.DataFrame(struttura)
    if df_raw.empty: st.stop()

    # Raggruppiamo preservando entrambe le colonne di prezzo
    df = df_raw.groupby(["Strike_ETF", "Strike_Future"]).sum().reset_index()

    # ==========================================
    # APPLICAZIONE LOGICA TOGGLE DINAMICA
    # ==========================================
    colonna_y = "Strike_ETF" if mostra_etf else "Strike_Future"
    spot_riferimento = etf_realtime_nasdaq if mostra_etf else future_realtime_yf
    nome_asset = ticker if mostra_etf else nome_future

    limite_inf = spot_riferimento * (1 - (filtro_percentuale / 100.0))
    limite_sup = spot_riferimento * (1 + (filtro_percentuale / 100.0))

    df_utile = df[(df[colonna_y] >= limite_inf) & (df[colonna_y] <= limite_sup)].copy()
    if df_utile.empty: df_utile = df.copy()

    # Ordinamento per calcolare l'HVL correttamente
    df_utile = df_utile.sort_values(colonna_y).reset_index(drop=True)

    call_wall = df_utile.loc[df_utile["GEX"].idxmax()][colonna_y]
    put_wall = df_utile.loc[df_utile["GEX"].idxmin()][colonna_y]

    # ==========================================
    # CALCOLO HVL (FLIP POINT) ESATTAMENTE A METÀ
    # ==========================================
    df_utile["GEX_Cum"] = df_utile["GEX"].cumsum()
    idx_flip = np.where(np.diff(np.sign(df_utile["GEX_Cum"])) != 0)[0]
    
    if len(idx_flip) > 0:
        indice_sotto = idx_flip[0]
        indice_sopra = indice_sotto + 1
        
        if indice_sopra < len(df_utile):
            strike_sotto = df_utile.iloc[indice_sotto][colonna_y]
            strike_sopra = df_utile.iloc[indice_sopra][colonna_y]
            gamma_flip = (strike_sotto + strike_sopra) / 2.0
        else:
            gamma_flip = df_utile.iloc[indice_sotto][colonna_y]
    else:
        gamma_flip = df_utile.loc[df_utile["GEX_Cum"].abs().idxmin()][colonna_y]

    metric_col = "GEX" if tipo_visualizzazione == "GEX (Gamma)" else "DEX"
    df_utile["Colore"] = np.where(df_utile[metric_col] >= 0, "#32CD32", "#FF3B30")

    # ==========================================
    # CALCOLO PUT/CALL RATIO E METRICHE
    # ==========================================
    tot_call_oi = df_nasdaq_grafico['c_Openinterest'].sum()
    tot_put_oi = df_nasdaq_grafico['p_Openinterest'].sum()
    pcr_oi = tot_put_oi / tot_call_oi if tot_call_oi > 0 else 0.0

    # ==========================================
    # TESTI PER I TOOLTIP (HELP)
    # ==========================================
    help_call_wall = """
    **🟢 CALL WALL (Il Magnete e il Tetto)**
    * **Cos'è:** Lo strike con la massima esposizione GEX positiva. I Market Maker sono "Long Gamma".
    * **Meccanica:** Per coprirsi, i dealer operano *contro-trend* (vendono futures sui rialzi, comprano sui ribassi).
    * **Operatività:** Quando il prezzo è sotto, fa da calamita. Una volta raggiunto, la volatilità crolla e funge da "tetto". Cerca setup di mean-reversion (es. short su mancata rottura).
    """

    help_gamma_flip = """
    **🟡 GAMMA FLIP (Punto di Flesso del Regime)**
    * **Cos'è:** Il livello di "Zero Gamma", dove l'esposizione totale passa da positiva a negativa.
    * **Meccanica:** Sopra il livello i dealer assorbono volatilità (mercato tranquillo). Sotto il livello la amplificano (mercato tossico).
    * **Operatività:** È il filtro direzionale primario. Sotto il Gamma Flip le discese diventano veloci e verticali. Evitare long avventati.
    """

    help_put_wall = """
    **🔴 PUT WALL (Il Pavimento e l'Acceleratore)**
    * **Cos'è:** Lo strike con la massima esposizione GEX negativa.
    * **Meccanica:** I dealer sono "Short Gamma". Se il prezzo scende, sono costretti a shortare futures, creando panico.
    * **Operatività:** Target naturale per gli short. Spesso genera un violento rimbalzo a V (*V-Bottom*) perché gli istituzionali incassano e i dealer ricomprano di colpo.
    """

    help_pcr = """
    **⚖️ PUT/CALL RATIO (Open Interest)**
    * Indica il sentiment del mercato opzionario su questa specifica scadenza.
    * Valori **> 1**: Prevalenza di Put (Sentiment difensivo/ribassista).
    * Valori **< 1**: Prevalenza di Call (Sentiment speculativo/rialzista).
    """

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"🟢 CALL WALL {nome_asset}", f"{call_wall:.0f}", help=help_call_wall)
    c2.metric(f"🟡 HVL (FLIP POINT) {nome_asset}", f"{gamma_flip:.2f}", help=help_gamma_flip)
    c3.metric(f"🔴 PUT WALL {nome_asset}", f"{put_wall:.0f}", help=help_put_wall)
    c4.metric("⚖️ P/C RATIO (OI)", f"{pcr_oi:.2f}", help=help_pcr)
    
# ==========================================
    # RENDERIZZAZIONE GRAFICO PLOTLY (OTTIMIZZATO)
    # ==========================================
    
    # 1. Creiamo liste dinamiche per far risaltare visivamente i livelli chiave
    testi_barre = []
    dimensioni_testo = []
    colori_testo = []

    for val in df_utile[colonna_y]:
        val_round = int(round(val))
        
        # Ingrandiamo e marchiamo i livelli operativi
        if val == call_wall:
            testi_barre.append(f"<b>{val_round} 🟢 CALL WALL</b>")
            dimensioni_testo.append(18)
            colori_testo.append("black")
        elif val == put_wall:
            testi_barre.append(f"<b>{val_round} 🔴 PUT WALL</b>")
            dimensioni_testo.append(18)
            colori_testo.append("black")
        else:
            testi_barre.append(f"<b>{val_round}</b>")
            dimensioni_testo.append(13) # Più piccoli per non creare confusione
            colori_testo.append("#111111") # Nero molto scuro

    fig = go.Figure()
    
    # 2. Renderizziamo le barre con il font in grassetto pesante
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

    # 3. Linee Orizzontali: layer="below" le spinge DIETRO ai testi. Aggiunto bgcolor per massima leggibilità.
    
    # Linea HVL mediatrice
    fig.add_hline(y=gamma_flip, line_dash="solid", line_color="#FFB300", line_width=4, layer="below",
                  annotation_text=f"<b>HVL (FLIP POINT): {gamma_flip:.2f}</b>", 
                  annotation_font_size=15, annotation_font_color="black",
                  annotation_bgcolor="#FFF3E0", annotation_bordercolor="#FFB300", annotation_borderpad=4,
                  annotation_position="top left")
    
    # Linea Call Wall (Annotazione a SINISTRA per non sovrapporsi al testo della barra a destra)
    fig.add_hline(y=call_wall, line_dash="dash", line_color="#32CD32", line_width=2, layer="below",
                  annotation_text=f"<b>CALL WALL: {call_wall:.0f}</b>", 
                  annotation_font_size=14, annotation_font_color="black",
                  annotation_bgcolor="#E8F5E9", annotation_borderpad=3,
                  annotation_position="top left")
                  
    # Linea Put Wall (Annotazione a DESTRA per non sovrapporsi al testo della barra a sinistra)
    fig.add_hline(y=put_wall, line_dash="dash", line_color="#FF3B30", line_width=2, layer="below",
                  annotation_text=f"<b>PUT WALL: {put_wall:.0f}</b>", 
                  annotation_font_size=14, annotation_font_color="black",
                  annotation_bgcolor="#FFEBEE", annotation_borderpad=3,
                  annotation_position="bottom right")
                  
    # Prezzo Spot Live
    fig.add_hline(y=spot_riferimento, line_color="#00FFFF", line_width=3, layer="below",
                  annotation_text=f"<b>SPOT {nome_asset}: {spot_riferimento:.2f}</b>", 
                  annotation_font_size=15, annotation_font_color="black",
                  annotation_bgcolor="#E0FFFF", annotation_bordercolor="#00FFFF", annotation_borderpad=4,
                  annotation_position="bottom right")

    # 4. Aggiorniamo il layout (Sfondo bianco per far esplodere il nero del testo)
    fig.update_layout(
        height=800, 
        template="plotly_white", # Ottimale per contrasto testi neri
        xaxis_title=f"<b>Esposizione Monetaria ({metric_col})</b>", 
        yaxis_title=f"<b>Prezzo del Sottostante ({nome_asset})</b>", 
        yaxis=dict(autorange=True, type='linear', tickfont=dict(color="black", size=12)), 
        showlegend=False,
        margin=dict(l=50, r=150, t=50, b=50) # Margine destro maggiorato per far respirare le scritte
    )
    
    st.plotly_chart(fig, use_container_width=True)
# =====================================================================
# PAGINA 2: REPLICA SITO NASDAQ CON HIGHLIGHT SPOT
# =====================================================================
elif pagina == "🗄️ Database Ufficiale":
    # Mappa titoli completi per abbellire l'intestazione
    NOMI_COMPLETI = {
        "QQQ": "Invesco QQQ Trust, Series 1",
        "SPY": "SPDR S&P 500 ETF Trust",
        "IWM": "iShares Russell 2000 ETF",
        "DIA": "SPDR Dow Jones Industrial Average ETF"
    }
    titolo_esteso = NOMI_COMPLETI.get(ticker, f"Asset: {ticker}")
    st.title(f"{titolo_esteso} ({ticker}) Option Chain")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filtro_scadenza = st.selectbox("Expiration Dates", scadenze_disponibili)
    with col2:
        filtro_strategy = st.selectbox("Strategy", ["Calls & Puts", "Calls", "Puts"])
    with col3:
        filtro_moneyness = st.selectbox("Moneyness", ["All", "Near the Money", "In the Money", "Out of the Money"], index=1)
    with col4:
        filtro_type = st.selectbox("Type", ["All (Types)", "Weekly", "Monthly", "Quarterly", "CEBO"])

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
        
        colonne_call = {
            'c_Last': 'Call Last', 'c_Change': 'Call Change', 'c_Bid': 'Call Bid', 
            'c_Ask': 'Call Ask', 'c_Volume': 'Call Volume', 'c_Openinterest': 'Call Open Int.'
        }
        colonne_put = {
            'p_Last': 'Put Last', 'p_Change': 'Put Change', 'p_Bid': 'Put Bid', 
            'p_Ask': 'Put Ask', 'p_Volume': 'Put Volume', 'p_Openinterest': 'Put Open Int.'
        }
        
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
            
        df_tabella = df_replica[colonne_finali].copy()
        
        # Sostituiamo gli zeri non scambiati con un trattino
        df_tabella = df_tabella.replace(0.0, "--")
        
        # =====================================================================
        # LOGICA DI STILE, HIGHLIGHT E FORMATTAZIONE TABELLA
        # =====================================================================
        if not df_tabella.empty:
            # 1. Trova lo strike ATM
            strike_piu_vicino = df_tabella.iloc[(df_tabella['Strike'] - spot_price_reale).abs().argsort()[:1]]['Strike'].values[0]
            
            # 2. Funzioni di formattazione del testo (2 decimali, interi, frecce)
            def formatta_base(val):
                if pd.isna(val) or val == "--": return "--"
                return f"{float(val):.2f}"
            
            def formatta_interi(val):
                if pd.isna(val) or val == "--": return "--"
                return f"{int(float(val))}"
                
            def formatta_change(val):
                if pd.isna(val) or val == "--": return "--"
                v = float(val)
                if v > 0: return f"▲ {v:.2f}"
                elif v < 0: return f"▼ {abs(v):.2f}"
                return "--"
            
            # Identificazione delle colonne per applicare la giusta formattazione
            colonne_change = [c for c in ['Call Change', 'Put Change'] if c in df_tabella.columns]
            colonne_intere = [c for c in ['Call Volume', 'Call Open Int.', 'Put Volume', 'Put Open Int.'] if c in df_tabella.columns]
            
            # Creazione del dizionario di formattazione per lo Styler
            dict_formattazione = {col: formatta_base for col in df_tabella.columns if col not in colonne_change and col not in colonne_intere}
            for col in colonne_change: dict_formattazione[col] = formatta_change
            for col in colonne_intere: dict_formattazione[col] = formatta_interi

            # 3. Funzioni per il colore (Verde/Rosso su Change, Azzurro su riga ATM)
            def colora_celle_change(s):
                stili = []
                for val in s:
                    if pd.isna(val) or val == "--":
                        stili.append("")
                    else:
                        v = float(val)
                        if v > 0: stili.append("color: #00C853; font-weight: bold;") # Verde
                        elif v < 0: stili.append("color: #FF3B30; font-weight: bold;") # Rosso
                        else: stili.append("")
                return stili
            
            def evidenzia_spot(row):
                if row['Strike'] == strike_piu_vicino:
                    return ['background-color: rgba(0, 255, 255, 0.15)'] * len(row)
                return [''] * len(row)
            
            # 4. Applicazione della pipeline di Stile Pandas
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
