"""
╔══════════════════════════════════════════════════════════════╗
║   FACEGLOSS — Dashboard Ejecutivo v2.0                       ║
║   Estructura: KPIs → Gráficos → Segmentación → Tabla        ║
╚══════════════════════════════════════════════════════════════╝

pip install streamlit plotly pandas numpy scikit-learn
streamlit run dashboard_facegloss_v2.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Facegloss · Dashboard",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS GLOBALES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Syne+Mono&family=Mulish:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0a0a0f;
    --surface:  #12121a;
    --surface2: #1a1a26;
    --border:   #22223a;
    --gold:     #d4a853;
    --gold2:    #f0c878;
    --blue:     #5b8dee;
    --green:    #3ecf8e;
    --red:      #e05c5c;
    --muted:    #6b6b8a;
    --text:     #e8e6f0;
    --text2:    #b0aec8;
}

html, body, [class*="css"] {
    font-family: 'Mulish', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background-color: var(--bg); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text2) !important; }

/* ── KPI CARD ── */
.kpi-wrap { padding: 4px; }
.kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px 16px;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.kpi-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-label {
    font-family: 'Syne Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}
.kpi-main {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-delta-pos { font-size: 0.75rem; color: var(--green); font-weight: 600; }
.kpi-delta-neg { font-size: 0.75rem; color: var(--red);   font-weight: 600; }
.kpi-sub {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 16px;
}
.kpi-sub-item { flex: 1; }
.kpi-sub-label { font-size: 0.6rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }
.kpi-sub-value { font-size: 0.85rem; font-weight: 600; color: var(--text2); }

/* ── Section header ── */
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 2px;
}
.sec-sub {
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 18px;
}

/* ── Divider ── */
.hr { border: none; border-top: 1px solid var(--border); margin: 28px 0; }

/* ── Tag pills ── */
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 6px;
}

/* ── Table ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────
@st.cache_data
def generar_datos():
    np.random.seed(42)
    catalogo = {
        'Hidratación': [('Crema Hidratante Intensiva',29.99),('Sérum Hialurónico',39.99),('Bálsamo Noche',34.99),('Gel Oil-Free',27.99)],
        'Antiedad':    [('Crema Antiedad Reafirmante',61.99),('Sérum Retinol 0.5%',54.99),('Contorno de Ojos',38.99),('Aceite Regenerador',52.99)],
        'Limpieza':    [('Espuma Limpiadora',18.99),('Agua Micelar 400ml',14.99),('Exfoliante Enzimático',22.99),('Mascarilla Purificante',19.99)],
        'Protección':  [('Protector Solar SPF50',24.99),('Sérum Vitamina C',45.99),('Tónico Equilibrante',21.99)],
    }
    canales  = ['Web','App Móvil','Email','Redes Sociales']
    ciudades = ['Barcelona','Madrid','Valencia','Sevilla','Bilbao','Zaragoza','Málaga','Murcia','Palma','Las Palmas']
    tipos = {
        'VIP':       {'peso':0.10,'fmin':8, 'fmax':20,'mult':2.0},
        'Fiel':      {'peso':0.25,'fmin':4, 'fmax':8, 'mult':1.2},
        'Normal':    {'peso':0.40,'fmin':2, 'fmax':4, 'mult':1.0},
        'En Riesgo': {'peso':0.15,'fmin':1, 'fmax':2, 'mult':0.8},
        'Dormido':   {'peso':0.10,'fmin':1, 'fmax':1, 'mult':0.6},
    }
    perfiles = np.random.choice(list(tipos.keys()), size=5000, p=[v['peso'] for v in tipos.values()])
    clientes = pd.DataFrame({
        'customer_id':    [f'CL{str(i).zfill(5)}' for i in range(1,5001)],
        'perfil':          perfiles,
        'ciudad':          np.random.choice(ciudades,5000),
        'edad':            np.random.randint(18,65,5000),
        'canal_preferido': np.random.choice(canales,5000),
    })
    fecha_ini = datetime(2022,1,1); fecha_fin = datetime(2024,12,31)
    dias = (fecha_fin - fecha_ini).days
    rows = []; oid = 100000
    for _, c in clientes.iterrows():
        t = tipos[c['perfil']]
        nc = np.random.randint(t['fmin'], t['fmax']+1)
        md = dias//3 if c['perfil']=='Dormido' else dias//2 if c['perfil']=='En Riesgo' else dias
        for _ in range(nc):
            cat = np.random.choice(list(catalogo.keys()))
            prod, precio = catalogo[cat][np.random.randint(0,len(catalogo[cat]))]
            qty  = np.random.randint(1,4)
            desc = np.random.choice([0,0.05,0.10,0.15,0.20], p=[0.6,0.15,0.1,0.1,0.05])
            imp  = round(precio * qty * t['mult'] * (1-desc), 2)
            fec  = fecha_ini + timedelta(days=int(np.random.randint(0, md)))
            rows.append({'order_id':f'ORD{oid}','customer_id':c['customer_id'],
                         'perfil_real':c['perfil'],'ciudad':c['ciudad'],'edad':c['edad'],
                         'canal':c['canal_preferido'],'fecha':fec,'categoria':cat,
                         'producto':prod,'cantidad':qty,'precio_unitario':precio,
                         'descuento_pct':desc,'importe':imp})
            oid += 1
    df = pd.DataFrame(rows).sort_values('fecha').reset_index(drop=True)
    df['año']     = df['fecha'].dt.year
    df['mes']     = df['fecha'].dt.month
    df['año_mes'] = df['fecha'].dt.to_period('M').astype(str)
    df['trimestre'] = df['fecha'].dt.to_period('Q').astype(str)
    return df

@st.cache_data
def calcular_rfm(df):
    ref = datetime(2025,1,1)
    rfm = df.groupby('customer_id').agg(
        recencia   =('fecha',    lambda x:(ref-x.max()).days),
        frecuencia =('order_id','nunique'),
        monetario  =('importe', 'sum'),
    ).reset_index()
    rfm['monetario'] = rfm['monetario'].round(2)
    rfm['R_score'] = pd.qcut(rfm['recencia'],5,labels=[5,4,3,2,1])
    rfm['F_score'] = pd.qcut(rfm['frecuencia'].rank(method='first'),5,labels=[1,2,3,4,5])
    rfm['M_score'] = pd.qcut(rfm['monetario'],5,labels=[1,2,3,4,5])
    rfm['RFM_total'] = rfm[['R_score','F_score','M_score']].astype(int).sum(axis=1)
    extra = df.groupby('customer_id').agg(
        canal_mode  =('canal',    lambda x: x.mode()[0]),
        edad        =('edad',     'first'),
        cat_fav     =('categoria',lambda x: x.mode()[0]),
    ).reset_index()
    rfm = rfm.merge(extra, on='customer_id')
    le1, le2 = LabelEncoder(), LabelEncoder()
    rfm['canal_enc'] = le1.fit_transform(rfm['canal_mode'])
    rfm['cat_enc']   = le2.fit_transform(rfm['cat_fav'])
    sc = StandardScaler()
    X  = sc.fit_transform(rfm[['recencia','frecuencia','monetario','edad','canal_enc','cat_enc']])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['segmento'] = km.fit_predict(X)
    med = rfm.groupby('segmento')['RFM_total'].mean().sort_values(ascending=False).reset_index()
    mapa = dict(zip(med['segmento'], ['VIP 🏆','Fiel 💛','En Riesgo ⚠️','Dormidos 😴']))
    rfm['segmento_nombre'] = rfm['segmento'].map(mapa)
    return rfm

df  = generar_datos()
rfm = calcular_rfm(df)

# ─────────────────────────────────────────────
# TEMA PLOTLY
# ─────────────────────────────────────────────
GOLD   = '#d4a853'
BLUE   = '#5b8dee'
GREEN  = '#3ecf8e'
RED    = '#e05c5c'
MUTED  = '#6b6b8a'
PALETTE = [GOLD, BLUE, GREEN, RED, '#a78bfa']
BG_P   = '#12121a'
GRID   = '#1e1e2e'
FONT_C = '#b0aec8'

BASE_LAYOUT = dict(
    paper_bgcolor=BG_P, plot_bgcolor=BG_P,
    font=dict(color=FONT_C, family='Mulish', size=12),
    margin=dict(l=16, r=16, t=36, b=16),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont_size=11),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont_size=11),
    legend=dict(bgcolor='rgba(0,0,0,0)', font_color=FONT_C, font_size=11),
    hoverlabel=dict(bgcolor='#1a1a26', font_color='#e8e6f0', font_size=12),
)

SEG_COL = {'VIP 🏆':GOLD,'Fiel 💛':BLUE,'En Riesgo ⚠️':'#f59e42','Dormidos 😴':MUTED}

def hex_to_rgba(hex_color, alpha=0.12):
    """Convierte hex a rgba string compatible con Plotly."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

def mini_sparkline(series, color=GOLD):
    """Genera un sparkline como figura Plotly compacta."""
    fig = go.Figure(go.Scatter(
        y=series, mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy', fillcolor=hex_to_rgba(color, 0.12),
        hoverinfo='skip'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=0,b=0), height=50,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR — FILTROS GLOBALES
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 20px;'>
        <div style='font-size:2.2rem'>💄</div>
        <div style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800;color:#e8e6f0;margin-top:4px;'>FACEGLOSS</div>
        <div style='font-size:0.6rem;color:#6b6b8a;letter-spacing:.2em;text-transform:uppercase;margin-top:2px;'>Executive Dashboard</div>
    </div>
    <hr style='border-color:#22223a;margin-bottom:20px;'>
    <div style='font-family:Syne Mono,monospace;font-size:0.6rem;letter-spacing:.15em;text-transform:uppercase;color:#6b6b8a;margin-bottom:10px;'>Filtros Globales</div>
    """, unsafe_allow_html=True)

    # Rango de fechas
    años = sorted(df['año'].unique().tolist())
    años_sel = st.multiselect("📅 Año", años, default=años)

    # Categorías
    cats = sorted(df['categoria'].unique().tolist())
    cats_sel = st.multiselect("🏷️ Categoría", cats, default=cats)

    # Ciudades
    cities = sorted(df['ciudad'].unique().tolist())
    cities_sel = st.multiselect("📍 Ciudad", cities, default=cities)

    # Canal
    canales = sorted(df['canal'].unique().tolist())
    canal_sel = st.multiselect("📡 Canal", canales, default=canales)

    # Segmento RFM
    segs = ['VIP 🏆','Fiel 💛','En Riesgo ⚠️','Dormidos 😴']
    seg_sel = st.multiselect("👥 Segmento RFM", segs, default=segs)

    st.markdown("""
    <hr style='border-color:#22223a;margin:20px 0 10px;'>
    <div style='font-size:0.6rem;color:#44445a;text-align:center;'>Proyecto IA & Big Data · 2024</div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTRADO
# ─────────────────────────────────────────────
df_f = df[
    df['año'].isin(años_sel) &
    df['categoria'].isin(cats_sel) &
    df['ciudad'].isin(cities_sel) &
    df['canal'].isin(canal_sel)
]

rfm_f = rfm[rfm['segmento_nombre'].isin(seg_sel)]

# Período anterior (para deltas)
años_ant = [a-1 for a in años_sel]
df_ant = df[
    df['año'].isin(años_ant) &
    df['categoria'].isin(cats_sel) &
    df['ciudad'].isin(cities_sel) &
    df['canal'].isin(canal_sel)
]

def delta_pct(actual, anterior):
    if anterior == 0: return 0
    return (actual - anterior) / anterior * 100

def fmt_delta(pct, suffix=""):
    arrow = "↑" if pct >= 0 else "↓"
    cls   = "kpi-delta-pos" if pct >= 0 else "kpi-delta-neg"
    return f'<span class="{cls}">{arrow} {abs(pct):.1f}%{suffix}</span>'

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:4px;'>
    <div style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;
                background:linear-gradient(135deg,#e8e6f0 30%,#d4a853);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;line-height:1;'>Dashboard Ejecutivo</div>
    <div style='font-family:Syne Mono,monospace;font-size:0.65rem;color:#6b6b8a;
                letter-spacing:.15em;text-transform:uppercase;padding-bottom:4px;'>Facegloss · v2.0</div>
</div>
<div style='font-size:0.75rem;color:#6b6b8a;letter-spacing:.1em;text-transform:uppercase;margin-bottom:24px;'>
    Análisis integral · KPIs + Ventas + Segmentación RFM + Detalle
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# BLOQUE 1 — KPIs CON SUB-MÉTRICAS
# ═══════════════════════════════════════════════════════
st.markdown('<div class="sec-title">① Métricas Clave</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-sub">Período seleccionado vs período anterior</div>', unsafe_allow_html=True)

# Cálculos principales
total_ventas   = df_f['importe'].sum()
total_ant      = df_ant['importe'].sum()
ticket_medio   = df_f['importe'].mean() if len(df_f) else 0
ticket_ant     = df_ant['importe'].mean() if len(df_ant) else 0
n_clientes_    = df_f['customer_id'].nunique()
n_cli_ant      = df_ant['customer_id'].nunique()
n_pedidos_     = df_f['order_id'].nunique()
n_ped_ant      = df_ant['order_id'].nunique()
frec_media_    = n_pedidos_ / n_clientes_ if n_clientes_ else 0
frec_ant       = (df_ant['order_id'].nunique() / df_ant['customer_id'].nunique()
                  if df_ant['customer_id'].nunique() else 0)

d_ventas  = delta_pct(total_ventas, total_ant)
d_ticket  = delta_pct(ticket_medio, ticket_ant)
d_cli     = delta_pct(n_clientes_, n_cli_ant)
d_ped     = delta_pct(n_pedidos_,  n_ped_ant)
d_frec    = delta_pct(frec_media_,  frec_ant)

# Sub-métricas para cada KPI
canal_top      = df_f.groupby('canal')['importe'].sum().idxmax() if len(df_f) else '—'
canal_top_pct  = (df_f.groupby('canal')['importe'].sum().max() / total_ventas * 100
                  if total_ventas else 0)
cat_top        = df_f.groupby('categoria')['importe'].sum().idxmax() if len(df_f) else '—'
vip_pct        = (rfm_f[rfm_f['segmento_nombre']=='VIP 🏆'].shape[0] /
                  rfm_f.shape[0] * 100 if rfm_f.shape[0] else 0)
desc_medio     = df_f['descuento_pct'].mean() * 100 if len(df_f) else 0
ciudad_top     = df_f.groupby('ciudad')['importe'].sum().idxmax() if len(df_f) else '—'
ticket_max     = df_f['importe'].max() if len(df_f) else 0

# Sparklines (ventas mensuales)
spark_data = df_f.groupby('año_mes')['importe'].sum().values if len(df_f) else [0]
spark_ped  = df_f.groupby('año_mes')['order_id'].nunique().values if len(df_f) else [0]
spark_cli  = df_f.groupby('año_mes')['customer_id'].nunique().values if len(df_f) else [0]

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-accent" style="background:linear-gradient(90deg,{GOLD},{GOLD}88);"></div>
        <div class="kpi-label">Facturación Total</div>
        <div class="kpi-main">{total_ventas/1e6:.2f}M€</div>
        {fmt_delta(d_ventas, ' vs ant.')}
        <div class="kpi-sub">
            <div class="kpi-sub-item">
                <div class="kpi-sub-label">Canal top</div>
                <div class="kpi-sub-value">{canal_top}</div>
            </div>
            <div class="kpi-sub-item">
                <div class="kpi-sub-label">% del total</div>
                <div class="kpi-sub-value">{canal_top_pct:.1f}%</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(mini_sparkline(spark_data, GOLD), use_container_width=True, config={'displayModeBar':False})

with c2:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-accent" style="background:linear-gradient(90deg,{BLUE},{BLUE}88);"></div>
        <div class="kpi-label">Ticket Medio</div>
        <div class="kpi-main">{ticket_medio:.0f}€</div>
        {fmt_delta(d_ticket, ' vs ant.')}
        <div class="kpi-sub">
            <div class="kpi-sub-item">
                <div class="kpi-sub-label">Ticket máx.</div>
                <div class="kpi-sub-value">{ticket_max:.0f}€</div>
            </div>
            <div class="kpi-sub-item">
                <div class="kpi-sub-label">Dto. medio</div>
                <div class="kpi-sub-value">{desc_medio:.1f}%</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(mini_sparkline(spark_data / np.maximum(spark_cli,1), BLUE), use_container_width=True, config={'displayModeBar':False})

with c3:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-accent" style="background:linear-gradient(90deg,{GREEN},{GREEN}88);"></div>
        <div class="kpi-label">Clientes Activos</div>
        <div class="kpi-main">{n_clientes_:,}</div>
        {fmt_delta(d_cli, ' vs ant.')}
        <div class="kpi-sub">
            <div class="kpi-sub-item">
                <div class="kpi-sub-label">% VIP</div>
                <div class="kpi-sub-value">{vip_pct:.1f}%</div>
            </div>
            <div class="kpi-sub-item">
                <div class="kpi-sub-label">Ciudad top</div>
                <div class="kpi-sub-value">{ciudad_top}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(mini_sparkline(spark_cli, GREEN), use_container_width=True, config={'displayModeBar':False})

with c4:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-accent" style="background:linear-gradient(90deg,#a78bfa,#a78bfa88);"></div>
        <div class="kpi-label">Total Pedidos</div>
        <div class="kpi-main">{n_pedidos_:,}</div>
        {fmt_delta(d_ped, ' vs ant.')}
        <div class="kpi-sub">
            <div class="kpi-sub-item">
                <div class="kpi-sub-label">Cat. top</div>
                <div class="kpi-sub-value">{cat_top}</div>
            </div>
            <div class="kpi-sub-item">
                <div class="kpi-sub-label">Dto. total</div>
                <div class="kpi-sub-value">{df_f["descuento_pct"].gt(0).mean()*100:.0f}% ped.</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(mini_sparkline(spark_ped, '#a78bfa'), use_container_width=True, config={'displayModeBar':False})

with c5:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-accent" style="background:linear-gradient(90deg,{RED},{RED}88);"></div>
        <div class="kpi-label">Frecuencia Media</div>
        <div class="kpi-main">{frec_media_:.1f}x</div>
        {fmt_delta(d_frec, ' vs ant.')}
        <div class="kpi-sub">
            <div class="kpi-sub-item">
                <div class="kpi-sub-label">VIP frec.</div>
                <div class="kpi-sub-value">{rfm_f[rfm_f['segmento_nombre']=='VIP 🏆']['frecuencia'].mean():.1f}x</div>
            </div>
            <div class="kpi-sub-item">
                <div class="kpi-sub-label">Dormidos</div>
                <div class="kpi-sub-value">{rfm_f[rfm_f['segmento_nombre']=='Dormidos 😴']['frecuencia'].mean():.1f}x</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(mini_sparkline(spark_ped / np.maximum(spark_cli,1), RED), use_container_width=True, config={'displayModeBar':False})

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# BLOQUE 2 — GRÁFICOS PRINCIPALES
# ═══════════════════════════════════════════════════════
st.markdown('<div class="sec-title">② Análisis de Ventas</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-sub">Evolución · Comparativas · Proporciones</div>', unsafe_allow_html=True)

# Tabs de granularidad temporal
tab_mes, tab_trim, tab_año = st.tabs(["📅 Mensual", "📆 Trimestral", "📊 Anual"])

with tab_mes:
    ventas_mes = df_f.groupby(['año_mes','canal'])['importe'].sum().reset_index()
    ventas_mes_total = df_f.groupby('año_mes')['importe'].sum().reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for i, canal in enumerate(df_f['canal'].unique()):
        sub = ventas_mes[ventas_mes['canal']==canal]
        fig.add_trace(go.Bar(
            name=canal, x=sub['año_mes'], y=sub['importe'],
            marker_color=PALETTE[i],
            hovertemplate='<b>%{x}</b><br>%{y:,.0f}€<extra>'+canal+'</extra>'
        ))
    fig.add_trace(go.Scatter(
        name='Total', x=ventas_mes_total['año_mes'], y=ventas_mes_total['importe'],
        mode='lines+markers', line=dict(color='white',width=2,dash='dot'),
        marker=dict(size=4), yaxis='y2',
        hovertemplate='<b>%{x}</b><br>Total: %{y:,.0f}€<extra></extra>'
    ), secondary_y=True)
    fig.update_layout(**BASE_LAYOUT, height=360, barmode='stack',
                      xaxis_tickangle=-45, title='Facturación mensual por canal')
    fig.update_yaxes(gridcolor=GRID, secondary_y=False)
    fig.update_yaxes(showgrid=False, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

with tab_trim:
    vt = df_f.groupby(['trimestre','categoria'])['importe'].sum().reset_index()
    fig = px.bar(vt, x='trimestre', y='importe', color='categoria',
                 barmode='group', color_discrete_sequence=PALETTE,
                 labels={'importe':'€','trimestre':'','categoria':''},
                 hover_data={'importe':':,.0f'})
    fig.update_layout(**BASE_LAYOUT, height=360, title='Facturación trimestral por categoría')
    st.plotly_chart(fig, use_container_width=True)

with tab_año:
    va = df_f.groupby(['año','categoria'])['importe'].sum().reset_index()
    fig = px.bar(va, x='categoria', y='importe', color='año',
                 barmode='group', color_discrete_sequence=PALETTE,
                 labels={'importe':'€','categoria':'','año':'Año'},
                 hover_data={'importe':':,.0f'})
    fig.update_layout(**BASE_LAYOUT, height=360, title='Comparativa interanual por categoría')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Fila 2: Barras horizontales + Pie + Heatmap
col_a, col_b, col_c = st.columns([1.4, 1, 1.6])

with col_a:
    st.markdown('<div class="sec-title" style="font-size:1rem">Top 10 Productos</div>', unsafe_allow_html=True)
    tp = (df_f.groupby('producto')
              .agg(facturacion=('importe','sum'), pedidos=('order_id','nunique'))
              .sort_values('facturacion', ascending=False).head(10).reset_index())
    fig = go.Figure(go.Bar(
        y=tp['producto'], x=tp['facturacion'], orientation='h',
        marker=dict(color=tp['facturacion'],
                    colorscale=[[0,'#1a1a26'],[0.5,'#7a5c20'],[1,GOLD]],
                    showscale=False),
        customdata=tp['pedidos'],
        hovertemplate='<b>%{y}</b><br>Facturación: %{x:,.0f}€<br>Pedidos: %{customdata:,}<extra></extra>'
    ))
    fig.update_layout(**BASE_LAYOUT, height=340, showlegend=False,
                      title='Ranking por facturación',
                      xaxis_title='€', yaxis_title='')
    fig.update_yaxes(gridcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown('<div class="sec-title" style="font-size:1rem">Mix Canales</div>', unsafe_allow_html=True)
    cv = df_f.groupby('canal')['importe'].sum().reset_index()
    fig = go.Figure(go.Pie(
        labels=cv['canal'], values=cv['importe'],
        hole=0.6, marker_colors=PALETTE,
        textinfo='percent', textfont_size=11,
        hovertemplate='<b>%{label}</b><br>%{value:,.0f}€<br>%{percent}<extra></extra>',
        pull=[0.04,0,0,0],
    ))
    fig.update_layout(**BASE_LAYOUT, height=340, showlegend=True,
                      title='Distribución por canal')
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    fig.update_layout(legend=dict(orientation='v', x=1, y=0.5))
    st.plotly_chart(fig, use_container_width=True)

with col_c:
    st.markdown('<div class="sec-title" style="font-size:1rem">Heatmap Categoría × Canal</div>', unsafe_allow_html=True)
    piv = df_f.pivot_table(values='importe', index='categoria', columns='canal', aggfunc='sum').fillna(0)
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
        colorscale=[[0,'#0a0a0f'],[0.4,'#2a1f0d'],[1,GOLD]],
        text=[[f'{v/1000:.0f}k€' for v in row] for row in piv.values],
        texttemplate='%{text}',
        hovertemplate='<b>%{y} × %{x}</b><br>%{z:,.0f}€<extra></extra>',
        showscale=True, colorbar=dict(tickfont_color=FONT_C, len=0.8)
    ))
    fig.update_layout(**BASE_LAYOUT, height=340, title='Facturación total (€)')
    fig.update_xaxes(side='bottom', tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# BLOQUE 3 — SEGMENTACIÓN RFM
# ═══════════════════════════════════════════════════════
st.markdown('<div class="sec-title">③ Segmentación RFM + ML</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-sub">K-Means · 4 clusters · Features: Recencia, Frecuencia, Monetario, Edad, Canal, Categoría</div>', unsafe_allow_html=True)

# Sub-KPIs de segmento
seg_kpis = rfm_f.groupby('segmento_nombre').agg(
    n      = ('customer_id','count'),
    rec    = ('recencia','mean'),
    frec   = ('frecuencia','mean'),
    mon    = ('monetario','mean'),
).round(1).reset_index()

colors_seg_bar = [SEG_COL.get(s, MUTED) for s in seg_kpis['segmento_nombre']]

sc1, sc2, sc3, sc4 = st.columns(4)
for col_, (_, row) in zip([sc1,sc2,sc3,sc4], seg_kpis.iterrows()):
    color = SEG_COL.get(row['segmento_nombre'], MUTED)
    pct_total = row['n'] / rfm_f.shape[0] * 100 if rfm_f.shape[0] else 0
    with col_:
        st.markdown(f"""
        <div class="kpi" style="border-color:{color}33;">
            <div class="kpi-accent" style="background:{color};"></div>
            <div class="kpi-label">{row['segmento_nombre']}</div>
            <div class="kpi-main" style="color:{color};font-size:1.7rem;">{int(row['n']):,}</div>
            <div style="font-size:0.7rem;color:{MUTED};margin-top:2px;">{pct_total:.1f}% del total</div>
            <div class="kpi-sub">
                <div class="kpi-sub-item">
                    <div class="kpi-sub-label">Recencia</div>
                    <div class="kpi-sub-value">{row['rec']:.0f} días</div>
                </div>
                <div class="kpi-sub-item">
                    <div class="kpi-sub-label">Frec.</div>
                    <div class="kpi-sub-value">{row['frec']:.1f}x</div>
                </div>
                <div class="kpi-sub-item">
                    <div class="kpi-sub-label">Gasto</div>
                    <div class="kpi-sub-value">{row['mon']:.0f}€</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# Scatter + Radar
r1, r2 = st.columns([3, 2])

with r1:
    seg_elegido = st.selectbox(
        "🔍 Resaltar segmento en scatter",
        ["Todos"] + segs,
        key="scatter_seg"
    )
    scatter_df = rfm_f.copy()
    if seg_elegido != "Todos":
        scatter_df['_alpha'] = np.where(scatter_df['segmento_nombre'] == seg_elegido, 0.85, 0.08)
    else:
        scatter_df['_alpha'] = 0.5

    fig = go.Figure()
    for seg, col_ in SEG_COL.items():
        sub = scatter_df[scatter_df['segmento_nombre'] == seg]
        alpha = 0.85 if (seg_elegido == "Todos" or seg_elegido == seg) else 0.08
        fig.add_trace(go.Scatter(
            x=sub['frecuencia'], y=sub['monetario'],
            mode='markers', name=seg,
            marker=dict(color=col_, size=5, opacity=alpha),
            hovertemplate=(
                f'<b>{seg}</b><br>'
                'Frecuencia: %{x}<br>'
                'Gasto total: %{y:,.0f}€<br>'
                'Recencia: %{customdata} días<extra></extra>'
            ),
            customdata=sub['recencia'],
        ))
    fig.update_layout(**BASE_LAYOUT, height=380,
                      title='Frecuencia vs Gasto total · cada punto = 1 cliente',
                      xaxis_title='Frecuencia (nº pedidos)',
                      yaxis_title='Gasto total (€)')
    st.plotly_chart(fig, use_container_width=True)

with r2:
    radar_data = rfm_f.groupby('segmento_nombre')[['recencia','frecuencia','monetario']].mean()
    if len(radar_data) > 0:
        rn = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min()).replace(0,1)
        rn['recencia'] = 1 - rn['recencia']
        rn.columns = ['Actualidad','Frecuencia','Valor monetario']
        cats_radar = list(rn.columns) + [rn.columns[0]]
        fig = go.Figure()
        for seg, col_ in SEG_COL.items():
            if seg in rn.index:
                v = list(rn.loc[seg]) + [rn.loc[seg].iloc[0]]
                fig.add_trace(go.Scatterpolar(
                    r=v, theta=cats_radar, name=seg,
                    fill='toself', fillcolor=hex_to_rgba(col_, 0.1),
                    line=dict(color=col_, width=2),
                    hovertemplate='<b>%{theta}</b>: %{r:.2f}<extra>'+seg+'</extra>'
                ))
        fig.update_layout(
            paper_bgcolor=BG_P, plot_bgcolor=BG_P,
            font=dict(color=FONT_C, family='Mulish'),
            polar=dict(bgcolor=BG_P,
                       radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID, color=MUTED, tickfont_size=9),
                       angularaxis=dict(gridcolor=GRID, color=FONT_C)),
            height=380, showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)', font_color=FONT_C, font_size=10, x=1),
            margin=dict(l=30,r=80,t=40,b=30),
            title=dict(text='Perfil RFM por segmento', font_color=FONT_C),
        )
        st.plotly_chart(fig, use_container_width=True)

# Validación heatmap
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown('<div class="sec-title" style="font-size:1rem">Validación ML vs Perfil Real</div>', unsafe_allow_html=True)

validacion = df.groupby('customer_id')['perfil_real'].first().reset_index()
rfm_val = rfm_f.merge(validacion, on='customer_id')
if len(rfm_val) > 0:
    tabla_val = pd.crosstab(rfm_val['segmento_nombre'], rfm_val['perfil_real'],
                            normalize='index').round(3) * 100
    fig = go.Figure(go.Heatmap(
        z=tabla_val.values,
        x=tabla_val.columns.tolist(),
        y=tabla_val.index.tolist(),
        colorscale=[[0,'#0a0a0f'],[0.5,'#1f2a10'],[1,GREEN]],
        text=[[f'{v:.1f}%' for v in row] for row in tabla_val.values],
        texttemplate='%{text}',
        hovertemplate='<b>ML: %{y}</b><br>Perfil real: %{x}<br>%{z:.1f}%<extra></extra>',
        showscale=True, colorbar=dict(tickfont_color=FONT_C, len=0.8)
    ))
    fig.update_layout(**BASE_LAYOUT, height=280,
                      title='% de perfiles reales dentro de cada segmento ML')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# BLOQUE 4 — TABLA DE DETALLE
# ═══════════════════════════════════════════════════════
st.markdown('<div class="sec-title">④ Detalle de Transacciones</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-sub">Vista granular con filtros combinados</div>', unsafe_allow_html=True)

col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    search_prod = st.text_input("🔎 Buscar producto", placeholder="ej: Sérum...")
with col_t2:
    order_by = st.selectbox("⬇️ Ordenar por", ['importe','cantidad','descuento_pct','fecha'])
with col_t3:
    top_n = st.slider("Nº de filas", 10, 200, 50, step=10)

# Aplicar búsqueda
df_tabla = df_f.copy()
if search_prod:
    df_tabla = df_tabla[df_tabla['producto'].str.contains(search_prod, case=False, na=False)]

df_tabla = df_tabla.sort_values(order_by, ascending=False).head(top_n)

# Merge con segmento
rfm_seg = rfm[['customer_id','segmento_nombre']]
df_tabla = df_tabla.merge(rfm_seg, on='customer_id', how='left')

# Formato
df_show = df_tabla[[
    'fecha','order_id','customer_id','segmento_nombre',
    'ciudad','canal','categoria','producto',
    'cantidad','precio_unitario','descuento_pct','importe'
]].copy()
df_show['fecha']         = df_show['fecha'].dt.strftime('%Y-%m-%d')
df_show['descuento_pct'] = (df_show['descuento_pct']*100).round(0).astype(int).astype(str) + '%'
df_show['importe']       = df_show['importe'].apply(lambda x: f'{x:,.2f}€')
df_show['precio_unitario'] = df_show['precio_unitario'].apply(lambda x: f'{x:.2f}€')
df_show.columns = ['Fecha','Pedido','Cliente','Segmento','Ciudad','Canal',
                   'Categoría','Producto','Uds.','Precio unit.','Descuento','Importe']

st.dataframe(
    df_show,
    use_container_width=True,
    height=420,
    hide_index=True,
)

# Resumen de la tabla
col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    st.markdown(f"<div style='font-size:0.75rem;color:{MUTED};'>📋 Mostrando <b style='color:{FONT_C}'>{len(df_tabla):,}</b> registros</div>", unsafe_allow_html=True)
with col_r2:
    total_tabla = df_f.groupby('producto')['importe'].sum()
    prod_top_tabla = total_tabla.idxmax() if len(total_tabla) else '—'
    st.markdown(f"<div style='font-size:0.75rem;color:{MUTED};'>🏆 Producto top: <b style='color:{GOLD}'>{prod_top_tabla}</b></div>", unsafe_allow_html=True)
with col_r3:
    facturacion_total = df_f['importe'].sum()
    st.markdown(f"<div style='font-size:0.75rem;color:{MUTED};'>💶 Facturación mostrada: <b style='color:{GREEN}'>{facturacion_total:,.0f}€</b></div>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;margin-top:40px;font-family:Syne Mono,monospace;
            font-size:0.6rem;color:#2a2a3a;letter-spacing:.1em;'>
    FACEGLOSS EXECUTIVE DASHBOARD v2.0 · PROYECTO IA & BIG DATA · 2024
</div>
""", unsafe_allow_html=True)
