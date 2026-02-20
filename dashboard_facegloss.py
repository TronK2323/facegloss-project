"""
FACEGLOSS — Dashboard Ejecutivo v3.0
Navegación: Resumen · Ventas · Segmentación · Recomendador · Tabla
pip install streamlit plotly pandas numpy scikit-learn
streamlit run dashboard_facegloss_final.py
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Syne+Mono&family=Mulish:wght@300;400;500;600&display=swap');
:root {
    --bg:#0a0a0f; --surface:#12121a; --border:#22223a;
    --gold:#d4a853; --blue:#5b8dee; --green:#3ecf8e;
    --red:#e05c5c; --muted:#6b6b8a; --text:#e8e6f0; --text2:#b0aec8;
}
html,body,[class*="css"] { font-family:'Mulish',sans-serif; background:var(--bg); color:var(--text); }
.stApp { background:var(--bg); }
[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color:var(--text2) !important; }
.kpi { background:var(--surface); border:1px solid var(--border); border-radius:14px;
       padding:20px 22px 16px; position:relative; overflow:hidden; height:100%; }
.kpi-accent { position:absolute; top:0; left:0; right:0; height:3px; }
.kpi-label { font-family:'Syne Mono',monospace; font-size:0.65rem; letter-spacing:0.15em;
             text-transform:uppercase; color:var(--muted); margin-bottom:10px; }
.kpi-main { font-family:'Syne',sans-serif; font-size:2rem; font-weight:700;
            color:var(--text); line-height:1; margin-bottom:6px; }
.kpi-delta-pos { font-size:0.75rem; color:var(--green); font-weight:600; }
.kpi-delta-neg { font-size:0.75rem; color:var(--red); font-weight:600; }
.kpi-sub { margin-top:14px; padding-top:12px; border-top:1px solid var(--border); display:flex; gap:16px; }
.kpi-sub-item { flex:1; }
.kpi-sub-label { font-size:0.6rem; color:var(--muted); letter-spacing:0.1em; text-transform:uppercase; }
.kpi-sub-value { font-size:0.85rem; font-weight:600; color:var(--text2); }
.sec-title { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700;
             color:var(--text); margin-bottom:2px; }
.sec-sub { font-size:0.72rem; color:var(--muted); letter-spacing:0.12em;
           text-transform:uppercase; margin-bottom:18px; }
.hr { border:none; border-top:1px solid var(--border); margin:28px 0; }
.stDataFrame { border-radius:12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────
@st.cache_data
def generar_datos():
    np.random.seed(42)
    catalogo = {
        'Hidratación': [('Crema Hidratante Intensiva',29.99),('Sérum Hialurónico',39.99),
                        ('Bálsamo Noche',34.99),('Gel Oil-Free',27.99)],
        'Antiedad':    [('Crema Antiedad Reafirmante',61.99),('Sérum Retinol 0.5%',54.99),
                        ('Contorno de Ojos',38.99),('Aceite Regenerador',52.99)],
        'Limpieza':    [('Espuma Limpiadora',18.99),('Agua Micelar 400ml',14.99),
                        ('Exfoliante Enzimático',22.99),('Mascarilla Purificante',19.99)],
        'Protección':  [('Protector Solar SPF50',24.99),('Sérum Vitamina C',45.99),
                        ('Tónico Equilibrante',21.99)],
    }
    canales  = ['Web','App Móvil','Email','Redes Sociales']
    ciudades = ['Barcelona','Madrid','Valencia','Sevilla','Bilbao',
                'Zaragoza','Málaga','Murcia','Palma','Las Palmas']
    tipos = {
        'VIP':       {'peso':0.10,'fmin':8, 'fmax':20,'mult':2.0},
        'Fiel':      {'peso':0.25,'fmin':4, 'fmax':8, 'mult':1.2},
        'Normal':    {'peso':0.40,'fmin':2, 'fmax':4, 'mult':1.0},
        'En Riesgo': {'peso':0.15,'fmin':1, 'fmax':2, 'mult':0.8},
        'Dormido':   {'peso':0.10,'fmin':1, 'fmax':1, 'mult':0.6},
    }
    perfiles = np.random.choice(list(tipos.keys()), size=5000,
                                p=[v['peso'] for v in tipos.values()])
    clientes = pd.DataFrame({
        'customer_id':    [f'CL{str(i).zfill(5)}' for i in range(1,5001)],
        'perfil':          perfiles,
        'ciudad':          np.random.choice(ciudades,5000),
        'edad':            np.random.randint(18,65,5000),
        'canal_preferido': np.random.choice(canales,5000),
    })
    fecha_ini = datetime(2022,1,1)
    fecha_fin = datetime(2024,12,31)
    dias = (fecha_fin - fecha_ini).days
    rows = []; oid = 100000
    for _, c in clientes.iterrows():
        t  = tipos[c['perfil']]
        nc = np.random.randint(t['fmin'], t['fmax']+1)
        md = dias//3 if c['perfil']=='Dormido' else dias//2 if c['perfil']=='En Riesgo' else dias
        for _ in range(nc):
            cat  = np.random.choice(list(catalogo.keys()))
            prod, precio = catalogo[cat][np.random.randint(0,len(catalogo[cat]))]
            qty  = np.random.randint(1,4)
            desc = np.random.choice([0,0.05,0.10,0.15,0.20],p=[0.6,0.15,0.1,0.1,0.05])
            imp  = round(precio*qty*t['mult']*(1-desc),2)
            fec  = fecha_ini + timedelta(days=int(np.random.randint(0,md)))
            rows.append({'order_id':f'ORD{oid}','customer_id':c['customer_id'],
                         'perfil_real':c['perfil'],'ciudad':c['ciudad'],'edad':c['edad'],
                         'canal':c['canal_preferido'],'fecha':fec,'categoria':cat,
                         'producto':prod,'cantidad':qty,'precio_unitario':precio,
                         'descuento_pct':desc,'importe':imp})
            oid += 1
    df = pd.DataFrame(rows).sort_values('fecha').reset_index(drop=True)
    df['año']      = df['fecha'].dt.year
    df['mes']      = df['fecha'].dt.month
    df['año_mes']  = df['fecha'].dt.to_period('M').astype(str)
    df['trimestre']= df['fecha'].dt.to_period('Q').astype(str)
    return df

@st.cache_data
def calcular_rfm(df):
    ref = datetime(2025,1,1)
    rfm = df.groupby('customer_id').agg(
        recencia  =('fecha',    lambda x:(ref-x.max()).days),
        frecuencia=('order_id','nunique'),
        monetario =('importe', 'sum'),
    ).reset_index()
    rfm['monetario'] = rfm['monetario'].round(2)
    rfm['R_score']   = pd.qcut(rfm['recencia'],5,labels=[5,4,3,2,1])
    rfm['F_score']   = pd.qcut(rfm['frecuencia'].rank(method='first'),5,labels=[1,2,3,4,5])
    rfm['M_score']   = pd.qcut(rfm['monetario'],5,labels=[1,2,3,4,5])
    rfm['RFM_total'] = rfm[['R_score','F_score','M_score']].astype(int).sum(axis=1)
    extra = df.groupby('customer_id').agg(
        canal_mode=('canal',    lambda x:x.mode()[0]),
        edad      =('edad',     'first'),
        cat_fav   =('categoria',lambda x:x.mode()[0]),
    ).reset_index()
    rfm = rfm.merge(extra, on='customer_id')
    le1,le2 = LabelEncoder(), LabelEncoder()
    rfm['canal_enc'] = le1.fit_transform(rfm['canal_mode'])
    rfm['cat_enc']   = le2.fit_transform(rfm['cat_fav'])
    sc = StandardScaler()
    X  = sc.fit_transform(rfm[['recencia','frecuencia','monetario','edad','canal_enc','cat_enc']])
    km = KMeans(n_clusters=4,random_state=42,n_init=10)
    rfm['segmento'] = km.fit_predict(X)
    med  = rfm.groupby('segmento')['RFM_total'].mean().sort_values(ascending=False).reset_index()
    mapa = dict(zip(med['segmento'],['VIP 🏆','Fiel 💛','En Riesgo ⚠️','Dormidos 😴']))
    rfm['segmento_nombre'] = rfm['segmento'].map(mapa)
    return rfm

@st.cache_data
def construir_motor(df, rfm):
    catalogo_tags = {
        'Crema Hidratante Intensiva':'hidratacion crema piel seca nutritiva',
        'Sérum Hialurónico':         'hidratacion serum hialuronico antiedad',
        'Bálsamo Noche':             'hidratacion noche nutritivo reparador',
        'Gel Oil-Free':              'hidratacion gel ligero piel grasa',
        'Crema Antiedad Reafirmante':'antiedad crema reafirmante arrugas colageno',
        'Sérum Retinol 0.5%':        'antiedad serum retinol renovacion celular',
        'Contorno de Ojos':          'antiedad contorno ojos ojeras bolsas',
        'Aceite Regenerador':        'antiedad aceite regenerador luminosidad',
        'Espuma Limpiadora':         'limpieza espuma suave desmaquillante',
        'Agua Micelar 400ml':        'limpieza micelar desmaquillante suave',
        'Exfoliante Enzimático':     'limpieza exfoliante enzimas renovacion',
        'Mascarilla Purificante':    'limpieza mascarilla poros purificante',
        'Protector Solar SPF50':     'proteccion solar spf50 uva uvb',
        'Sérum Vitamina C':          'proteccion vitamina antioxidante luminosidad',
        'Tónico Equilibrante':       'proteccion tonico equilibrante ph piel',
    }
    rows_cat = []
    for prod in df['producto'].unique():
        cat    = df[df['producto']==prod]['categoria'].iloc[0]
        precio = df[df['producto']==prod]['precio_unitario'].iloc[0]
        tags   = catalogo_tags.get(prod, cat.lower())
        rows_cat.append({'producto':prod,'categoria':cat,'precio':precio,'tags':tags})
    df_cat = pd.DataFrame(rows_cat).drop_duplicates('producto').reset_index(drop=True)

    tfidf     = TfidfVectorizer()
    tfidf_mat = tfidf.fit_transform(df_cat['tags'])
    sim_prods = cosine_similarity(tfidf_mat)
    sim_prod_df = pd.DataFrame(sim_prods, index=df_cat['producto'], columns=df_cat['producto'])

    matriz_cp = df.pivot_table(index='customer_id',columns='producto',
                               values='importe',aggfunc='sum',fill_value=0)
    muestra   = matriz_cp.sample(n=min(800,len(matriz_cp)),random_state=42)
    sim_cli   = cosine_similarity(muestra)
    sim_cli_df = pd.DataFrame(sim_cli,index=muestra.index,columns=muestra.index)

    df_seg = df.merge(rfm[['customer_id','segmento_nombre']],on='customer_id')
    rec_seg = (df_seg.groupby(['segmento_nombre','producto'])
               .agg(facturacion=('importe','sum'),clientes_unicos=('customer_id','nunique'))
               .reset_index())
    rec_seg['score'] = (
        rec_seg['facturacion']/rec_seg.groupby('segmento_nombre')['facturacion'].transform('max')*0.6 +
        rec_seg['clientes_unicos']/rec_seg.groupby('segmento_nombre')['clientes_unicos'].transform('max')*0.4
    )
    top_seg = (rec_seg.sort_values(['segmento_nombre','score'],ascending=[True,False])
               .groupby('segmento_nombre').head(8).reset_index(drop=True))
    return df_cat, sim_prod_df, sim_cli_df, muestra, top_seg

def recomendar_hibrido(customer_id, rfm, df, df_cat, sim_prod_df, sim_cli_df, muestra, top_seg, top_n=5):
    todos  = df_cat['producto'].tolist()
    scores = {p:0.0 for p in todos}
    info   = rfm[rfm['customer_id']==customer_id]
    seg    = info.iloc[0]['segmento_nombre'] if not info.empty else 'Normal'
    comprados = set(df[df['customer_id']==customer_id]['producto'].unique())
    w_seg,w_col,w_cont = (0.60,0.25,0.15) if seg in ['Dormidos 😴','En Riesgo ⚠️'] else (0.40,0.40,0.20)

    seg_recs = top_seg[top_seg['segmento_nombre']==seg]
    maxs = seg_recs['score'].max() if len(seg_recs) else 1
    for _,row in seg_recs.iterrows():
        if row['producto'] not in comprados and row['producto'] in scores:
            scores[row['producto']] += w_seg*(row['score']/maxs)

    if customer_id in sim_cli_df.index:
        similares = sim_cli_df[customer_id].drop(customer_id,errors='ignore').nlargest(10)
        csc = {}
        for sid,sv in similares.items():
            if sid in muestra.index:
                for prod,gasto in muestra.loc[sid].items():
                    if gasto>0 and prod not in comprados:
                        csc[prod] = csc.get(prod,0)+gasto*sv
        if csc:
            maxc = max(csc.values())
            for prod,sc in csc.items():
                if prod in scores: scores[prod] += w_col*(sc/maxc)

    for pc in list(comprados)[:3]:
        if pc in sim_prod_df.index:
            sims = sim_prod_df[pc].drop(pc,errors='ignore').nlargest(3)
            for i,(ps,_) in enumerate(sims.items()):
                if ps not in comprados and ps in scores:
                    scores[ps] += w_cont*(1/(i+1))

    resultado = (pd.DataFrame.from_dict(scores,orient='index',columns=['score'])
                 .reset_index().rename(columns={'index':'producto'})
                 .merge(df_cat[['producto','categoria','precio']],on='producto')
                 .sort_values('score',ascending=False)
                 .query('score>0').head(top_n).reset_index(drop=True))
    resultado.index += 1
    return resultado, seg

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
df  = generar_datos()
rfm = calcular_rfm(df)

# ─────────────────────────────────────────────
# TEMA PLOTLY
# ─────────────────────────────────────────────
GOLD    = '#d4a853'
BLUE    = '#5b8dee'
GREEN   = '#3ecf8e'
RED     = '#e05c5c'
MUTED   = '#6b6b8a'
PALETTE = [GOLD, BLUE, GREEN, RED, '#a78bfa']
BG_P    = '#12121a'
GRID    = '#1e1e2e'
FONT_C  = '#b0aec8'
SEG_COL = {'VIP 🏆':GOLD,'Fiel 💛':BLUE,'En Riesgo ⚠️':'#f59e42','Dormidos 😴':MUTED}
CAT_COL = {'Hidratación':BLUE,'Antiedad':GOLD,'Limpieza':GREEN,'Protección':'#a78bfa'}

BASE_LAYOUT = dict(
    paper_bgcolor=BG_P, plot_bgcolor=BG_P,
    font=dict(color=FONT_C, family='Mulish', size=12),
    margin=dict(l=16,r=16,t=36,b=16),
    xaxis=dict(gridcolor=GRID,zerolinecolor=GRID,tickfont_size=11),
    yaxis=dict(gridcolor=GRID,zerolinecolor=GRID,tickfont_size=11),
    legend=dict(bgcolor='rgba(0,0,0,0)',font_color=FONT_C,font_size=11),
    hoverlabel=dict(bgcolor='#1a1a26',font_color='#e8e6f0',font_size=12),
)

def hex_to_rgba(hex_color, alpha=0.12):
    h = hex_color.lstrip('#')
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

def mini_sparkline(series, color=GOLD):
    fig = go.Figure(go.Scatter(
        y=series, mode='lines',
        line=dict(color=color,width=2),
        fill='tozeroy', fillcolor=hex_to_rgba(color,0.12),
        hoverinfo='skip'
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=0,r=0,t=0,b=0),height=50,
                      xaxis=dict(visible=False),yaxis=dict(visible=False),showlegend=False)
    return fig

def fmt_delta(pct, suffix=""):
    arrow = "↑" if pct>=0 else "↓"
    cls   = "kpi-delta-pos" if pct>=0 else "kpi-delta-neg"
    return f'<span class="{cls}">{arrow} {abs(pct):.1f}%{suffix}</span>'

def delta_pct(a,b):
    return (a-b)/b*100 if b else 0

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 20px;'>
        <div style='font-size:2.2rem'>💄</div>
        <div style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800;color:#e8e6f0;margin-top:4px;'>FACEGLOSS</div>
        <div style='font-size:0.6rem;color:#6b6b8a;letter-spacing:.2em;text-transform:uppercase;margin-top:2px;'>Executive Dashboard v3.0</div>
    </div>
    <hr style='border-color:#22223a;margin-bottom:20px;'>
    """, unsafe_allow_html=True)

    seccion = st.radio("Nav", [
        "📊 Resumen General",
        "🛍️ Ventas & Productos",
        "👥 Segmentación RFM",
        "🤖 Recomendador",
        "📋 Tabla de Detalle",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#22223a;margin:20px 0 12px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.6rem;color:#6b6b8a;letter-spacing:.15em;text-transform:uppercase;margin-bottom:10px;'>Filtros Globales</div>", unsafe_allow_html=True)

    años_disp = sorted(df['año'].unique().tolist())
    años_sel  = st.multiselect("Año", años_disp, default=años_disp)
    cats_disp = sorted(df['categoria'].unique().tolist())
    cats_sel  = st.multiselect("Categoría", cats_disp, default=cats_disp)
    city_disp = sorted(df['ciudad'].unique().tolist())
    city_sel  = st.multiselect("Ciudad", city_disp, default=city_disp)
    can_disp  = sorted(df['canal'].unique().tolist())
    can_sel   = st.multiselect("Canal", can_disp, default=can_disp)
    segs_all  = ['VIP 🏆','Fiel 💛','En Riesgo ⚠️','Dormidos 😴']
    seg_sel   = st.multiselect("Segmento RFM", segs_all, default=segs_all)

    st.markdown("<hr style='border-color:#22223a;margin:16px 0 8px;'><div style='font-size:0.6rem;color:#333;text-align:center;'>Proyecto IA & Big Data · 2024</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTRADO
# ─────────────────────────────────────────────
df_f = df[df['año'].isin(años_sel) & df['categoria'].isin(cats_sel) &
          df['ciudad'].isin(city_sel) & df['canal'].isin(can_sel)]
rfm_f = rfm[rfm['segmento_nombre'].isin(seg_sel)]
df_ant = df[df['año'].isin([a-1 for a in años_sel]) & df['categoria'].isin(cats_sel) &
            df['ciudad'].isin(city_sel) & df['canal'].isin(can_sel)]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
TITULOS = {
    "📊 Resumen General":   ("Dashboard Ejecutivo",        "KPIs · Ventas · Segmentación · Tendencias"),
    "🛍️ Ventas & Productos":("Ventas & Productos",          "Rendimiento por producto, canal y categoría"),
    "👥 Segmentación RFM":  ("Segmentación RFM + ML",       "K-Means · 4 clusters · Features enriquecidas"),
    "🤖 Recomendador":      ("Recomendador de Productos",   "Sistema híbrido · Segmento + Colaborativo + Contenido"),
    "📋 Tabla de Detalle":  ("Tabla de Transacciones",      "Vista granular con filtros combinados"),
}
titulo, subtitulo = TITULOS.get(seccion, ("Dashboard",""))
st.markdown(
    f'<div style="font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;'
    f'background:linear-gradient(135deg,#e8e6f0 30%,#d4a853);'
    f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
    f'background-clip:text;line-height:1;margin-bottom:4px;">{titulo}</div>',
    unsafe_allow_html=True
)
st.markdown(
    f'<div style="font-size:0.75rem;color:#6b6b8a;letter-spacing:.1em;'
    f'text-transform:uppercase;margin-bottom:24px;">{subtitulo}</div>',
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN GENERAL
# ═══════════════════════════════════════════════════════
if seccion == "📊 Resumen General":

    total_v  = df_f['importe'].sum()
    total_a  = df_ant['importe'].sum()
    ticket   = df_f['importe'].mean() if len(df_f) else 0
    ticket_a = df_ant['importe'].mean() if len(df_ant) else 0
    n_cli    = df_f['customer_id'].nunique()
    n_cli_a  = df_ant['customer_id'].nunique()
    n_ped    = df_f['order_id'].nunique()
    n_ped_a  = df_ant['order_id'].nunique()
    frec     = n_ped/n_cli if n_cli else 0
    frec_a   = df_ant['order_id'].nunique()/df_ant['customer_id'].nunique() if df_ant['customer_id'].nunique() else 0

    canal_top     = df_f.groupby('canal')['importe'].sum().idxmax() if len(df_f) else '-'
    canal_top_pct = df_f.groupby('canal')['importe'].sum().max()/total_v*100 if total_v else 0
    cat_top       = df_f.groupby('categoria')['importe'].sum().idxmax() if len(df_f) else '-'
    vip_pct       = rfm_f[rfm_f['segmento_nombre']=='VIP 🏆'].shape[0]/rfm_f.shape[0]*100 if rfm_f.shape[0] else 0
    desc_medio    = df_f['descuento_pct'].mean()*100 if len(df_f) else 0
    ciudad_top    = df_f.groupby('ciudad')['importe'].sum().idxmax() if len(df_f) else '-'
    ticket_max    = df_f['importe'].max() if len(df_f) else 0

    spark_v  = df_f.groupby('año_mes')['importe'].sum().values if len(df_f) else [0]
    spark_c  = df_f.groupby('año_mes')['customer_id'].nunique().values if len(df_f) else [0]
    spark_p  = df_f.groupby('año_mes')['order_id'].nunique().values if len(df_f) else [0]

    st.markdown('<div class="sec-title">Métricas Clave</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Período seleccionado vs período anterior</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        (c1, "Facturación Total", f"{total_v/1e6:.2f}M€", delta_pct(total_v,total_a), GOLD,
         "Canal top", canal_top, "% canal", f"{canal_top_pct:.1f}%", spark_v),
        (c2, "Ticket Medio", f"{ticket:.0f}€", delta_pct(ticket,ticket_a), BLUE,
         "Ticket máx.", f"{ticket_max:.0f}€", "Dto. medio", f"{desc_medio:.1f}%",
         spark_v/np.maximum(spark_c,1)),
        (c3, "Clientes Activos", f"{n_cli:,}", delta_pct(n_cli,n_cli_a), GREEN,
         "% VIP", f"{vip_pct:.1f}%", "Ciudad top", ciudad_top, spark_c),
        (c4, "Total Pedidos", f"{n_ped:,}", delta_pct(n_ped,n_ped_a), '#a78bfa',
         "Cat. top", cat_top, "Con dto.", f"{df_f['descuento_pct'].gt(0).mean()*100:.0f}%", spark_p),
        (c5, "Frec. Media", f"{frec:.1f}x", delta_pct(frec,frec_a), RED,
         "VIP frec.", f"{rfm_f[rfm_f['segmento_nombre']=='VIP 🏆']['frecuencia'].mean():.1f}x",
         "Dormidos", f"{rfm_f[rfm_f['segmento_nombre']=='Dormidos 😴']['frecuencia'].mean():.1f}x",
         spark_p/np.maximum(spark_c,1)),
    ]
    for col_,lbl,val,dlt,clr,sl1,sv1,sl2,sv2,spk in kpis:
        with col_:
            st.markdown(
                f'<div class="kpi">'
                f'<div class="kpi-accent" style="background:linear-gradient(90deg,{clr},{clr}88);"></div>'
                f'<div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-main">{val}</div>'
                f'{fmt_delta(dlt," vs ant.")}'
                f'<div class="kpi-sub">'
                f'<div class="kpi-sub-item"><div class="kpi-sub-label">{sl1}</div>'
                f'<div class="kpi-sub-value">{sv1}</div></div>'
                f'<div class="kpi-sub-item"><div class="kpi-sub-label">{sl2}</div>'
                f'<div class="kpi-sub-value">{sv2}</div></div>'
                f'</div></div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(mini_sparkline(spk,clr), use_container_width=True,
                           config={'displayModeBar':False})

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([2,1])
    with col_l:
        st.markdown('<div class="sec-title">Evolución de Ventas</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Facturación mensual</div>', unsafe_allow_html=True)
        vm = df_f.groupby('año_mes')['importe'].sum().reset_index()
        fig = go.Figure(go.Scatter(
            x=vm['año_mes'], y=vm['importe'], fill='tozeroy',
            fillcolor=hex_to_rgba(GOLD,0.12), line=dict(color=GOLD,width=2), mode='lines',
            hovertemplate='<b>%{x}</b><br>%{y:,.0f}EUR<extra></extra>'
        ))
        fig.update_layout(**BASE_LAYOUT, height=300, xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec-title">Mix de Canales</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Por facturación</div>', unsafe_allow_html=True)
        cv = df_f.groupby('canal')['importe'].sum().reset_index()
        fig = go.Figure(go.Pie(
            labels=cv['canal'], values=cv['importe'], hole=0.55,
            marker_colors=PALETTE, textinfo='label+percent', textfont_size=11,
            hovertemplate='<b>%{label}</b><br>%{value:,.0f}EUR<extra></extra>'
        ))
        fig.update_layout(**BASE_LAYOUT, height=300, showlegend=False,
                          margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="sec-title">Ventas por Categoría & Año</div>', unsafe_allow_html=True)
        cat_año = df_f.groupby(['año','categoria'])['importe'].sum().reset_index()
        fig = px.bar(cat_año, x='categoria', y='importe', color='año',
                     barmode='group', color_discrete_sequence=PALETTE,
                     labels={'importe':'EUR','categoria':''})
        fig.update_layout(**BASE_LAYOUT, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="sec-title">Segmentos de Clientes</div>', unsafe_allow_html=True)
        sd = rfm['segmento_nombre'].value_counts().reset_index()
        sd.columns = ['Segmento','Clientes']
        fig = go.Figure(go.Bar(
            x=sd['Clientes'], y=sd['Segmento'], orientation='h',
            marker_color=[SEG_COL.get(s,MUTED) for s in sd['Segmento']],
            hovertemplate='<b>%{y}</b><br>%{x:,} clientes<extra></extra>'
        ))
        fig.update_layout(**BASE_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PÁGINA 2 — VENTAS & PRODUCTOS
# ═══════════════════════════════════════════════════════
elif seccion == "🛍️ Ventas & Productos":

    st.markdown('<div class="sec-title">Top 10 Productos</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Por facturación total en el período</div>', unsafe_allow_html=True)
    tp = (df_f.groupby('producto')
          .agg(facturacion=('importe','sum'), pedidos=('order_id','nunique'))
          .sort_values('facturacion',ascending=False).head(10).reset_index())
    fig = go.Figure(go.Bar(
        y=tp['producto'], x=tp['facturacion'], orientation='h',
        marker=dict(color=tp['facturacion'],
                    colorscale=[[0,'#2a1f0d'],[1,GOLD]], showscale=False),
        customdata=tp['pedidos'],
        hovertemplate='<b>%{y}</b><br>%{x:,.0f}EUR<br>Pedidos: %{customdata}<extra></extra>'
    ))
    fig.update_layout(**BASE_LAYOUT, height=380, showlegend=False)
    fig.update_yaxes(gridcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    tab_mes, tab_trim, tab_año = st.tabs(["Mensual", "Trimestral", "Anual"])
    with tab_mes:
        vm2 = df_f.groupby(['año_mes','canal'])['importe'].sum().reset_index()
        vmt = df_f.groupby('año_mes')['importe'].sum().reset_index()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        for i,canal in enumerate(df_f['canal'].unique()):
            sub = vm2[vm2['canal']==canal]
            fig.add_trace(go.Bar(name=canal, x=sub['año_mes'], y=sub['importe'],
                                 marker_color=PALETTE[i],
                                 hovertemplate='<b>%{x}</b><br>%{y:,.0f}EUR<extra>'+canal+'</extra>'))
        fig.add_trace(go.Scatter(name='Total', x=vmt['año_mes'], y=vmt['importe'],
                                 mode='lines+markers', line=dict(color='white',width=2,dash='dot'),
                                 marker=dict(size=4), yaxis='y2',
                                 hovertemplate='Total: %{y:,.0f}EUR<extra></extra>'), secondary_y=True)
        fig.update_layout(**BASE_LAYOUT, height=360, barmode='stack', xaxis_tickangle=-45)
        fig.update_yaxes(showgrid=False, secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab_trim:
        vt2 = df_f.groupby(['trimestre','categoria'])['importe'].sum().reset_index()
        fig = px.bar(vt2, x='trimestre', y='importe', color='categoria',
                     barmode='group', color_discrete_sequence=PALETTE,
                     labels={'importe':'EUR','trimestre':'','categoria':''})
        fig.update_layout(**BASE_LAYOUT, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with tab_año:
        va2 = df_f.groupby(['año','categoria'])['importe'].sum().reset_index()
        fig = px.bar(va2, x='categoria', y='importe', color='año',
                     barmode='group', color_discrete_sequence=PALETTE,
                     labels={'importe':'EUR','categoria':'','año':'Año'})
        fig.update_layout(**BASE_LAYOUT, height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="sec-title">Heatmap Categoría x Canal</div>', unsafe_allow_html=True)
        piv = df_f.pivot_table(values='importe',index='categoria',columns='canal',
                               aggfunc='sum').fillna(0)
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
            colorscale=[[0,'#0a0a0f'],[0.4,'#2a1f0d'],[1,GOLD]],
            text=[[f'{v/1000:.0f}k' for v in row] for row in piv.values],
            texttemplate='%{text}EUR',
            hovertemplate='<b>%{y} x %{x}</b><br>%{z:,.0f}EUR<extra></extra>',
            showscale=True
        ))
        fig.update_layout(**BASE_LAYOUT, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="sec-title">Distribución de Descuentos</div>', unsafe_allow_html=True)
        dd = df_f['descuento_pct'].apply(lambda x: f"{int(x*100)}%").value_counts().reset_index()
        dd.columns = ['Descuento','Pedidos']
        fig = go.Figure(go.Pie(
            labels=dd['Descuento'], values=dd['Pedidos'], hole=0.5,
            marker_colors=PALETTE, textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>%{value:,} pedidos<extra></extra>'
        ))
        fig.update_layout(**BASE_LAYOUT, height=320, showlegend=False,
                          margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PÁGINA 3 — SEGMENTACIÓN RFM
# ═══════════════════════════════════════════════════════
elif seccion == "👥 Segmentación RFM":

    seg_stats = rfm_f.groupby('segmento_nombre').agg(
        clientes  =('customer_id','count'),
        rec_media =('recencia',  'mean'),
        frec_media=('frecuencia','mean'),
        mon_media =('monetario', 'mean'),
    ).round(1).reset_index()

    acciones = {
        'VIP 🏆':       'Programa exclusivo · Acceso anticipado',
        'Fiel 💛':      'Descuentos volumen · Puntos fidelidad',
        'En Riesgo ⚠️': 'Email reactivación · Oferta personalizada',
        'Dormidos 😴':  'Win-back agresivo · Descuento >20%',
    }

    cols_seg = st.columns(4)
    for i, (_, row) in enumerate(seg_stats.iterrows()):
        clr    = SEG_COL.get(row['segmento_nombre'], MUTED)
        accion = acciones.get(row['segmento_nombre'], '')
        pct    = row['clientes']/rfm_f.shape[0]*100 if rfm_f.shape[0] else 0
        with cols_seg[i]:
            st.markdown(
                f'<div class="kpi" style="border-color:{clr}33;">'
                f'<div class="kpi-accent" style="background:{clr};"></div>'
                f'<div class="kpi-label">{row["segmento_nombre"]}</div>'
                f'<div class="kpi-main" style="font-size:1.8rem;color:{clr};">{int(row["clientes"]):,}</div>'
                f'<div style="font-size:0.7rem;color:#6b6b8a;">{pct:.1f}% del total</div>'
                f'<div class="kpi-sub">'
                f'<div class="kpi-sub-item"><div class="kpi-sub-label">Recencia</div>'
                f'<div class="kpi-sub-value">{row["rec_media"]:.0f}d</div></div>'
                f'<div class="kpi-sub-item"><div class="kpi-sub-label">Frec.</div>'
                f'<div class="kpi-sub-value">{row["frec_media"]:.1f}x</div></div>'
                f'<div class="kpi-sub-item"><div class="kpi-sub-label">Gasto</div>'
                f'<div class="kpi-sub-value">{row["mon_media"]:.0f}EUR</div></div>'
                f'</div>'
                f'<div style="font-size:0.68rem;color:#444;margin-top:8px;font-style:italic;">{accion}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_sc, col_rd = st.columns([3,2])

    with col_sc:
        seg_elegido = st.selectbox("Resaltar segmento", ["Todos"]+segs_all)
        fig = go.Figure()
        for seg, clr in SEG_COL.items():
            sub   = rfm_f[rfm_f['segmento_nombre']==seg]
            alpha = 0.85 if seg_elegido in ("Todos", seg) else 0.05
            fig.add_trace(go.Scatter(
                x=sub['frecuencia'], y=sub['monetario'], mode='markers', name=seg,
                marker=dict(color=clr,size=5,opacity=alpha),
                hovertemplate='<b>'+seg+'</b><br>Frec: %{x}<br>Gasto: %{y:,.0f}EUR<extra></extra>'
            ))
        fig.update_layout(**BASE_LAYOUT, height=380,
                          xaxis_title='Frecuencia', yaxis_title='Gasto total (EUR)')
        st.plotly_chart(fig, use_container_width=True)

    with col_rd:
        rd = rfm_f.groupby('segmento_nombre')[['recencia','frecuencia','monetario']].mean()
        if len(rd) > 0:
            rn = (rd-rd.min())/(rd.max()-rd.min()).replace(0,1)
            rn['recencia'] = 1-rn['recencia']
            rn.columns = ['Actualidad','Frecuencia','Valor']
            cats_r = list(rn.columns)+[rn.columns[0]]
            fig = go.Figure()
            for seg, clr in SEG_COL.items():
                if seg in rn.index:
                    vals = list(rn.loc[seg])+[rn.loc[seg].iloc[0]]
                    fig.add_trace(go.Scatterpolar(
                        r=vals, theta=cats_r, name=seg, fill='toself',
                        fillcolor=hex_to_rgba(clr,0.1),
                        line=dict(color=clr,width=2),
                    ))
            fig.update_layout(
                paper_bgcolor=BG_P, plot_bgcolor=BG_P,
                font=dict(color=FONT_C,family='Mulish'),
                polar=dict(bgcolor=BG_P,
                           radialaxis=dict(visible=True,range=[0,1],gridcolor=GRID,color=MUTED),
                           angularaxis=dict(gridcolor=GRID,color=FONT_C)),
                height=380, showlegend=True,
                legend=dict(bgcolor='rgba(0,0,0,0)',font_color=FONT_C,font_size=10,x=1),
                margin=dict(l=30,r=80,t=40,b=30)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Validación ML vs Perfil Real</div>', unsafe_allow_html=True)
    val = df.groupby('customer_id')['perfil_real'].first().reset_index()
    rfm_v = rfm_f.merge(val, on='customer_id')
    if len(rfm_v):
        tabla_v = pd.crosstab(rfm_v['segmento_nombre'],rfm_v['perfil_real'],normalize='index').round(3)*100
        fig = go.Figure(go.Heatmap(
            z=tabla_v.values, x=tabla_v.columns.tolist(), y=tabla_v.index.tolist(),
            colorscale=[[0,'#0a0a0f'],[0.5,'#1f2a10'],[1,GREEN]],
            text=[[f'{v:.1f}%' for v in row] for row in tabla_v.values],
            texttemplate='%{text}',
            hovertemplate='ML: %{y}<br>Real: %{x}<br>%{z:.1f}%<extra></extra>',
            showscale=True
        ))
        fig.update_layout(**BASE_LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PÁGINA 4 — RECOMENDADOR
# ═══════════════════════════════════════════════════════
elif seccion == "🤖 Recomendador":

    with st.spinner("Construyendo motor de recomendación..."):
        df_cat, sim_prod_df, sim_cli_df, muestra, top_seg = construir_motor(df, rfm)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Capas del sistema
    c1, c2, c3 = st.columns(3)
    capas = [
        (c1, "Capa 1 · 40%", "Por Segmento RFM",
         "Productos más comprados por clientes del mismo segmento. Mayor peso en Dormidos y En Riesgo.", GOLD),
        (c2, "Capa 2 · 40%", "Filtrado Colaborativo",
         "Clientes con perfil de compra similar compraron estos productos. Similitud coseno.", BLUE),
        (c3, "Capa 3 · 20%", "Basado en Contenido",
         "Productos similares a los ya comprados. TF-IDF sobre ingredientes y beneficios.", GREEN),
    ]
    for col_, lbl, tit, desc, clr in capas:
        with col_:
            st.markdown(
                f'<div class="kpi" style="border-color:{clr}33;">'
                f'<div class="kpi-accent" style="background:{clr};"></div>'
                f'<div class="kpi-label">{lbl}</div>'
                f'<div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;'
                f'color:{clr};margin:8px 0;">{tit}</div>'
                f'<div style="font-size:0.8rem;color:#b0aec8;line-height:1.5;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Consulta por Cliente</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Introduce un ID para ver recomendaciones personalizadas</div>', unsafe_allow_html=True)

    col_i1, col_i2 = st.columns([3,1])
    with col_i1:
        customer_input = st.text_input("cliente_id", placeholder="ej: CL00001", label_visibility="collapsed")
    with col_i2:
        top_n_rec = st.slider("Nº recs", 3, 10, 5)

    btn = st.button("Recomendar")

    # Botones prueba rápida
    st.markdown("<div style='font-size:0.72rem;color:#6b6b8a;margin:8px 0 6px;'>Prueba rápida:</div>", unsafe_allow_html=True)
    cols_ej = st.columns(4)
    ej_map = {
        "VIP":      rfm[rfm['segmento_nombre']=='VIP 🏆']['customer_id'].iloc[0],
        "Fiel":     rfm[rfm['segmento_nombre']=='Fiel 💛']['customer_id'].iloc[0],
        "En Riesgo":rfm[rfm['segmento_nombre']=='En Riesgo ⚠️']['customer_id'].iloc[0],
        "Dormido":  rfm[rfm['segmento_nombre']=='Dormidos 😴']['customer_id'].iloc[0],
    }
    for col_, (lbl_ej, cid_ej) in zip(cols_ej, ej_map.items()):
        with col_:
            if st.button(f"{lbl_ej}: {cid_ej}", use_container_width=True, key=f"ej_{cid_ej}"):
                customer_input = cid_ej
                btn = True

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    if btn and customer_input:
        cid = customer_input.strip().upper()
        if cid not in df['customer_id'].values:
            st.error(f"Cliente '{cid}' no encontrado. Formato: CL00001")
        else:
            recs, seg = recomendar_hibrido(
                cid, rfm, df, df_cat, sim_prod_df, sim_cli_df, muestra, top_seg, top_n=top_n_rec
            )
            info_cli     = rfm[rfm['customer_id']==cid].iloc[0]
            ya_comprados = df[df['customer_id']==cid]['producto'].unique().tolist()
            clr_seg      = SEG_COL.get(seg, MUTED)

            st.markdown(f'<div class="sec-title">Perfil: {cid}</div>', unsafe_allow_html=True)
            pk1,pk2,pk3,pk4,pk5 = st.columns(5)
            mets = [
                ("Segmento",    seg,                               clr_seg),
                ("Recencia",    f"{info_cli['recencia']} dias",    BLUE),
                ("Frecuencia",  f"{info_cli['frecuencia']} compras", GREEN),
                ("Gasto total", f"{info_cli['monetario']:,.0f}EUR", GOLD),
                ("Productos",   f"{len(ya_comprados)} distintos",  '#a78bfa'),
            ]
            for col_, (l2,v2,c2) in zip([pk1,pk2,pk3,pk4,pk5], mets):
                with col_:
                    st.markdown(
                        f'<div class="kpi" style="border-color:{c2}33;">'
                        f'<div class="kpi-accent" style="background:{c2};"></div>'
                        f'<div class="kpi-label">{l2}</div>'
                        f'<div class="kpi-main" style="font-size:1.3rem;color:{c2};">{v2}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            col_h, col_r2 = st.columns([1,1.6])

            with col_h:
                st.markdown('<div class="sec-title" style="font-size:1rem">Historial</div>', unsafe_allow_html=True)
                hist = (df[df['customer_id']==cid]
                        .groupby('producto')
                        .agg(gasto=('importe','sum'))
                        .sort_values('gasto',ascending=False).reset_index())
                fig_h = go.Figure(go.Bar(
                    x=hist['gasto'], y=hist['producto'], orientation='h',
                    marker=dict(color=hist['gasto'],
                                colorscale=[[0,'#1a1a26'],[1,BLUE]], showscale=False),
                    hovertemplate='<b>%{y}</b><br>%{x:,.0f}EUR<extra></extra>'
                ))
                fig_h.update_layout(**BASE_LAYOUT, height=max(200,len(hist)*45), showlegend=False)
                fig_h.update_yaxes(gridcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_h, use_container_width=True)

            with col_r2:
                st.markdown(f'<div class="sec-title" style="font-size:1rem">Top {top_n_rec} Recomendaciones</div>', unsafe_allow_html=True)
                max_sc = recs['score'].max() if recs['score'].max()>0 else 1
                for i, (_,row) in enumerate(recs.iterrows()):
                    bw  = int((row['score']/max_sc)*100)
                    cc  = CAT_COL.get(row['categoria'], MUTED)
                    st.markdown(
                        f'<div style="background:#12121a;border:1px solid #22223a;'
                        f'border-radius:12px;padding:14px 18px;margin-bottom:10px;'
                        f'position:relative;overflow:hidden;">'
                        f'<div style="position:absolute;bottom:0;left:0;height:3px;'
                        f'width:{bw}%;background:{cc};opacity:0.6;"></div>'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<div><span style="font-family:Syne Mono,monospace;font-size:0.65rem;'
                        f'color:#6b6b8a;margin-right:8px;">#{i+1}</span>'
                        f'<span style="font-weight:600;color:#e8e6f0;">{row["producto"]}</span></div>'
                        f'<span style="font-size:1rem;font-weight:700;color:{cc};">{row["precio"]:.2f}EUR</span>'
                        f'</div>'
                        f'<div style="margin-top:6px;display:flex;gap:12px;">'
                        f'<span style="font-size:0.7rem;color:#6b6b8a;">&#9679; {row["categoria"]}</span>'
                        f'<span style="font-size:0.7rem;color:#6b6b8a;">Score: '
                        f'<b style="color:#b0aec8;">{row["score"]:.3f}</b></span>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-title" style="font-size:1rem">Mapa de similitud entre productos</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-sub">Historial del cliente + productos recomendados</div>', unsafe_allow_html=True)

            prods_m = list(ya_comprados[:6]) + recs['producto'].tolist()[:4]
            prods_m = list(dict.fromkeys(prods_m))
            prods_m = [p for p in prods_m if p in sim_prod_df.index][:10]
            if len(prods_m) > 1:
                sub_s = sim_prod_df.loc[prods_m, prods_m]
                etiq  = [p[:22]+'...' if len(p)>22 else p for p in sub_s.index]
                fig_s = go.Figure(go.Heatmap(
                    z=sub_s.values, x=etiq, y=etiq,
                    colorscale=[[0,'#0a0a0f'],[0.5,'#1f2a10'],[1,GREEN]],
                    text=[[f'{v:.2f}' for v in fila] for fila in sub_s.values],
                    texttemplate='%{text}',
                    hovertemplate='%{y} vs %{x}<br>Similitud: %{z:.3f}<extra></extra>',
                    showscale=True
                ))
                fig_s.update_layout(**BASE_LAYOUT, height=400)
                fig_s.update_layout(xaxis=dict(tickangle=-35,tickfont_size=9,gridcolor=GRID))
                fig_s.update_layout(yaxis=dict(tickfont_size=9,gridcolor=GRID))
                st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.markdown(
            '<div style="background:#12121a;border:1px dashed #22223a;border-radius:16px;'
            'padding:60px;text-align:center;margin-top:16px;">'
            '<div style="font-size:2.5rem;margin-bottom:12px;">🤖</div>'
            '<div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:#e8e6f0;">'
            'Introduce un ID de cliente o usa los botones de prueba</div>'
            '<div style="font-size:0.85rem;color:#6b6b8a;margin-top:8px;">'
            'Formato: CL00001 hasta CL05000</div>'
            '</div>',
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════════════
# PÁGINA 5 — TABLA DE DETALLE
# ═══════════════════════════════════════════════════════
elif seccion == "📋 Tabla de Detalle":

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        search_prod = st.text_input("Buscar producto", placeholder="ej: Sérum...")
    with col_t2:
        order_by = st.selectbox("Ordenar por", ['importe','cantidad','descuento_pct','fecha'])
    with col_t3:
        top_n = st.slider("Nº de filas", 10, 200, 50, step=10)

    df_tabla = df_f.copy()
    if search_prod:
        df_tabla = df_tabla[df_tabla['producto'].str.contains(search_prod, case=False, na=False)]
    df_tabla = df_tabla.sort_values(order_by, ascending=False).head(top_n)

    rfm_seg  = rfm[['customer_id','segmento_nombre']]
    df_tabla = df_tabla.merge(rfm_seg, on='customer_id', how='left')

    df_show  = df_tabla[['fecha','order_id','customer_id','segmento_nombre','ciudad','canal',
                          'categoria','producto','cantidad','precio_unitario','descuento_pct','importe']].copy()
    df_show['fecha']           = df_show['fecha'].dt.strftime('%Y-%m-%d')
    df_show['descuento_pct']   = (df_show['descuento_pct']*100).round(0).astype(int).astype(str)+'%'
    df_show['importe']         = df_show['importe'].apply(lambda x: f'{x:,.2f}EUR')
    df_show['precio_unitario'] = df_show['precio_unitario'].apply(lambda x: f'{x:.2f}EUR')
    df_show.columns = ['Fecha','Pedido','Cliente','Segmento','Ciudad','Canal',
                       'Categoría','Producto','Uds.','Precio unit.','Descuento','Importe']

    st.dataframe(df_show, use_container_width=True, height=500, hide_index=True)

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown(
            f"<div style='font-size:0.75rem;color:{MUTED};'>"
            f"Mostrando <b style='color:{FONT_C}'>{len(df_tabla):,}</b> registros</div>",
            unsafe_allow_html=True
        )
    with col_r2:
        tt = df_f.groupby('producto')['importe'].sum()
        pt = tt.idxmax() if len(tt) else '-'
        st.markdown(
            f"<div style='font-size:0.75rem;color:{MUTED};'>"
            f"Producto top: <b style='color:{GOLD}'>{pt}</b></div>",
            unsafe_allow_html=True
        )
    with col_r3:
        ft = df_f['importe'].sum()
        st.markdown(
            f"<div style='font-size:0.75rem;color:{MUTED};'>"
            f"Facturación total filtrada: <b style='color:{GREEN}'>{ft:,.0f}EUR</b></div>",
            unsafe_allow_html=True
        )

st.markdown(
    '<div style="text-align:center;margin-top:40px;font-family:Syne Mono,monospace;'
    'font-size:0.6rem;color:#2a2a3a;letter-spacing:.1em;">'
    'FACEGLOSS EXECUTIVE DASHBOARD v3.0 · PROYECTO IA &amp; BIG DATA · 2024'
    '</div>',
    unsafe_allow_html=True
)
