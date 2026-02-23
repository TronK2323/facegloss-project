"""
FACEGLOSS — Dashboard Ejecutivo v4.0
Paleta: Lila suave + Crema blanco — Estilo luxury beauty
pip install streamlit plotly pandas numpy scikit-learn
streamlit run dashboard_facegloss.py
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

st.set_page_config(
    page_title="Facegloss · Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PALETA DE COLORES — Facegloss Brand
# ─────────────────────────────────────────────
LILAC      = '#9B8EC4'      # lila principal
LILAC_L    = '#C8BDE8'      # lila claro
LILAC_XL   = '#EDE8F7'      # lila muy claro (fondos)
LILAC_D    = '#6B5FA0'      # lila oscuro (texto activo)
CREAM      = '#FAF7F2'      # crema fondo principal
CREAM_D    = '#F0EBE3'      # crema oscuro (cards)
CREAM_DD   = '#E4DDD4'      # crema borde
ROSE       = '#E8A0B0'      # acento rosa
SAGE       = '#8BB8A0'      # acento verde salvia
MAUVE      = '#C49BBB'      # acento malva
TEXT_D     = '#2D2640'      # texto oscuro principal
TEXT_M     = '#6B6480'      # texto medio
TEXT_L     = '#9B96A8'      # texto claro

PALETTE    = [LILAC, ROSE, SAGE, MAUVE, '#C4A882', '#7BB8C4']
SEG_COL    = {
    'VIP 🏆':       LILAC_D,
    'Fiel 💛':      SAGE,
    'En Riesgo ⚠️': ROSE,
    'Dormidos 😴':  TEXT_L,
}
CAT_COL    = {
    'Hidratación': '#7BB8C4',
    'Antiedad':    LILAC,
    'Limpieza':    SAGE,
    'Protección':  ROSE,
}

# ─────────────────────────────────────────────
# ESTILOS GLOBALES
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@300;400&display=swap');

:root {{
    --cream:    {CREAM};
    --cream-d:  {CREAM_D};
    --cream-dd: {CREAM_DD};
    --lilac:    {LILAC};
    --lilac-l:  {LILAC_L};
    --lilac-xl: {LILAC_XL};
    --lilac-d:  {LILAC_D};
    --rose:     {ROSE};
    --sage:     {SAGE};
    --text-d:   {TEXT_D};
    --text-m:   {TEXT_M};
    --text-l:   {TEXT_L};
}}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--text-d);
}}

.stApp {{ background-color: var(--cream); }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: white;
    border-right: 1px solid var(--cream-dd);
    box-shadow: 2px 0 20px rgba(155,142,196,0.08);
}}
[data-testid="stSidebar"] * {{
    color: var(--text-m) !important;
}}

/* Botones radio sidebar */
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    border-radius: 12px !important;
    padding: 10px 16px !important;
    margin: 2px 0 !important;
    transition: all 0.2s ease;
    font-size: 0.9rem !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background: var(--lilac-xl) !important;
    color: var(--lilac-d) !important;
}}

/* ── KPI Card ── */
.kpi {{
    background: white;
    border: 1px solid var(--cream-dd);
    border-radius: 20px;
    padding: 22px 24px 18px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(155,142,196,0.08);
    height: 100%;
    transition: box-shadow 0.2s ease;
}}
.kpi:hover {{
    box-shadow: 0 4px 24px rgba(155,142,196,0.16);
}}
.kpi-accent {{
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 20px 20px 0 0;
}}
.kpi-dot {{
    position: absolute;
    top: 18px; right: 20px;
    width: 8px; height: 8px;
    border-radius: 50%;
    opacity: 0.5;
}}
.kpi-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-l);
    margin-bottom: 12px;
}}
.kpi-main {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 500;
    color: var(--text-d);
    line-height: 1;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}}
.kpi-delta-pos {{ font-size: 0.75rem; color: {SAGE}; font-weight: 600; }}
.kpi-delta-neg {{ font-size: 0.75rem; color: {ROSE}; font-weight: 600; }}
.kpi-sub {{
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid var(--cream-dd);
    display: flex;
    gap: 16px;
}}
.kpi-sub-label {{
    font-size: 0.58rem;
    color: var(--text-l);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 3px;
}}
.kpi-sub-value {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-m);
}}

/* ── Section headers ── */
.sec-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.7rem;
    font-weight: 400;
    font-style: italic;
    color: var(--text-d);
    margin-bottom: 2px;
    letter-spacing: -0.3px;
}}
.sec-sub {{
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-l);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 20px;
}}

/* ── Divider ── */
.hr {{
    border: none;
    border-top: 1px solid var(--cream-dd);
    margin: 28px 0;
}}

/* ── Rec card ── */
.rec-card {{
    background: white;
    border: 1px solid var(--cream-dd);
    border-radius: 16px;
    padding: 14px 18px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 8px rgba(155,142,196,0.06);
    transition: box-shadow 0.2s;
}}
.rec-card:hover {{ box-shadow: 0 4px 20px rgba(155,142,196,0.15); }}

/* ── Tags ── */
.tag {{
    display: inline-block;
    background: var(--lilac-xl);
    color: var(--lilac-d);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}}

/* Streamlit overrides */
.stButton > button {{
    background: white;
    border: 1.5px solid var(--cream-dd);
    border-radius: 12px;
    color: var(--text-m);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    padding: 8px 16px;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    border-color: var(--lilac);
    color: var(--lilac-d);
    background: var(--lilac-xl);
}}
.stTextInput > div > input {{
    border-radius: 12px !important;
    border-color: var(--cream-dd) !important;
    background: white !important;
    color: var(--text-d) !important;
    font-family: 'DM Sans', sans-serif !important;
}}
.stSelectbox > div, .stMultiSelect > div {{
    border-radius: 12px !important;
}}
.stDataFrame {{ border-radius: 16px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER: layout sin conflictos
# ─────────────────────────────────────────────
BG_PLOT = 'rgba(0,0,0,0)'  # transparente para heredar fondo cream

def make_layout(**overrides):
    """Devuelve un dict de layout Plotly sin conflictos de claves."""
    base = dict(
        paper_bgcolor=BG_PLOT,
        plot_bgcolor=BG_PLOT,
        font=dict(color=TEXT_M, family='DM Sans', size=12),
        margin=dict(l=16, r=16, t=36, b=16),
        xaxis=dict(gridcolor=CREAM_DD, zerolinecolor=CREAM_DD,
                   tickfont=dict(size=11, color=TEXT_L)),
        yaxis=dict(gridcolor=CREAM_DD, zerolinecolor=CREAM_DD,
                   tickfont=dict(size=11, color=TEXT_L)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font_color=TEXT_M, font_size=11),
        hoverlabel=dict(bgcolor='white', font_color=TEXT_D, font_size=12,
                        bordercolor=CREAM_DD),
    )
    base.update(overrides)  # override sin conflicto
    return base

def hex_rgba(hex_c, a=0.12):
    h = hex_c.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{a})'

def mini_spark(series, color=LILAC):
    fig = go.Figure(go.Scatter(
        y=series, mode='lines',
        line=dict(color=color, width=1.5),
        fill='tozeroy', fillcolor=hex_rgba(color, 0.1),
        hoverinfo='skip'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=4,b=0), height=44,
        xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False
    )
    return fig

def delta_pct(a, b):
    return (a-b)/b*100 if b else 0

def fmt_delta(pct, suffix=''):
    arrow = '↑' if pct >= 0 else '↓'
    cls   = 'kpi-delta-pos' if pct >= 0 else 'kpi-delta-neg'
    return f'<span class="{cls}">{arrow} {abs(pct):.1f}%{suffix}</span>'

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
    fecha_ini = datetime(2022,1,1); fecha_fin = datetime(2024,12,31)
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
    df['año']       = df['fecha'].dt.year
    df['año_mes']   = df['fecha'].dt.to_period('M').astype(str)
    df['trimestre'] = df['fecha'].dt.to_period('Q').astype(str)
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
def calcular_churn(df, rfm):
    FECHA_REF    = datetime(2025, 1, 1)
    UMBRAL_CHURN = 180

    ch = df.groupby('customer_id').agg(
        ultima_compra   = ('fecha',         'max'),
        primera_compra  = ('fecha',         'min'),
        frecuencia      = ('order_id',      'nunique'),
        monetario       = ('importe',       'sum'),
        n_categorias    = ('categoria',     'nunique'),
        n_productos     = ('producto',      'nunique'),
        descuento_medio = ('descuento_pct', 'mean'),
        ticket_medio    = ('importe',       'mean'),
        ticket_max      = ('importe',       'max'),
        perfil_real     = ('perfil_real',   'first'),
        edad            = ('edad',          'first'),
        canal           = ('canal',         lambda x: x.mode()[0]),
        cat_favorita    = ('categoria',     lambda x: x.mode()[0]),
    ).reset_index()

    ch['recencia']    = (FECHA_REF - ch['ultima_compra']).dt.days
    ch['antiguedad']  = (ch['ultima_compra'] - ch['primera_compra']).dt.days
    ch['monetario']   = ch['monetario'].round(2)
    ch['ticket_medio']= ch['ticket_medio'].round(2)
    ch['churn']       = (ch['recencia'] > UMBRAL_CHURN).astype(int)
    ch['gasto_por_compra'] = ch['monetario'] / ch['frecuencia']
    ch['densidad_compra']  = ch['frecuencia'] / (ch['antiguedad'] + 1)
    ch['diversidad_cat']   = ch['n_categorias'] / 4

    le1, le2 = LabelEncoder(), LabelEncoder()
    ch['canal_enc'] = le1.fit_transform(ch['canal'])
    ch['cat_enc']   = le2.fit_transform(ch['cat_favorita'])

    FEATURES = ['recencia','frecuencia','monetario','ticket_medio','ticket_max',
                'n_categorias','n_productos','descuento_medio','antiguedad','edad',
                'canal_enc','cat_enc','gasto_por_compra','densidad_compra','diversidad_cat']

    X = ch[FEATURES].copy()
    y = ch['churn'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train)
    X_te_sc  = sc.transform(X_test)
    X_all_sc = sc.transform(X)

    modelo = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    modelo.fit(X_tr_sc, y_train)

    auc = roc_auc_score(y_test, modelo.predict_proba(X_te_sc)[:, 1])

    ch['prob_churn'] = modelo.predict_proba(X_all_sc)[:, 1].round(4)
    ch['riesgo'] = pd.cut(
        ch['prob_churn'],
        bins=[0, 0.30, 0.55, 0.75, 1.0],
        labels=['Bajo', 'Medio', 'Alto', 'Crítico'])

    # Merge con segmento RFM
    ch = ch.merge(rfm[['customer_id','segmento_nombre']], on='customer_id', how='left')

    fi = pd.DataFrame({'feature': FEATURES, 'importance': modelo.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)

    fpr, tpr, _ = roc_curve(y_test, modelo.predict_proba(X_te_sc)[:, 1])

    return ch, auc, fi, fpr, tpr

@st.cache_data
def construir_motor(df, rfm):
    tags_map = {
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
        rows_cat.append({'producto':prod,'categoria':cat,'precio':precio,
                         'tags':tags_map.get(prod, cat.lower())})
    df_cat = pd.DataFrame(rows_cat).drop_duplicates('producto').reset_index(drop=True)

    tfidf       = TfidfVectorizer()
    tfidf_mat   = tfidf.fit_transform(df_cat['tags'])
    sim_prod_df = pd.DataFrame(cosine_similarity(tfidf_mat),
                               index=df_cat['producto'], columns=df_cat['producto'])

    matriz_cp   = df.pivot_table(index='customer_id',columns='producto',
                                  values='importe',aggfunc='sum',fill_value=0)
    muestra     = matriz_cp.sample(n=min(800,len(matriz_cp)),random_state=42)
    sim_cli_df  = pd.DataFrame(cosine_similarity(muestra),
                               index=muestra.index, columns=muestra.index)

    df_seg = df.merge(rfm[['customer_id','segmento_nombre']],on='customer_id')
    rec_s  = (df_seg.groupby(['segmento_nombre','producto'])
              .agg(facturacion=('importe','sum'),cu=('customer_id','nunique'))
              .reset_index())
    rec_s['score'] = (
        rec_s['facturacion']/rec_s.groupby('segmento_nombre')['facturacion'].transform('max')*0.6 +
        rec_s['cu']/rec_s.groupby('segmento_nombre')['cu'].transform('max')*0.4
    )
    top_seg = (rec_s.sort_values(['segmento_nombre','score'],ascending=[True,False])
               .groupby('segmento_nombre').head(8).reset_index(drop=True))
    return df_cat, sim_prod_df, sim_cli_df, muestra, top_seg

def recomendar(cid, rfm, df, df_cat, sprod, scli, muestra, top_seg, n=5):
    todos  = df_cat['producto'].tolist()
    scores = {p:0.0 for p in todos}
    info   = rfm[rfm['customer_id']==cid]
    seg    = info.iloc[0]['segmento_nombre'] if not info.empty else 'Normal'
    comp   = set(df[df['customer_id']==cid]['producto'].unique())
    ws,wc,wt = (0.60,0.25,0.15) if seg in ['Dormidos 😴','En Riesgo ⚠️'] else (0.40,0.40,0.20)

    sr = top_seg[top_seg['segmento_nombre']==seg]
    ms = sr['score'].max() if len(sr) else 1
    for _,r in sr.iterrows():
        if r['producto'] not in comp and r['producto'] in scores:
            scores[r['producto']] += ws*(r['score']/ms)

    if cid in scli.index:
        sims = scli[cid].drop(cid,errors='ignore').nlargest(10)
        cs = {}
        for sid,sv in sims.items():
            if sid in muestra.index:
                for prod,g in muestra.loc[sid].items():
                    if g>0 and prod not in comp: cs[prod]=cs.get(prod,0)+g*sv
        if cs:
            mc = max(cs.values())
            for prod,sc in cs.items():
                if prod in scores: scores[prod]+=wc*(sc/mc)

    for pc in list(comp)[:3]:
        if pc in sprod.index:
            for i,(ps,_) in enumerate(sprod[pc].drop(pc,errors='ignore').nlargest(3).items()):
                if ps not in comp and ps in scores: scores[ps]+=wt*(1/(i+1))

    res = (pd.DataFrame.from_dict(scores,orient='index',columns=['score'])
           .reset_index().rename(columns={'index':'producto'})
           .merge(df_cat[['producto','categoria','precio']],on='producto')
           .sort_values('score',ascending=False).query('score>0')
           .head(n).reset_index(drop=True))
    res.index += 1
    return res, seg

# ─────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────
df  = generar_datos()
rfm = calcular_rfm(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:24px 0 20px;'>
        <div style='font-size:2rem;margin-bottom:6px;'>🌸</div>
        <div style='font-family:Cormorant Garamond,serif;font-size:1.5rem;
                    font-weight:400;color:{TEXT_D};letter-spacing:2px;'>FACEGLOSS</div>
        <div style='font-family:DM Mono,monospace;font-size:0.55rem;color:{TEXT_L};
                    letter-spacing:.2em;text-transform:uppercase;margin-top:4px;'>
            Executive Dashboard
        </div>
    </div>
    <hr style='border-color:{CREAM_DD};margin-bottom:16px;'>
    """, unsafe_allow_html=True)

    seccion = st.radio("Nav", [
        "📊  Resumen General",
        "🛍️  Ventas & Productos",
        "👥  Segmentación RFM",
        "🤖  Recomendador",
        "🚨  Alertas Churn",
        "📋  Tabla de Detalle",
    ], label_visibility="collapsed")

    st.markdown(f"<hr style='border-color:{CREAM_DD};margin:16px 0 12px;'><div style='font-family:DM Mono,monospace;font-size:0.55rem;color:{TEXT_L};letter-spacing:.15em;text-transform:uppercase;margin-bottom:10px;'>Filtros</div>", unsafe_allow_html=True)

    años_d  = sorted(df['año'].unique().tolist())
    años_s  = st.multiselect("Año", años_d, default=años_d)
    cats_d  = sorted(df['categoria'].unique().tolist())
    cats_s  = st.multiselect("Categoría", cats_d, default=cats_d)
    city_d  = sorted(df['ciudad'].unique().tolist())
    city_s  = st.multiselect("Ciudad", city_d, default=city_d)
    can_d   = sorted(df['canal'].unique().tolist())
    can_s   = st.multiselect("Canal", can_d, default=can_d)
    segs_a  = ['VIP 🏆','Fiel 💛','En Riesgo ⚠️','Dormidos 😴']
    seg_s   = st.multiselect("Segmento", segs_a, default=segs_a)

    st.markdown(f"<hr style='border-color:{CREAM_DD};margin:16px 0 8px;'><div style='font-family:DM Mono,monospace;font-size:0.5rem;color:{CREAM_DD};text-align:center;'>Proyecto IA & Big Data · 2024</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTRADO
# ─────────────────────────────────────────────
df_f   = df[df['año'].isin(años_s) & df['categoria'].isin(cats_s) &
            df['ciudad'].isin(city_s) & df['canal'].isin(can_s)]
rfm_f  = rfm[rfm['segmento_nombre'].isin(seg_s)]
df_ant = df[df['año'].isin([a-1 for a in años_s]) & df['categoria'].isin(cats_s) &
            df['ciudad'].isin(city_s) & df['canal'].isin(can_s)]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
TITULOS = {
    "📊  Resumen General":   ("Resumen General",        "KPIs · Evolución · Canales · Tendencias"),
    "🛍️  Ventas & Productos": ("Ventas & Productos",     "Rendimiento por producto, canal y categoría"),
    "👥  Segmentación RFM":  ("Segmentación RFM",        "K-Means · 4 clusters · Features enriquecidas"),
    "🤖  Recomendador":      ("Recomendador",             "Sistema híbrido · Segmento + Colaborativo + Contenido"),
    "🚨  Alertas Churn":     ("Alertas Churn",            "Gradient Boosting · Predicción de abandono · AUC-ROC"),
    "📋  Tabla de Detalle":  ("Tabla de Detalle",         "Vista granular con búsqueda y filtros"),
}
titulo, subtitulo = TITULOS.get(seccion, ("Dashboard", ""))
st.markdown(
    f'<div style="font-family:Cormorant Garamond,serif;font-size:3rem;font-weight:300;'
    f'font-style:italic;color:{TEXT_D};line-height:1;margin-bottom:4px;'
    f'letter-spacing:-1px;">{titulo}</div>',
    unsafe_allow_html=True
)
st.markdown(
    f'<div style="font-family:DM Mono,monospace;font-size:0.6rem;color:{TEXT_L};'
    f'letter-spacing:.15em;text-transform:uppercase;margin-bottom:28px;">{subtitulo}</div>',
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════
# P1 — RESUMEN GENERAL
# ═══════════════════════════════════════════════════════
if seccion == "📊  Resumen General":

    tv  = df_f['importe'].sum()
    ta  = df_ant['importe'].sum()
    tk  = df_f['importe'].mean() if len(df_f) else 0
    tka = df_ant['importe'].mean() if len(df_ant) else 0
    nc  = df_f['customer_id'].nunique()
    nca = df_ant['customer_id'].nunique()
    np_ = df_f['order_id'].nunique()
    npa = df_ant['order_id'].nunique()
    fr  = np_/nc if nc else 0
    fra = df_ant['order_id'].nunique()/df_ant['customer_id'].nunique() if df_ant['customer_id'].nunique() else 0

    ct  = df_f.groupby('canal')['importe'].sum().idxmax() if len(df_f) else '-'
    ctp = df_f.groupby('canal')['importe'].sum().max()/tv*100 if tv else 0
    cat = df_f.groupby('categoria')['importe'].sum().idxmax() if len(df_f) else '-'
    vip = rfm_f[rfm_f['segmento_nombre']=='VIP 🏆'].shape[0]/rfm_f.shape[0]*100 if rfm_f.shape[0] else 0
    ciu = df_f.groupby('ciudad')['importe'].sum().idxmax() if len(df_f) else '-'
    tkm = df_f['importe'].max() if len(df_f) else 0

    sp1 = df_f.groupby('año_mes')['importe'].sum().values if len(df_f) else [0]
    sp2 = df_f.groupby('año_mes')['customer_id'].nunique().values if len(df_f) else [0]
    sp3 = df_f.groupby('año_mes')['order_id'].nunique().values if len(df_f) else [0]

    st.markdown('<div class="sec-title">Métricas clave</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Período seleccionado vs período anterior</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        (c1, "Facturación", f"{tv/1e6:.2f}M€", delta_pct(tv,ta), LILAC,
         "Canal top", ct, "% canal", f"{ctp:.1f}%", sp1),
        (c2, "Ticket Medio", f"{tk:.0f}€", delta_pct(tk,tka), ROSE,
         "Ticket máx.", f"{tkm:.0f}€", "Dto. medio",
         f"{df_f['descuento_pct'].mean()*100:.1f}%", sp1/np.maximum(sp2,1)),
        (c3, "Clientes", f"{nc:,}", delta_pct(nc,nca), SAGE,
         "% VIP", f"{vip:.1f}%", "Ciudad top", ciu, sp2),
        (c4, "Pedidos", f"{np_:,}", delta_pct(np_,npa), MAUVE,
         "Cat. top", cat, "Con dto.",
         f"{df_f['descuento_pct'].gt(0).mean()*100:.0f}%", sp3),
        (c5, "Frec. Media", f"{fr:.1f}x", delta_pct(fr,fra), '#C4A882',
         "VIP frec.",
         f"{rfm_f[rfm_f['segmento_nombre']=='VIP 🏆']['frecuencia'].mean():.1f}x",
         "Dormidos",
         f"{rfm_f[rfm_f['segmento_nombre']=='Dormidos 😴']['frecuencia'].mean():.1f}x",
         sp3/np.maximum(sp2,1)),
    ]
    for col_,lbl,val,dlt,clr,sl1,sv1,sl2,sv2,spk in kpis:
        with col_:
            st.markdown(
                f'<div class="kpi">'
                f'<div class="kpi-accent" style="background:linear-gradient(90deg,{clr},{clr}66);"></div>'
                f'<div class="kpi-dot" style="background:{clr};"></div>'
                f'<div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-main">{val}</div>'
                f'{fmt_delta(dlt," vs ant.")}'
                f'<div class="kpi-sub">'
                f'<div><div class="kpi-sub-label">{sl1}</div><div class="kpi-sub-value">{sv1}</div></div>'
                f'<div><div class="kpi-sub-label">{sl2}</div><div class="kpi-sub-value">{sv2}</div></div>'
                f'</div></div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(mini_spark(spk,clr), width="stretch",
                            config={'displayModeBar':False})

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([2,1])
    with col_l:
        st.markdown('<div class="sec-title">Evolución de ventas</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Facturación mensual</div>', unsafe_allow_html=True)
        vm = df_f.groupby('año_mes')['importe'].sum().reset_index()
        fig = go.Figure(go.Scatter(
            x=vm['año_mes'], y=vm['importe'], fill='tozeroy',
            fillcolor=hex_rgba(LILAC_L, 0.2),
            line=dict(color=LILAC, width=2), mode='lines',
            hovertemplate='<b>%{x}</b><br>%{y:,.0f}€<extra></extra>'
        ))
        fig.update_layout(**make_layout(height=300, xaxis_tickangle=-45, showlegend=False))
        st.plotly_chart(fig, width="stretch")

    with col_r:
        st.markdown('<div class="sec-title">Mix canales</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Por facturación</div>', unsafe_allow_html=True)
        cv = df_f.groupby('canal')['importe'].sum().reset_index()
        fig = go.Figure(go.Pie(
            labels=cv['canal'], values=cv['importe'], hole=0.6,
            marker=dict(colors=PALETTE,
                        line=dict(color='white', width=3)),
            textinfo='label+percent', textfont_size=11,
            hovertemplate='<b>%{label}</b><br>%{value:,.0f}€<extra></extra>'
        ))
        fig.update_layout(**make_layout(height=300, showlegend=False,
                          margin=dict(l=10,r=10,t=30,b=10)))
        st.plotly_chart(fig, width="stretch")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="sec-title">Por categoría & año</div>', unsafe_allow_html=True)
        ca = df_f.groupby(['año','categoria'])['importe'].sum().reset_index()
        fig = px.bar(ca, x='categoria', y='importe', color='año',
                     barmode='group', color_discrete_sequence=PALETTE,
                     labels={'importe':'€','categoria':'','año':'Año'})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(**make_layout(height=320))
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.markdown('<div class="sec-title">Segmentos RFM</div>', unsafe_allow_html=True)
        sd = rfm['segmento_nombre'].value_counts().reset_index()
        sd.columns = ['Segmento','Clientes']
        fig = go.Figure(go.Bar(
            x=sd['Clientes'], y=sd['Segmento'], orientation='h',
            marker=dict(color=[SEG_COL.get(s,TEXT_L) for s in sd['Segmento']],
                        line_width=0),
            hovertemplate='<b>%{y}</b><br>%{x:,} clientes<extra></extra>'
        ))
        fig.update_layout(**make_layout(height=320, showlegend=False))
        st.plotly_chart(fig, width="stretch")

# ═══════════════════════════════════════════════════════
# P2 — VENTAS & PRODUCTOS
# ═══════════════════════════════════════════════════════
elif seccion == "🛍️  Ventas & Productos":

    st.markdown('<div class="sec-title">Top 10 productos</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Por facturación en el período seleccionado</div>', unsafe_allow_html=True)
    tp = (df_f.groupby('producto')
          .agg(facturacion=('importe','sum'),pedidos=('order_id','nunique'))
          .sort_values('facturacion',ascending=False).head(10).reset_index())
    fig = go.Figure(go.Bar(
        y=tp['producto'], x=tp['facturacion'], orientation='h',
        marker=dict(color=tp['facturacion'],
                    colorscale=[[0,LILAC_XL],[0.5,LILAC_L],[1,LILAC_D]],
                    showscale=False, line_width=0),
        customdata=tp['pedidos'],
        hovertemplate='<b>%{y}</b><br>%{x:,.0f}€<br>Pedidos: %{customdata}<extra></extra>'
    ))
    fig.update_layout(**make_layout(height=380, showlegend=False))
    fig.update_yaxes(gridcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, width="stretch")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    tab_m, tab_t, tab_a = st.tabs(["Mensual", "Trimestral", "Anual"])
    with tab_m:
        vm2 = df_f.groupby(['año_mes','canal'])['importe'].sum().reset_index()
        vmt = df_f.groupby('año_mes')['importe'].sum().reset_index()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        for i,canal in enumerate(df_f['canal'].unique()):
            sub = vm2[vm2['canal']==canal]
            fig.add_trace(go.Bar(name=canal, x=sub['año_mes'], y=sub['importe'],
                                 marker=dict(color=PALETTE[i],line_width=0),
                                 hovertemplate='<b>%{x}</b><br>%{y:,.0f}€<extra>'+canal+'</extra>'))
        fig.add_trace(go.Scatter(name='Total', x=vmt['año_mes'], y=vmt['importe'],
                                 mode='lines', line=dict(color=TEXT_M,width=1.5,dash='dot'),
                                 yaxis='y2',
                                 hovertemplate='Total: %{y:,.0f}€<extra></extra>'), secondary_y=True)
        fig.update_layout(**make_layout(height=360, barmode='stack', xaxis_tickangle=-45))
        fig.update_yaxes(showgrid=False, secondary_y=True)
        st.plotly_chart(fig, width="stretch")

    with tab_t:
        vt2 = df_f.groupby(['trimestre','categoria'])['importe'].sum().reset_index()
        fig = px.bar(vt2, x='trimestre', y='importe', color='categoria',
                     barmode='group', color_discrete_sequence=PALETTE,
                     labels={'importe':'€','trimestre':'','categoria':''})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(**make_layout(height=360))
        st.plotly_chart(fig, width="stretch")

    with tab_a:
        va2 = df_f.groupby(['año','categoria'])['importe'].sum().reset_index()
        fig = px.bar(va2, x='categoria', y='importe', color='año',
                     barmode='group', color_discrete_sequence=PALETTE,
                     labels={'importe':'€','categoria':'','año':'Año'})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(**make_layout(height=360))
        st.plotly_chart(fig, width="stretch")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="sec-title">Heatmap categoría × canal</div>', unsafe_allow_html=True)
        piv = df_f.pivot_table(values='importe',index='categoria',
                               columns='canal',aggfunc='sum').fillna(0)
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
            colorscale=[[0,CREAM],[0.5,LILAC_L],[1,LILAC_D]],
            text=[[f'{v/1000:.0f}k€' for v in row] for row in piv.values],
            texttemplate='%{text}',
            hovertemplate='<b>%{y} × %{x}</b><br>%{z:,.0f}€<extra></extra>',
            showscale=True
        ))
        fig.update_layout(**make_layout(height=300))
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.markdown('<div class="sec-title">Distribución descuentos</div>', unsafe_allow_html=True)
        dd = df_f['descuento_pct'].apply(lambda x:f"{int(x*100)}%").value_counts().reset_index()
        dd.columns = ['Descuento','Pedidos']
        fig = go.Figure(go.Pie(
            labels=dd['Descuento'], values=dd['Pedidos'], hole=0.55,
            marker=dict(colors=PALETTE, line=dict(color='white',width=3)),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>%{value:,} pedidos<extra></extra>'
        ))
        fig.update_layout(**make_layout(height=300, showlegend=False,
                          margin=dict(l=10,r=10,t=30,b=10)))
        st.plotly_chart(fig, width="stretch")

# ═══════════════════════════════════════════════════════
# P3 — SEGMENTACIÓN RFM
# ═══════════════════════════════════════════════════════
elif seccion == "👥  Segmentación RFM":

    ss = rfm_f.groupby('segmento_nombre').agg(
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

    cols_s = st.columns(4)
    for i,(_, row) in enumerate(ss.iterrows()):
        clr = SEG_COL.get(row['segmento_nombre'], TEXT_L)
        pct = row['clientes']/rfm_f.shape[0]*100 if rfm_f.shape[0] else 0
        ac  = acciones.get(row['segmento_nombre'], '')
        with cols_s[i]:
            st.markdown(
                f'<div class="kpi" style="border-top:4px solid {clr};">'
                f'<div class="kpi-label">{row["segmento_nombre"]}</div>'
                f'<div class="kpi-main" style="color:{clr};">{int(row["clientes"]):,}</div>'
                f'<div style="font-family:DM Mono,monospace;font-size:0.6rem;'
                f'color:{TEXT_L};">{pct:.1f}% del total</div>'
                f'<div class="kpi-sub">'
                f'<div><div class="kpi-sub-label">Recencia</div>'
                f'<div class="kpi-sub-value">{row["rec_media"]:.0f}d</div></div>'
                f'<div><div class="kpi-sub-label">Frec.</div>'
                f'<div class="kpi-sub-value">{row["frec_media"]:.1f}x</div></div>'
                f'<div><div class="kpi-sub-label">Gasto</div>'
                f'<div class="kpi-sub-value">{row["mon_media"]:.0f}€</div></div>'
                f'</div>'
                f'<div style="font-size:0.68rem;color:{TEXT_L};margin-top:8px;'
                f'font-style:italic;">{ac}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_sc, col_rd = st.columns([3,2])

    with col_sc:
        seg_sel2 = st.selectbox("Resaltar segmento", ["Todos"]+segs_a)
        fig = go.Figure()
        for seg, clr in SEG_COL.items():
            sub   = rfm_f[rfm_f['segmento_nombre']==seg]
            alpha = 0.85 if seg_sel2 in ("Todos", seg) else 0.07
            fig.add_trace(go.Scatter(
                x=sub['frecuencia'], y=sub['monetario'], mode='markers', name=seg,
                marker=dict(color=clr,size=5,opacity=alpha,line_width=0),
                hovertemplate='Frec: %{x}<br>Gasto: %{y:,.0f}€<extra>'+seg+'</extra>'
            ))
        fig.update_layout(**make_layout(height=380,
                          xaxis_title='Frecuencia', yaxis_title='Gasto total (€)'))
        st.plotly_chart(fig, width="stretch")

    with col_rd:
        rd = rfm_f.groupby('segmento_nombre')[['recencia','frecuencia','monetario']].mean()
        if len(rd) > 0:
            rn = (rd-rd.min())/(rd.max()-rd.min()).replace(0,1)
            rn['recencia'] = 1-rn['recencia']
            rn.columns     = ['Actualidad','Frecuencia','Valor']
            cats_r = list(rn.columns)+[rn.columns[0]]
            fig = go.Figure()
            for seg, clr in SEG_COL.items():
                if seg in rn.index:
                    vals = list(rn.loc[seg])+[rn.loc[seg].iloc[0]]
                    fig.add_trace(go.Scatterpolar(
                        r=vals, theta=cats_r, name=seg, fill='toself',
                        fillcolor=hex_rgba(clr,0.12),
                        line=dict(color=clr,width=2),
                    ))
            fig.update_layout(
                paper_bgcolor=BG_PLOT, plot_bgcolor=BG_PLOT,
                font=dict(color=TEXT_M,family='DM Sans'),
                polar=dict(bgcolor=BG_PLOT,
                           radialaxis=dict(visible=True,range=[0,1],
                                           gridcolor=CREAM_DD,color=TEXT_L),
                           angularaxis=dict(gridcolor=CREAM_DD,color=TEXT_M)),
                height=380, showlegend=True,
                legend=dict(bgcolor='rgba(0,0,0,0)',font_color=TEXT_M,font_size=10),
                margin=dict(l=30,r=80,t=40,b=30)
            )
            st.plotly_chart(fig, width="stretch")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Validación ML vs perfil real</div>', unsafe_allow_html=True)
    val   = df.groupby('customer_id')['perfil_real'].first().reset_index()
    rfm_v = rfm_f.merge(val, on='customer_id')
    if len(rfm_v):
        tv2 = pd.crosstab(rfm_v['segmento_nombre'],rfm_v['perfil_real'],
                          normalize='index').round(3)*100
        fig = go.Figure(go.Heatmap(
            z=tv2.values, x=tv2.columns.tolist(), y=tv2.index.tolist(),
            colorscale=[[0,CREAM],[0.5,LILAC_L],[1,LILAC_D]],
            text=[[f'{v:.1f}%' for v in row] for row in tv2.values],
            texttemplate='%{text}', showscale=True,
            hovertemplate='ML: %{y}<br>Real: %{x}<br>%{z:.1f}%<extra></extra>'
        ))
        fig.update_layout(**make_layout(height=300))
        st.plotly_chart(fig, width="stretch")

# ═══════════════════════════════════════════════════════
# P4 — RECOMENDADOR
# ═══════════════════════════════════════════════════════
elif seccion == "🤖  Recomendador":

    with st.spinner("Construyendo motor de recomendación..."):
        df_cat, sprod, scli, muestra, top_seg = construir_motor(df, rfm)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Capas del sistema
    c1,c2,c3 = st.columns(3)
    capas = [
        (c1,"Capa 1 · 40%","Por Segmento RFM",
         "Qué compran los clientes del mismo segmento. Más peso en clientes Dormidos y En Riesgo.", LILAC),
        (c2,"Capa 2 · 40%","Filtrado Colaborativo",
         "Clientes con perfil de compra similar compraron estos productos. Similitud coseno.", ROSE),
        (c3,"Capa 3 · 20%","Basado en Contenido",
         "Productos similares a los ya comprados. TF-IDF sobre ingredientes y beneficios.", SAGE),
    ]
    for col_,lbl,tit,desc,clr in capas:
        with col_:
            st.markdown(
                f'<div class="kpi" style="border-top:4px solid {clr};">'
                f'<div class="kpi-label">{lbl}</div>'
                f'<div style="font-family:Cormorant Garamond,serif;font-size:1.2rem;'
                f'font-weight:500;color:{clr};margin:8px 0 10px;">{tit}</div>'
                f'<div style="font-size:0.82rem;color:{TEXT_M};line-height:1.6;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Consulta por cliente</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Introduce un ID para ver recomendaciones personalizadas</div>', unsafe_allow_html=True)

    col_i1, col_i2 = st.columns([3,1])
    with col_i1:
        cust_in = st.text_input("id", placeholder="ej: CL00001", label_visibility="collapsed")
    with col_i2:
        top_nr = st.slider("Nº recs", 3, 10, 5)

    btn = st.button("✦ Recomendar")

    st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:0.6rem;color:{TEXT_L};margin:10px 0 6px;letter-spacing:.1em;text-transform:uppercase;'>Prueba rápida</div>", unsafe_allow_html=True)
    cols_ej = st.columns(4)
    ej_map = {
        "VIP":       rfm[rfm['segmento_nombre']=='VIP 🏆']['customer_id'].iloc[0],
        "Fiel":      rfm[rfm['segmento_nombre']=='Fiel 💛']['customer_id'].iloc[0],
        "En Riesgo": rfm[rfm['segmento_nombre']=='En Riesgo ⚠️']['customer_id'].iloc[0],
        "Dormido":   rfm[rfm['segmento_nombre']=='Dormidos 😴']['customer_id'].iloc[0],
    }
    for col_,(lej,cej) in zip(cols_ej, ej_map.items()):
        with col_:
            if st.button(f"{lej} · {cej}", width="stretch", key=f"ej_{cej}"):
                cust_in = cej; btn = True

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    if btn and cust_in:
        cid = cust_in.strip().upper()
        if cid not in df['customer_id'].values:
            st.error(f"Cliente '{cid}' no encontrado. Formato: CL00001")
        else:
            recs, seg = recomendar(cid, rfm, df, df_cat, sprod, scli, muestra, top_seg, n=top_nr)
            ic   = rfm[rfm['customer_id']==cid].iloc[0]
            comp = df[df['customer_id']==cid]['producto'].unique().tolist()
            cs   = SEG_COL.get(seg, TEXT_L)

            st.markdown(f'<div class="sec-title">Perfil — {cid}</div>', unsafe_allow_html=True)
            p1,p2,p3,p4,p5 = st.columns(5)
            mets = [
                ("Segmento",   seg,                        cs),
                ("Recencia",   f"{ic['recencia']} días",   LILAC),
                ("Frecuencia", f"{ic['frecuencia']} compras", SAGE),
                ("Gasto total",f"{ic['monetario']:,.0f}€", ROSE),
                ("Productos",  f"{len(comp)} distintos",   MAUVE),
            ]
            for col_,(l2,v2,c2) in zip([p1,p2,p3,p4,p5], mets):
                with col_:
                    st.markdown(
                        f'<div class="kpi" style="border-top:4px solid {c2};">'
                        f'<div class="kpi-label">{l2}</div>'
                        f'<div class="kpi-main" style="font-size:1.3rem;color:{c2};">{v2}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            ch, cr = st.columns([1,1.5])

            with ch:
                st.markdown('<div class="sec-title" style="font-size:1.3rem">Historial</div>', unsafe_allow_html=True)
                hist = (df[df['customer_id']==cid].groupby('producto')
                        .agg(gasto=('importe','sum'))
                        .sort_values('gasto',ascending=False).reset_index())
                fig_h = go.Figure(go.Bar(
                    x=hist['gasto'], y=hist['producto'], orientation='h',
                    marker=dict(color=hist['gasto'],
                                colorscale=[[0,LILAC_XL],[1,LILAC_D]],
                                showscale=False, line_width=0),
                    hovertemplate='<b>%{y}</b><br>%{x:,.0f}€<extra></extra>'
                ))
                fig_h.update_layout(**make_layout(height=max(200,len(hist)*45), showlegend=False))
                fig_h.update_yaxes(gridcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_h, width="stretch")

            with cr:
                st.markdown(f'<div class="sec-title" style="font-size:1.3rem">Top {top_nr} recomendaciones</div>', unsafe_allow_html=True)
                ms2 = recs['score'].max() if recs['score'].max()>0 else 1
                for i,(_,row) in enumerate(recs.iterrows()):
                    bw  = int((row['score']/ms2)*100)
                    cc  = CAT_COL.get(row['categoria'], TEXT_L)
                    st.markdown(
                        f'<div class="rec-card">'
                        f'<div style="position:absolute;bottom:0;left:0;height:3px;'
                        f'width:{bw}%;background:{cc};border-radius:0 0 0 16px;"></div>'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<div>'
                        f'<span style="font-family:DM Mono,monospace;font-size:0.6rem;'
                        f'color:{TEXT_L};margin-right:8px;">#{i+1}</span>'
                        f'<span style="font-weight:600;color:{TEXT_D};">{row["producto"]}</span>'
                        f'</div>'
                        f'<span style="font-family:Cormorant Garamond,serif;font-size:1.2rem;'
                        f'font-weight:500;color:{cc};">{row["precio"]:.2f}€</span>'
                        f'</div>'
                        f'<div style="margin-top:8px;display:flex;align-items:center;gap:10px;">'
                        f'<span class="tag">{row["categoria"]}</span>'
                        f'<span style="font-family:DM Mono,monospace;font-size:0.6rem;'
                        f'color:{TEXT_L};">score {row["score"]:.3f}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Similitud entre productos</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-sub">Historial del cliente + productos recomendados</div>', unsafe_allow_html=True)

            pm = list(comp[:6]) + recs['producto'].tolist()[:4]
            pm = list(dict.fromkeys(pm))
            pm = [p for p in pm if p in sprod.index][:10]
            if len(pm) > 1:
                ss2   = sprod.loc[pm,pm]
                etiq  = [p[:20]+'...' if len(p)>20 else p for p in ss2.index]
                fig_s = go.Figure(go.Heatmap(
                    z=ss2.values, x=etiq, y=etiq,
                    colorscale=[[0,CREAM],[0.5,LILAC_L],[1,LILAC_D]],
                    text=[[f'{v:.2f}' for v in fila] for fila in ss2.values],
                    texttemplate='%{text}', showscale=True,
                    hovertemplate='%{y} ↔ %{x}<br>%{z:.3f}<extra></extra>'
                ))
                fig_s.update_layout(**make_layout(height=380))
                fig_s.update_layout(xaxis=dict(tickangle=-35,tickfont_size=9,gridcolor=CREAM_DD))
                fig_s.update_layout(yaxis=dict(tickfont_size=9,gridcolor=CREAM_DD))
                st.plotly_chart(fig_s, width="stretch")
    else:
        st.markdown(
            f'<div style="background:white;border:1.5px dashed {CREAM_DD};'
            f'border-radius:20px;padding:64px;text-align:center;margin-top:8px;">'
            f'<div style="font-size:2rem;margin-bottom:12px;">✦</div>'
            f'<div style="font-family:Cormorant Garamond,serif;font-size:1.5rem;'
            f'font-weight:400;font-style:italic;color:{TEXT_D};">'
            f'Introduce un ID de cliente</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.6rem;'
            f'color:{TEXT_L};margin-top:8px;letter-spacing:.1em;">'
            f'FORMATO: CL00001 — CL05000</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════════════
# P5 — ALERTAS CHURN
# ═══════════════════════════════════════════════════════
elif seccion == "🚨  Alertas Churn":

    with st.spinner("Entrenando modelo de churn (Gradient Boosting)..."):
        ch, auc, fi, fpr, tpr = calcular_churn(df, rfm)

    RIESGO_COL = {'Bajo': SAGE, 'Medio': LILAC, 'Alto': ROSE, 'Crítico': '#B0305A'}
    RIESGO_BG  = {'Bajo': '#F0FAF5', 'Medio': LILAC_XL, 'Alto': '#FEF0F3', 'Crítico': '#FDE8EF'}

    # ── KPIs ──
    n_critico = (ch['riesgo'] == 'Crítico').sum()
    n_alto    = (ch['riesgo'] == 'Alto').sum()
    n_medio   = (ch['riesgo'] == 'Medio').sum()
    n_bajo    = (ch['riesgo'] == 'Bajo').sum()
    gasto_riesgo = ch[ch['riesgo'].isin(['Alto','Crítico'])]['monetario'].sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi_churn = [
        (c1, "AUC-ROC del modelo",  f"{auc:.3f}", LILAC_D,
         "Gradient Boosting · 200 estimadores"),
        (c2, "Riesgo Crítico",      f"{n_critico:,}", '#B0305A',
         "Prob. churn > 75% · acción urgente"),
        (c3, "Riesgo Alto",         f"{n_alto:,}", ROSE,
         "Prob. churn 55–75% · intervenir"),
        (c4, "Riesgo Medio",        f"{n_medio:,}", LILAC,
         "Prob. churn 30–55% · vigilar"),
        (c5, "Gasto en riesgo",     f"{gasto_riesgo/1e3:.0f}k€", MAUVE,
         "Facturación histórica de clientes Alto+Crítico"),
    ]
    for col_, lbl, val, clr, desc in kpi_churn:
        with col_:
            st.markdown(
                f'<div class="kpi" style="border-top:4px solid {clr};">'
                f'<div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-main" style="color:{clr};font-size:2rem;">{val}</div>'
                f'<div style="font-size:0.7rem;color:{TEXT_L};margin-top:6px;">{desc}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ── Fila 1: Distribución + Curva ROC ──
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<div class="sec-title">Distribución de riesgo</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">5.000 clientes clasificados en 4 niveles</div>', unsafe_allow_html=True)

        dist = ch['riesgo'].value_counts().reindex(['Crítico','Alto','Medio','Bajo']).reset_index()
        dist.columns = ['Riesgo','Clientes']
        dist['Color'] = dist['Riesgo'].map(RIESGO_COL)
        dist['Pct']   = (dist['Clientes'] / len(ch) * 100).round(1)

        fig = go.Figure(go.Bar(
            y=dist['Riesgo'], x=dist['Clientes'], orientation='h',
            marker=dict(color=dist['Color'], line_width=0),
            text=[f"{r} ({p}%)" for r, p in zip(dist['Clientes'], dist['Pct'])],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>%{x:,} clientes<extra></extra>'
        ))
        fig.update_layout(**make_layout(height=280, showlegend=False,
                          xaxis_title='Nº de clientes'))
        fig.update_yaxes(gridcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width="stretch")

    with col_r:
        st.markdown('<div class="sec-title">Curva ROC</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-sub">AUC = {auc:.4f} · Gradient Boosting</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            line=dict(color=CREAM_DD, width=1.5, dash='dash'),
            showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', name=f'GBM (AUC={auc:.3f})',
            line=dict(color=LILAC_D, width=2.5),
            fill='tozeroy', fillcolor=hex_rgba(LILAC, 0.1),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'))
        fig.update_layout(**make_layout(height=280,
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate'))
        st.plotly_chart(fig, width="stretch")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ── Fila 2: Feature importance + Riesgo por segmento ──
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="sec-title">Variables más importantes</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Feature importance del modelo GBM</div>', unsafe_allow_html=True)

        fi_top = fi.head(8).sort_values('importance')
        fig = go.Figure(go.Bar(
            y=fi_top['feature'], x=fi_top['importance'], orientation='h',
            marker=dict(
                color=fi_top['importance'],
                colorscale=[[0, LILAC_XL], [1, LILAC_D]],
                showscale=False, line_width=0),
            hovertemplate='<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>'
        ))
        fig.update_layout(**make_layout(height=300, showlegend=False,
                          xaxis_title='Importancia relativa'))
        fig.update_yaxes(gridcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.markdown('<div class="sec-title">Riesgo por segmento RFM</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Composición de riesgo dentro de cada segmento</div>', unsafe_allow_html=True)

        seg_riesgo = (ch.groupby(['segmento_nombre','riesgo'])
                      .size().reset_index(name='n'))
        seg_total  = ch.groupby('segmento_nombre')['customer_id'].count().reset_index(name='total')
        seg_riesgo = seg_riesgo.merge(seg_total, on='segmento_nombre')
        seg_riesgo['pct'] = seg_riesgo['n'] / seg_riesgo['total'] * 100

        fig = go.Figure()
        for nivel in ['Bajo','Medio','Alto','Crítico']:
            sub = seg_riesgo[seg_riesgo['riesgo']==nivel]
            fig.add_trace(go.Bar(
                name=nivel, x=sub['segmento_nombre'], y=sub['pct'],
                marker=dict(color=RIESGO_COL.get(nivel, TEXT_L), line_width=0),
                hovertemplate=f'<b>%{{x}}</b> · {nivel}<br>%{{y:.1f}}%<extra></extra>'
            ))
        fig.update_layout(**make_layout(height=300, barmode='stack',
                          yaxis_title='% clientes'))
        st.plotly_chart(fig, width="stretch")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ── Lista de alertas ──
    st.markdown('<div class="sec-title">Lista de alertas por nivel de riesgo</div>', unsafe_allow_html=True)

    nivel_sel = st.selectbox(
        "Filtrar nivel",
        ['Crítico', 'Alto', 'Medio', 'Bajo'],
        index=0,
        key='churn_nivel'
    )
    n_mostrar = st.slider("Clientes a mostrar", 10, 100, 25, step=5, key='churn_n')

    alerta_df = (ch[ch['riesgo']==nivel_sel]
                 .sort_values('prob_churn', ascending=False)
                 .head(n_mostrar)
                 [['customer_id','segmento_nombre','perfil_real','recencia',
                   'frecuencia','monetario','prob_churn','riesgo']]
                 .copy())

    alerta_df.columns = ['Cliente','Segmento RFM','Perfil real','Recencia (d)',
                         'Frecuencia','Gasto (€)','Prob. Churn','Riesgo']
    alerta_df['Prob. Churn'] = (alerta_df['Prob. Churn'] * 100).round(1).astype(str) + '%'
    alerta_df['Gasto (€)']   = alerta_df['Gasto (€)'].apply(lambda x: f'{x:,.0f}€')

    clr_nivel = RIESGO_COL.get(nivel_sel, TEXT_L)
    bg_nivel  = RIESGO_BG.get(nivel_sel, CREAM)

    acciones = {
        'Crítico': '🔴 Win-back agresivo · descuento >20% + regalo · contacto directo',
        'Alto':    '🟠 Descuento 15% · email personalizado · llamada proactiva',
        'Medio':   '🟡 Email con oferta suave · recordatorio de productos favoritos',
        'Bajo':    '🟢 Mantenimiento estándar · newsletter mensual',
    }
    st.markdown(
        f'<div style="background:{bg_nivel};border-left:4px solid {clr_nivel};'
        f'border-radius:0 12px 12px 0;padding:12px 18px;margin-bottom:16px;">'
        f'<span style="font-family:DM Mono,monospace;font-size:0.65rem;'
        f'color:{clr_nivel};font-weight:700;letter-spacing:.1em;">ACCIÓN RECOMENDADA</span><br>'
        f'<span style="font-size:0.9rem;color:{TEXT_D};">{acciones[nivel_sel]}</span>'
        f'</div>',
        unsafe_allow_html=True)

    st.dataframe(alerta_df, width="stretch", height=420, hide_index=True)

    st.markdown(
        f'<div style="font-size:0.78rem;color:{TEXT_L};margin-top:8px;">'
        f'Mostrando <b style="color:{clr_nivel}">{len(alerta_df)}</b> clientes de nivel '
        f'<b style="color:{clr_nivel}">{nivel_sel}</b> · '
        f'Total en este nivel: <b>{(ch["riesgo"]==nivel_sel).sum():,}</b></div>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# P6 — TABLA DE DETALLE
# ═══════════════════════════════════════════════════════
elif seccion == "📋  Tabla de Detalle":

    ct1,ct2,ct3 = st.columns(3)
    with ct1:
        srch = st.text_input("Buscar producto", placeholder="ej: Sérum...")
    with ct2:
        ord_ = st.selectbox("Ordenar por", ['importe','cantidad','descuento_pct','fecha'])
    with ct3:
        nrow = st.slider("Filas", 10, 200, 50, step=10)

    dt = df_f.copy()
    if srch:
        dt = dt[dt['producto'].str.contains(srch,case=False,na=False)]
    dt = dt.sort_values(ord_,ascending=False).head(nrow)
    dt = dt.merge(rfm[['customer_id','segmento_nombre']],on='customer_id',how='left')

    ds = dt[['fecha','order_id','customer_id','segmento_nombre','ciudad','canal',
             'categoria','producto','cantidad','precio_unitario','descuento_pct','importe']].copy()
    ds['fecha']           = ds['fecha'].dt.strftime('%Y-%m-%d')
    ds['descuento_pct']   = (ds['descuento_pct']*100).round(0).astype(int).astype(str)+'%'
    ds['importe']         = ds['importe'].apply(lambda x:f'{x:,.2f}€')
    ds['precio_unitario'] = ds['precio_unitario'].apply(lambda x:f'{x:.2f}€')
    ds.columns = ['Fecha','Pedido','Cliente','Segmento','Ciudad','Canal',
                  'Categoría','Producto','Uds.','Precio unit.','Descuento','Importe']

    st.dataframe(ds, width="stretch", height=500, hide_index=True)

    cr1,cr2,cr3 = st.columns(3)
    with cr1:
        st.markdown(f"<div style='font-size:0.78rem;color:{TEXT_L};'>"
                    f"<b style='color:{TEXT_M}'>{len(dt):,}</b> registros mostrados</div>",
                    unsafe_allow_html=True)
    with cr2:
        tt = df_f.groupby('producto')['importe'].sum()
        pt = tt.idxmax() if len(tt) else '-'
        st.markdown(f"<div style='font-size:0.78rem;color:{TEXT_L};'>"
                    f"Top: <b style='color:{LILAC_D}'>{pt}</b></div>",
                    unsafe_allow_html=True)
    with cr3:
        st.markdown(f"<div style='font-size:0.78rem;color:{TEXT_L};'>"
                    f"Total: <b style='color:{SAGE}'>{df_f['importe'].sum():,.0f}€</b></div>",
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    f'<div style="text-align:center;margin-top:48px;padding-bottom:16px;'
    f'font-family:DM Mono,monospace;font-size:0.55rem;color:{CREAM_DD};'
    f'letter-spacing:.15em;text-transform:uppercase;">'
    f'Facegloss · Executive Dashboard · Proyecto IA &amp; Big Data · 2024'
    f'</div>',
    unsafe_allow_html=True
)
