import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings, os
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction Ventes · Dakar",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette du rapport HTML
ORANGE  = "#C2601A"
VIOLET  = "#7B4FA6"
AMBER   = "#E8832A"
PURPLE2 = "#9B6BC5"
BG      = "#FDF6F0"
SURFACE = "#FFFFFF"
SURFACE2= "#F5EEF8"
BORDER  = "#E8D5C4"
TEXT    = "#2D1B0E"
MUTED   = "#8B6A55"
DANGER  = "#DC2626"

PALETTE = [ORANGE, VIOLET, AMBER, PURPLE2, "#A67BC5", "#D97706", "#C2601A", "#7B4FA6"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

* {{ font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif !important; }}

/* Header caché */
header[data-testid="stHeader"], #MainMenu, footer {{ visibility: hidden; height:0; }}
.stDeployButton {{ display: none; }}

/* Bouton collapse sidebar (keyboard_double_arrow) */
button[data-testid="collapsedControl"],
button[kind="header"],
[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] button[kind="header"],
div[data-testid="stSidebarCollapsedControl"] {{
  display: none !important;
}}
/* Icône material icons dans sidebar */
.material-symbols-rounded,
span.material-symbols-rounded {{ display: none !important; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: #FFFFFF !important;
  border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {MUTED} !important; }}
section[data-testid="stSidebar"] h3 {{ color: {TEXT} !important; font-weight: 700; }}

/* Hero banner */
.hero-banner {{
  background: linear-gradient(135deg, #FDF0E6 0%, #F0E6FA 50%, #FDF0E6 100%);
  border: 1px solid {BORDER}; border-radius: 14px;
  padding: 28px 32px 22px; margin-bottom: 24px;
  position: relative; overflow: hidden;
}}
.hero-banner::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(ellipse at 20% 50%, rgba(194,96,26,0.10) 0%, transparent 60%),
              radial-gradient(ellipse at 80% 20%, rgba(123,79,166,0.10) 0%, transparent 60%);
}}
.hero-badge {{
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(194,96,26,0.12); border: 1px solid rgba(194,96,26,0.4);
  color: {ORANGE}; padding: 4px 12px; border-radius: 20px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; margin-bottom: 10px;
}}
.hero-title {{
  font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;
  color: {TEXT} !important; line-height: 1.2;
}}
.hero-title span {{
  background: linear-gradient(90deg, {ORANGE}, {VIOLET});
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero-sub {{ font-size: 13px; color: {MUTED}; margin-top: 8px; }}

/* KPI cards */
.metric-card {{
  background: #FFFFFF; border: 1px solid {BORDER};
  border-radius: 10px; padding: 18px 20px;
  position: relative; overflow: hidden;
}}
.metric-card::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: var(--accent-c);
}}
.metric-card.c-orange {{ --accent-c: {ORANGE}; }}
.metric-card.c-violet {{ --accent-c: {VIOLET}; }}
.metric-card.c-amber  {{ --accent-c: {AMBER};  }}
.metric-card.c-purple {{ --accent-c: {PURPLE2};}}
.metric-label {{ font-size: 11px; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }}
.metric-value {{ font-size: 1.8rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }}
.metric-value.orange {{ color: {ORANGE}; }}
.metric-value.violet {{ color: {VIOLET}; }}
.metric-value.amber  {{ color: {AMBER};  }}
.metric-value.purple {{ color: {PURPLE2};}}
.metric-sub {{ font-size: 11px; color: {MUTED}; margin-top: 4px; }}
.metric-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; margin-top: 6px; font-family: 'IBM Plex Mono', monospace; }}
.mb-orange {{ background: rgba(194,96,26,0.12); color: {ORANGE}; }}
.mb-violet {{ background: rgba(123,79,166,0.15); color: {VIOLET}; }}
.mb-amber  {{ background: rgba(232,131,42,0.15); color: {AMBER};  }}
.mb-green  {{ background: rgba(123,79,166,0.10); color: {PURPLE2};}}

/* Section header */
.section-label {{ font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: {VIOLET}; margin-bottom: 4px; font-family: 'IBM Plex Mono', monospace; }}
.section-title {{ font-size: 1.1rem; font-weight: 600; color: {TEXT} !important; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid {BORDER}; }}

/* Insight */
.insight {{ background: {SURFACE2}; border-left: 3px solid {VIOLET}; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 10px 0; font-size: 13px; color: {MUTED}; }}
.insight b {{ color: {VIOLET}; }}

/* Pred box */
.pred-box {{ background: linear-gradient(135deg, #FDF0E6, #F0E6FA); border: 1px solid {BORDER}; border-radius: 14px; padding: 32px; text-align: center; margin: 10px 0; }}
.pred-label {{ font-size: 11px; font-weight: 600; color: {ORANGE}; letter-spacing: 0.10em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace; margin-bottom: 10px; }}
.pred-value {{ font-size: 56px; font-weight: 700; color: {TEXT}; font-family: 'IBM Plex Mono', monospace; line-height: 1; }}
.pred-unit  {{ font-size: 16px; color: {MUTED}; margin-top: 6px; }}
.pred-range {{ font-size: 12px; color: {MUTED}; margin-top: 8px; }}

/* Reco card */
.reco-card {{ background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 10px; padding: 20px 18px; }}
.reco-icon {{ font-size: 1.5rem; margin-bottom: 8px; }}
.reco-card h4 {{ font-size: 13px; font-weight: 600; color: {TEXT}; margin-bottom: 6px; }}
.reco-card p  {{ font-size: 12px; color: {MUTED}; line-height: 1.7; }}

/* Multiselect tags */
span[data-baseweb="tag"] {{
  background-color: rgba(194,96,26,0.12) !important;
  border: 1px solid rgba(194,96,26,0.35) !important;
  border-radius: 6px !important;
}}
span[data-baseweb="tag"] span {{ color: {ORANGE} !important; font-size: 12px !important; font-weight: 600 !important; }}
span[data-baseweb="tag"] svg {{ fill: {ORANGE} !important; }}

/* Tabs */
.stTabs [data-baseweb="tab"] {{ color: {MUTED} !important; }}
.stTabs [aria-selected="true"] {{ color: {ORANGE} !important; border-bottom: 2px solid {ORANGE} !important; }}

/* Button */
.stButton > button {{ background: {ORANGE} !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 10px 24px !important; font-weight: 600 !important; font-size: 14px !important; width: 100% !important; }}
.stButton > button:hover {{ background: #A8521A !important; }}

/* Footer */
.footer-bar {{ border-top: 1px solid {BORDER}; background: linear-gradient(90deg,#FDF0E6,#F0E6FA); padding: 16px 0; text-align: center; color: {MUTED}; font-size: 12px; margin-top: 2rem; border-radius: 0 0 10px 10px; }}
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def kpi(label, value, sub="", badge="", color="orange"):
    b = f'<div class="metric-badge mb-{color}">{badge}</div>' if badge else ""
    return f"""<div class="metric-card c-{color}">
      <div class="metric-label">{label}</div>
      <div class="metric-value {color}">{value}</div>
      <div class="metric-sub">{sub}</div>{b}</div>"""

def section(title, label=""):
    lbl = f'<div class="section-label">{label}</div>' if label else ""
    st.markdown(f'{lbl}<div class="section-title">{title}</div>', unsafe_allow_html=True)

def light_fig(fig, height=360):
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor="#FDFAF7",
        font=dict(family="IBM Plex Sans", color=MUTED),
        margin=dict(l=10,r=10,t=10,b=10), height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)),
        xaxis=dict(showgrid=True, gridcolor="#EDE0D4", linecolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor="#EDE0D4", linecolor=BORDER, tickfont=dict(color=MUTED)),
        title=dict(text="", font=dict(color=TEXT, size=13, family="IBM Plex Sans"))
    )
    return fig

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = "ventes_dakar.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "ventes_dakar.csv")
    df = pd.read_csv(csv_path)
    df.rename(columns={"Ventes_S-1": "Ventes_S1"}, inplace=True)
    return df

@st.cache_resource
def train_models(df):
    df_m = df.copy()
    le_cat = LabelEncoder(); df_m['Catégorie_enc'] = le_cat.fit_transform(df_m['Catégorie'])
    le_mar = LabelEncoder(); df_m['Marque_enc']    = le_mar.fit_transform(df_m['Marque'])
    le_met = LabelEncoder(); df_m['Météo_enc']     = le_met.fit_transform(df_m['Météo'])

    df_m['Semaine_sin']     = np.sin(2*np.pi*df_m['Semaine']/52)
    df_m['Semaine_cos']     = np.cos(2*np.pi*df_m['Semaine']/52)
    df_m['Mois_sin']        = np.sin(2*np.pi*df_m['Mois']/12)
    df_m['Mois_cos']        = np.cos(2*np.pi*df_m['Mois']/12)
    df_m['Promo_Réduction'] = df_m['Promotion']*df_m['Réduction_%']
    df_m['Ferie_Promo']     = df_m['Jour_férié']*df_m['Promotion']
    df_m['Stock_ratio']     = df_m['Stock_début']/(df_m['Ventes_S1']+1)
    df_m['Haute_saison']    = ((df_m['Semaine'].between(10,14))|(df_m['Semaine']>=50)).astype(int)

    FEATURES = ['Prix','Promotion','Réduction_%','Jour_férié','Ventes_S1','Stock_début',
                'Catégorie_enc','Marque_enc','Météo_enc',
                'Semaine_sin','Semaine_cos','Mois_sin','Mois_cos',
                'Promo_Réduction','Ferie_Promo','Stock_ratio','Haute_saison']

    X, y = df_m[FEATURES], df_m['Ventes']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train)
    Xte_sc = scaler.transform(X_test)

    def ev(name, model, Xtr, ytr, Xte, yte):
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)
        return {'Modèle':name,
                'MAE':   round(mean_absolute_error(yte,yp),2),
                'RMSE':  round(np.sqrt(mean_squared_error(yte,yp)),2),
                'R²':    round(r2_score(yte,yp),4),
                'MAPE':  round(np.mean(np.abs((yte-yp)/yte))*100,2),
                'CV-R²': round(cross_val_score(model,Xtr,ytr,cv=5,scoring='r2').mean(),4),
                'y_pred':yp,'model':model,'y_test':yte,'X_test':Xte}

    results = [
        ev('Régression Linéaire', LinearRegression(),   Xtr_sc,y_train,Xte_sc,y_test),
        ev('Ridge',               Ridge(alpha=1.0),      Xtr_sc,y_train,Xte_sc,y_test),
        ev('Random Forest',       RandomForestRegressor(n_estimators=200,max_depth=12,
                                  min_samples_leaf=4,random_state=42,n_jobs=-1),
                                  X_train,y_train,X_test,y_test),
        ev('Gradient Boosting',   GradientBoostingRegressor(n_estimators=250,
                                  learning_rate=0.07,max_depth=5,random_state=42),
                                  X_train,y_train,X_test,y_test),
    ]
    encoders = {'cat':le_cat,'mar':le_mar,'met':le_met}
    return results, FEATURES, X_test, y_test, df_m, scaler, encoders

# ─── LOAD ────────────────────────────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Fichier **ventes_dakar.csv** introuvable. Placez-le dans le même dossier que app.py.")
    st.stop()

results, FEATURES, X_test, y_test, df_m, scaler, encoders = train_models(df)
best = max(results, key=lambda r: r['R²'])

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<h3 style="color:{TEXT}!important;font-size:15px;font-weight:700;margin-bottom:2px">🛒 Ventes Dakar</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:12px;color:{MUTED}">Machine Learning · Commerce Retail</p>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Tableau de bord",
        "📊 Analyse EDA",
        "🤖 Modélisation ML",
        "🎯 Prédire les ventes",
        "💡 Recommandations"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f'<p style="font-size:11px;font-weight:600;color:{MUTED};text-transform:uppercase;letter-spacing:.05em">Filtres</p>', unsafe_allow_html=True)
    cats    = st.multiselect("Catégories", df['Catégorie'].unique().tolist(), default=df['Catégorie'].unique().tolist())
    meteos  = st.multiselect("Météo",      df['Météo'].unique().tolist(),     default=df['Météo'].unique().tolist())
    sem_rng = st.slider("Semaines", 1, 52, (1, 52))
    st.markdown("---")
    st.markdown(f'<p style="font-size:11px;color:{MUTED}">Dataset : <b style="color:{ORANGE}">{len(df):,} obs.</b> · {df.shape[1]} variables</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:11px;color:{MUTED}">Meilleur modèle : <b style="color:{VIOLET}">{best["Modèle"]}</b></p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:11px;color:{MUTED}">R² = <b style="color:{ORANGE}">{best["R²"]:.4f}</b> · MAPE = <b style="color:{AMBER}">{best["MAPE"]:.1f}%</b></p>', unsafe_allow_html=True)

df_f = df[df['Catégorie'].isin(cats) & df['Météo'].isin(meteos) & df['Semaine'].between(*sem_rng)]

# Garde : si filtres trop restrictifs
if df_f.empty:
    st.warning("⚠️ Aucune donnée avec ces filtres. Veuillez élargir la sélection.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TABLEAU DE BORD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Tableau de bord":
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-badge">Machine Learning · Commerce / Retail</div>
      <div class="hero-title">Prédiction des Ventes · <span>Supermarché Dakar</span></div>
      <div class="hero-sub">
        {len(df_f):,} observations · {df_f['Catégorie'].nunique()} catégories · {df_f['Marque'].nunique()} marques · 17 features
      </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi("Ventes totales",   f"{df_f['Ventes'].sum():,}", "unités",               "Dataset réel","orange"), unsafe_allow_html=True)
    c2.markdown(kpi("Vente moyenne",    f"{df_f['Ventes'].mean():.0f}", f"±{df_f['Ventes'].std():.0f} σ", "Par obs.","violet"), unsafe_allow_html=True)
    c3.markdown(kpi("Taux promo",       f"{df_f['Promotion'].mean()*100:.1f}%", "des produits", "Actif","amber"), unsafe_allow_html=True)
    c4.markdown(kpi("Meilleur R²",      f"{best['R²']:.4f}", best['Modèle'],    f"MAPE {best['MAPE']:.1f}%","violet"), unsafe_allow_html=True)
    c5.markdown(kpi("Top catégorie",    df_f.groupby('Catégorie')['Ventes'].sum().idxmax(), "ventes cumulées","N°1","orange"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns([2,1])
    with col1:
        section("Évolution hebdomadaire des ventes", "Saisonnalité")
        vs = df_f.groupby('Semaine')['Ventes'].agg(['mean','std']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([vs['Semaine'],vs['Semaine'][::-1]]),
            y=pd.concat([vs['mean']+vs['std'],(vs['mean']-vs['std'])[::-1]]),
            fill='toself',fillcolor='rgba(194,96,26,0.10)',line=dict(color='rgba(0,0,0,0)'),name='±1σ'))
        fig.add_trace(go.Scatter(x=vs['Semaine'],y=vs['mean'],mode='lines',
            line=dict(color=ORANGE,width=2.5),name='Ventes moy.'))
        fig.add_vrect(x0=10,x1=14,fillcolor="rgba(123,79,166,0.10)",line_width=0,
            annotation_text="Ramadan",annotation=dict(font_size=10,font_color=VIOLET))
        fig.add_vrect(x0=50,x1=52,fillcolor="rgba(232,131,42,0.12)",line_width=0,
            annotation_text="Fêtes",annotation=dict(font_size=10,font_color=AMBER))
        st.plotly_chart(light_fig(fig,320), use_container_width=True)

    with col2:
        section("Répartition par catégorie", "Catégories")
        vcat = df_f.groupby('Catégorie')['Ventes'].sum().reset_index()
        fig = px.pie(vcat,values='Ventes',names='Catégorie',
            color_discrete_sequence=PALETTE,hole=0.52)
        fig.update_traces(textposition='outside',textfont_size=10,textfont_color=MUTED)
        fig.update_layout(paper_bgcolor=SURFACE,margin=dict(l=0,r=0,t=10,b=10),height=320,
            legend=dict(font=dict(size=10,color=MUTED),bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        section("Impact des promotions", "Promo vs Sans promo")
        promo = df_f.groupby(['Catégorie','Promotion'])['Ventes'].mean().unstack().reset_index()
        promo.columns = ['Catégorie','Sans promo','Avec promo']
        promo['Gain'] = ((promo['Avec promo']-promo['Sans promo'])/promo['Sans promo']*100).round(1)
        fig = go.Figure()
        fig.add_bar(x=promo['Catégorie'],y=promo['Sans promo'],name='Sans promo',
            marker_color=BORDER,opacity=0.9,marker_line=dict(color=BORDER,width=0.5))
        fig.add_bar(x=promo['Catégorie'],y=promo['Avec promo'],name='Avec promo',
            marker_color=ORANGE,opacity=0.85,
            text=[f"+{g:.0f}%" for g in promo['Gain']],textposition='outside',
            textfont=dict(color=VIOLET,size=11))
        fig.update_layout(barmode='group')
        st.plotly_chart(light_fig(fig,300), use_container_width=True)

    with col4:
        section("Top 10 marques par ventes", "Classement")
        top_mar = df_f.groupby('Marque')['Ventes'].sum().sort_values(ascending=True).tail(10).reset_index()
        fig = px.bar(top_mar,x='Ventes',y='Marque',orientation='h',
            color='Ventes',color_continuous_scale=[[0,SURFACE2],[1,ORANGE]],
            labels={'Ventes':'Unités vendues'})
        fig.update_layout(coloraxis_showscale=False,showlegend=False)
        st.plotly_chart(light_fig(fig,300), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analyse EDA":
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-badge">Analyse Exploratoire · EDA</div>
      <div class="hero-title"><span>Exploration</span> des Données</div>
      <div class="hero-sub">{len(df_f):,} observations filtrées · Distribution, corrélations, saisonnalité</div>
    </div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(["📈 Distribution","📦 Catégories","🗓️ Saisonnalité","🔗 Corrélations"])

    with tab1:
        c1,c2 = st.columns(2)
        with c1:
            section("Distribution des ventes", "Histogramme")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df_f['Ventes'],nbinsx=45,
                marker_color=ORANGE,opacity=0.75,marker_line=dict(color=SURFACE,width=0.5)))
            fig.add_vline(x=df_f['Ventes'].mean(),line_dash='dash',line_color=VIOLET,
                annotation_text=f"Moy. {df_f['Ventes'].mean():.0f}",
                annotation=dict(font_color=VIOLET,font_size=11))
            fig.add_vline(x=df_f['Ventes'].median(),line_dash='dot',line_color=AMBER,
                annotation_text=f"Méd. {df_f['Ventes'].median():.0f}",
                annotation=dict(font_color=AMBER,font_size=11))
            st.plotly_chart(light_fig(fig,320), use_container_width=True)
        with c2:
            section("Boxplot par catégorie", "Dispersion")
            fig = px.box(df_f,x='Catégorie',y='Ventes',color='Catégorie',
                color_discrete_sequence=PALETTE,notched=True,points=False)
            fig.update_layout(showlegend=False)
            st.plotly_chart(light_fig(fig,320), use_container_width=True)

        st.markdown(f'<div class="insight">📌 <b>Insight :</b> Distribution à skewness positive — quelques semaines de très fortes ventes tirent la moyenne au-dessus de la médiane. Les Boissons dominent en volume.</div>', unsafe_allow_html=True)

        c3,c4 = st.columns(2)
        with c3:
            section("Réduction % vs Ventes", "Produits promus")
            fig = px.scatter(df_f[df_f['Promotion']==1].sample(min(600,len(df_f[df_f['Promotion']==1])),random_state=1),
                x='Réduction_%',y='Ventes',color='Catégorie',
                color_discrete_sequence=PALETTE,opacity=0.55,
                trendline='ols',trendline_scope='overall',
                trendline_color_override=VIOLET)
            st.plotly_chart(light_fig(fig,290), use_container_width=True)
        with c4:
            section("Prix vs Ventes", "Effet prix")
            fig = px.scatter(df_f.sample(min(500,len(df_f)),random_state=2),
                x='Prix',y='Ventes',color='Catégorie',
                color_discrete_sequence=PALETTE,opacity=0.5,log_x=True)
            st.plotly_chart(light_fig(fig,290), use_container_width=True)

    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            section("Volume total par catégorie", "Classement")
            vcat = df_f.groupby('Catégorie')['Ventes'].sum().sort_values().reset_index()
            fig = px.bar(vcat,x='Ventes',y='Catégorie',orientation='h',
                color='Catégorie',color_discrete_sequence=PALETTE,
                text='Ventes',labels={'Ventes':'Unités'})
            fig.update_traces(texttemplate='%{text:,}',textposition='outside',textfont_size=11,textfont_color=TEXT)
            fig.update_layout(showlegend=False)
            st.plotly_chart(light_fig(fig,310), use_container_width=True)
        with c2:
            section("Ventes moy. Marque × Catégorie", "Heatmap")
            hm = df_f.groupby(['Marque','Catégorie'])['Ventes'].mean().unstack().fillna(0).round(0)
            fig = px.imshow(hm,color_continuous_scale=[[0,SURFACE2],[1,VIOLET]],text_auto=True,aspect='auto')
            fig.update_layout(paper_bgcolor=SURFACE,margin=dict(l=0,r=0,t=10,b=10),height=310,
                font=dict(color=MUTED))
            st.plotly_chart(fig, use_container_width=True)

        section("Statistiques descriptives", "Par catégorie")
        desc = df_f.groupby('Catégorie')['Ventes'].describe().round(1)
        desc.columns = ['Nb obs.','Moyenne','Écart-type','Min','Q1','Médiane','Q3','Max']
        st.dataframe(desc.style.background_gradient(cmap='Oranges',subset=['Moyenne','Médiane']),
                     use_container_width=True)

    with tab3:
        c1,c2 = st.columns([3,2])
        with c1:
            section("Ventes hebdomadaires par catégorie", "52 semaines")
            vs_cat = df_f.groupby(['Semaine','Catégorie'])['Ventes'].mean().reset_index()
            fig = px.line(vs_cat,x='Semaine',y='Ventes',color='Catégorie',
                color_discrete_sequence=PALETTE,markers=False,labels={'Ventes':'Ventes moy.'})
            fig.add_vrect(x0=10,x1=14,fillcolor="rgba(123,79,166,0.08)",line_width=0)
            fig.add_vrect(x0=50,x1=52,fillcolor="rgba(232,131,42,0.10)",line_width=0)
            st.plotly_chart(light_fig(fig,320), use_container_width=True)
        with c2:
            section("Ventes par mois", "Saisonnalité")
            vm = df_f.groupby('Mois')['Ventes'].mean().reset_index()
            labels = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
            vm['Mois_label'] = vm['Mois'].apply(lambda x: labels[x-1])
            fig = px.bar(vm,x='Mois_label',y='Ventes',
                color='Ventes',color_continuous_scale=[[0,SURFACE2],[1,ORANGE]],
                labels={'Ventes':'Ventes moy.','Mois_label':'Mois'})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(light_fig(fig,320), use_container_width=True)

        section("Effet météo sur les ventes", "Météo × Catégorie")
        met_eff = df_f.groupby(['Météo','Catégorie'])['Ventes'].mean().unstack().fillna(0).round(1)
        fig = px.imshow(met_eff,color_continuous_scale=[[0,SURFACE2],[0.5,"#E8C4A8"],[1,ORANGE]],
            text_auto=True,aspect='auto')
        fig.update_layout(paper_bgcolor=SURFACE,margin=dict(l=0,r=0,t=10,b=10),height=280,
            font=dict(color=MUTED))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_cols = ['Ventes','Prix','Réduction_%','Semaine','Mois','Promotion','Jour_férié','Ventes_S1','Stock_début']
        corr = df_f[num_cols].corr().round(3)
        fig = px.imshow(corr,
            color_continuous_scale=[[0,VIOLET],[0.5,SURFACE],[1,ORANGE]],
            zmin=-1,zmax=1,text_auto=True,aspect='auto',labels=dict(color="Corrélation"))
        fig.update_layout(paper_bgcolor=SURFACE,margin=dict(l=0,r=0,t=10,b=10),height=480,
            font=dict(color=MUTED),coloraxis_colorbar=dict(thickness=14,len=0.8,tickfont=dict(color=MUTED)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="insight">📌 <b>Corrélations clés :</b> Promotion (+0.35), Réduction_% (+0.28) et Ventes_S1 (+0.45) sont les plus liées aux ventes. Prix : effet négatif modéré.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODÉLISATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Modélisation ML":
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-badge">Machine Learning · Régression</div>
      <div class="hero-title">Comparaison des <span>Modèles ML</span></div>
      <div class="hero-sub">4 algorithmes entraînés · {len(FEATURES)} features · Split 80/20 · Validation croisée 5 folds</div>
    </div>""", unsafe_allow_html=True)

    section("Comparaison des modèles", "Métriques d'évaluation")
    df_res = pd.DataFrame([{k:v for k,v in r.items() if k in ['Modèle','MAE','RMSE','R²','MAPE','CV-R²']} for r in results])
    df_res = df_res.sort_values('R²',ascending=False).reset_index(drop=True)

    def hl(val,col,s):
        if col in ['R²','CV-R²']:
            return f'background-color:rgba(123,79,166,0.12);color:{VIOLET};font-weight:600' if val==s.max() else ''
        return f'background-color:rgba(123,79,166,0.12);color:{VIOLET};font-weight:600' if val==s.min() else ''

    st.dataframe(df_res.style
        .apply(lambda s: [hl(v,s.name,s) for v in s], subset=['MAE','RMSE','R²','MAPE','CV-R²'])
        .format({'MAE':'{:.2f}','RMSE':'{:.2f}','R²':'{:.4f}','MAPE':'{:.2f}%','CV-R²':'{:.4f}'})
        .set_properties(**{'text-align':'center','font-family':'IBM Plex Mono, monospace','font-size':'12px'}),
        use_container_width=True, height=200)

    st.markdown(f'<div class="insight">🏆 <b>Meilleur modèle : {best["Modèle"]}</b> — R² = {best["R²"]:.4f} · MAE = {best["MAE"]:.1f} unités · MAPE = {best["MAPE"]:.1f}%</div>', unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["📉 Diagnostics","🔍 Importance variables","📦 Performance par catégorie"])

    with tab1:
        ytest = best['y_test']; ypred = best['y_pred']; resid = ytest.values - ypred
        c1,c2 = st.columns(2)
        with c1:
            section("Réel vs Prédit", "Qualité du modèle")
            lim = max(ytest.max(),ypred.max())*1.05
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ytest,y=ypred,mode='markers',
                marker=dict(color=ORANGE,opacity=0.30,size=5),name='Observations'))
            fig.add_trace(go.Scatter(x=[0,lim],y=[0,lim],mode='lines',
                line=dict(color=VIOLET,dash='dash',width=2),name='Idéal'))
            fig.update_xaxes(title='Réel'); fig.update_yaxes(title='Prédit')
            fig.add_annotation(x=0.08,y=0.9,xref='paper',yref='paper',
                text=f'R² = {best["R²"]:.4f}',showarrow=False,
                font=dict(size=13,color=VIOLET,family='IBM Plex Mono'),
                bgcolor=SURFACE,bordercolor=BORDER,borderwidth=1)
            st.plotly_chart(light_fig(fig,320), use_container_width=True)
        with c2:
            section("Distribution des résidus", "Biais du modèle")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=resid,nbinsx=45,
                marker_color=VIOLET,opacity=0.70,
                marker_line=dict(color=SURFACE,width=0.5)))
            fig.add_vline(x=0,line_dash='dash',line_color=ORANGE)
            fig.add_vline(x=resid.mean(),line_dash='dot',line_color=AMBER,
                annotation_text=f'Moy={resid.mean():.1f}',
                annotation=dict(font_color=AMBER,font_size=11))
            st.plotly_chart(light_fig(fig,320), use_container_width=True)

        c3,c4 = st.columns(2)
        with c3:
            section("Résidus vs Prédits", "Homoscédasticité")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ypred,y=resid,mode='markers',
                marker=dict(color=AMBER,opacity=0.30,size=5)))
            fig.add_hline(y=0,line_dash='dash',line_color=ORANGE)
            fig.update_xaxes(title='Prédites'); fig.update_yaxes(title='Résidu')
            st.plotly_chart(light_fig(fig,280), use_container_width=True)
        with c4:
            section("R² comparatif", "Tous les modèles")
            fig = px.bar(df_res,x='Modèle',y='R²',
                color='R²',color_continuous_scale=[[0,SURFACE2],[1,ORANGE]],
                text='R²',labels={'Modèle':''})
            fig.update_traces(texttemplate='%{text:.4f}',textposition='outside',
                textfont_size=11,textfont_color=TEXT)
            fig.update_layout(coloraxis_showscale=False,yaxis_range=[0,1.1])
            st.plotly_chart(light_fig(fig,280), use_container_width=True)

    with tab2:
        m = best['model']
        if hasattr(m,'feature_importances_'):
            imp = m.feature_importances_
        else:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(m,best['X_test'],ytest,n_repeats=10,random_state=42)
            imp = pi.importances_mean
        fi = pd.DataFrame({'Feature':FEATURES,'Importance':imp}).sort_values('Importance',ascending=False)
        fi['Niveau'] = fi['Importance'].apply(lambda v:
            'Très important' if v>=fi['Importance'].quantile(0.8) else
            ('Important' if v>=fi['Importance'].quantile(0.5) else 'Modéré'))

        c1,c2 = st.columns([3,2])
        with c1:
            section("Top 15 variables", best['Modèle'])
            fi15 = fi.head(15).sort_values('Importance')
            cmap = {'Très important':ORANGE,'Important':VIOLET,'Modéré':MUTED}
            fig = px.bar(fi15,x='Importance',y='Feature',orientation='h',color='Niveau',
                color_discrete_map=cmap,labels={'Importance':'Score','Feature':'Variable'})
            fig.update_traces(texttemplate='%{x:.4f}',textposition='outside',
                textfont_size=10,textfont_color=TEXT)
            st.plotly_chart(light_fig(fig,450), use_container_width=True)
        with c2:
            section("Tableau complet")
            st.dataframe(fi.reset_index(drop=True)[['Feature','Importance','Niveau']]
                .style.format({'Importance':'{:.4f}'})
                .set_properties(**{'font-family':'IBM Plex Mono, monospace','font-size':'12px'}),
                use_container_width=True, height=450)

    with tab3:
        cat_perf = []
        for cat in sorted(df['Catégorie'].unique()):
            mask = df_m.loc[ytest.index,'Catégorie']==cat
            if mask.sum()<10: continue
            yt = ytest[mask]; yp = best['y_pred'][mask.values]
            cat_perf.append({'Catégorie':cat,'MAE':round(mean_absolute_error(yt,yp),1),
                'R²':round(r2_score(yt,yp),4),
                'MAPE (%)':round(np.mean(np.abs((yt.values-yp)/yt.values))*100,1),'N':int(mask.sum())})
        df_cp = pd.DataFrame(cat_perf)
        c1,c2 = st.columns(2)
        with c1:
            section("R² par catégorie", "Précision")
            fig = px.bar(df_cp.sort_values('R²'),x='R²',y='Catégorie',orientation='h',
                color='R²',color_continuous_scale=[[0,SURFACE2],[1,VIOLET]],text='R²')
            fig.update_traces(texttemplate='%{text:.4f}',textposition='outside',textfont_color=TEXT)
            fig.add_vline(x=0.7,line_dash='dot',line_color=AMBER,
                annotation_text='Seuil 0.70',annotation=dict(font_color=AMBER,font_size=11))
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(light_fig(fig,300), use_container_width=True)
        with c2:
            section("MAPE (%) par catégorie", "Erreur relative")
            fig = px.bar(df_cp.sort_values('MAPE (%)'),x='MAPE (%)',y='Catégorie',orientation='h',
                color='MAPE (%)',color_continuous_scale=[[0,SURFACE2],[1,ORANGE]],text='MAPE (%)')
            fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside',textfont_color=TEXT)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(light_fig(fig,300), use_container_width=True)
        st.dataframe(df_cp.style
            .format({'MAE':'{:.1f}','R²':'{:.4f}','MAPE (%)':'{:.1f}%'})
            .background_gradient(cmap='Purples',subset=['R²'])
            .background_gradient(cmap='Oranges',subset=['MAPE (%)'])
            .set_properties(**{'font-family':'IBM Plex Mono, monospace','font-size':'12px'}),
            use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRÉDICTION INTERACTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Prédire les ventes":
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-badge">Simulateur · Prédiction temps réel</div>
      <div class="hero-title">Prédire les <span>Ventes Hebdomadaires</span></div>
      <div class="hero-sub">Modèle : <b>{best['Modèle']}</b> · R² = {best['R²']:.4f} · MAPE = {best['MAPE']:.1f}% · {len(FEATURES)} features</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="insight">💡 Renseignez les caractéristiques d\'un produit et obtenez une <b>prédiction instantanée</b> des ventes hebdomadaires avec projection sur 8 semaines.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1,1])

    with col_form:
        section("Paramètres du produit", "Configuration")
        sub1,sub2 = st.columns(2)
        with sub1:
            categorie  = st.selectbox("🏷️ Catégorie",  sorted(df['Catégorie'].unique()))
            semaine    = st.slider("📅 Semaine", 1, 52, 26)
            promotion  = st.selectbox("🎁 Promotion", ["Non (0)", "Oui (1)"])
            jour_ferie = st.selectbox("📌 Jour férié", ["Non (0)", "Oui (1)"])
        with sub2:
            marque     = st.selectbox("🏢 Marque", sorted(df[df['Catégorie']==categorie]['Marque'].unique()))
            meteo      = st.selectbox("🌤️ Météo",  sorted(df['Météo'].unique()))
            reduction  = st.slider("💸 Réduction (%)", 0, 40, 0, disabled=(promotion=="Non (0)"))
            stock_debut= st.number_input("📦 Stock initial", min_value=1, max_value=500, value=80)
        sub3,sub4 = st.columns(2)
        with sub3:
            ventes_s1 = st.number_input("📈 Ventes semaine préc.", min_value=1, max_value=500, value=50)
        with sub4:
            prix_min = int(df[df['Catégorie']==categorie]['Prix'].min())
            prix_max = int(df[df['Catégorie']==categorie]['Prix'].max())
            prix_def = int(df[df['Catégorie']==categorie]['Prix'].median())
            prix = st.number_input(f"💰 Prix FCFA [{prix_min:,}–{prix_max:,}]",
                min_value=prix_min, max_value=prix_max*2, value=prix_def)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Prédire les ventes")

    with col_result:
        section("Résultat", "Prédiction ML")
        if predict_btn:
            promo_val = 1 if "Oui" in promotion else 0
            ferie_val = 1 if "Oui" in jour_ferie else 0
            mois = ((semaine-1)//4)+1
            try:
                cat_enc = encoders['cat'].transform([categorie])[0]
                mar_enc = encoders['mar'].transform([marque])[0]
                met_enc = encoders['met'].transform([meteo])[0]
            except ValueError:
                st.error("Valeur inconnue."); st.stop()

            X_new = pd.DataFrame([{
                'Prix':prix,'Promotion':promo_val,
                'Réduction_%':reduction if promo_val else 0,
                'Jour_férié':ferie_val,'Ventes_S1':ventes_s1,'Stock_début':stock_debut,
                'Catégorie_enc':cat_enc,'Marque_enc':mar_enc,'Météo_enc':met_enc,
                'Semaine_sin':np.sin(2*np.pi*semaine/52),
                'Semaine_cos':np.cos(2*np.pi*semaine/52),
                'Mois_sin':np.sin(2*np.pi*mois/12),
                'Mois_cos':np.cos(2*np.pi*mois/12),
                'Promo_Réduction':promo_val*(reduction if promo_val else 0),
                'Ferie_Promo':ferie_val*promo_val,
                'Stock_ratio':stock_debut/(ventes_s1+1),
                'Haute_saison':int((10<=semaine<=14) or semaine>=50),
            }])[FEATURES]

            pred_val = int(round(best['model'].predict(X_new)[0]))
            mae_val  = best['MAE']
            lo = max(1, pred_val-int(mae_val)); hi = pred_val+int(mae_val)

            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-label">Ventes prédites · Semaine {semaine}</div>
              <div class="pred-value">{pred_val}</div>
              <div class="pred-unit">unités</div>
              <div class="pred-range">Intervalle de confiance : {lo} – {hi} unités (±MAE)</div>
            </div>""", unsafe_allow_html=True)

            moy_cat = df[df['Catégorie']==categorie]['Ventes'].mean()
            delta = pred_val - moy_cat; pct = delta/moy_cat*100

            st.markdown("<br>", unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            c1.markdown(kpi("Prédiction", f"{pred_val}", "unités/sem.", f"±{int(mae_val)}","orange"), unsafe_allow_html=True)
            c2.markdown(kpi("vs Moy. catégorie", f"{delta:+.0f}", f"Moy. {moy_cat:.0f} u.",
                f"{pct:+.1f}%","violet" if delta>=0 else "orange"), unsafe_allow_html=True)
            c3.markdown(kpi("Précision", f"{100-best['MAPE']:.1f}%", best['Modèle'], f"R²={best['R²']:.4f}","amber"), unsafe_allow_html=True)

            # Projection 8 semaines
            st.markdown("<br>", unsafe_allow_html=True)
            section("Projection sur 8 semaines", "Forecast")
            preds_multi = []; pv = pred_val
            for s in range(semaine, min(semaine+8,53)):
                m_s = ((s-1)//4)+1
                row = {
                    'Prix':prix,'Promotion':promo_val,
                    'Réduction_%':reduction if promo_val else 0,
                    'Jour_férié':0,'Ventes_S1':pv,
                    'Stock_début':max(stock_debut-pv,10),
                    'Catégorie_enc':cat_enc,'Marque_enc':mar_enc,'Météo_enc':met_enc,
                    'Semaine_sin':np.sin(2*np.pi*s/52),
                    'Semaine_cos':np.cos(2*np.pi*s/52),
                    'Mois_sin':np.sin(2*np.pi*m_s/12),
                    'Mois_cos':np.cos(2*np.pi*m_s/12),
                    'Promo_Réduction':promo_val*(reduction if promo_val else 0),
                    'Ferie_Promo':0,
                    'Stock_ratio':max(stock_debut-pv,10)/(pv+1),
                    'Haute_saison':int((10<=s<=14) or s>=50),
                }
                p = int(round(best['model'].predict(pd.DataFrame([row])[FEATURES])[0]))
                preds_multi.append({'Semaine':s,'Prédiction':p,'Low':max(1,p-int(mae_val)),'High':p+int(mae_val)})
                pv = p

            df_proj = pd.DataFrame(preds_multi)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.concat([df_proj['Semaine'],df_proj['Semaine'][::-1]]),
                y=pd.concat([df_proj['High'],df_proj['Low'][::-1]]),
                fill='toself',fillcolor='rgba(194,96,26,0.10)',
                line=dict(color='rgba(0,0,0,0)'),name='Intervalle ±MAE'))
            fig.add_trace(go.Scatter(x=df_proj['Semaine'],y=df_proj['Prédiction'],
                mode='lines+markers',
                line=dict(color=ORANGE,width=2.5),
                marker=dict(size=7,color=ORANGE,line=dict(color=SURFACE,width=1.5)),
                name='Prédictions'))
            fig.update_xaxes(title='Semaine'); fig.update_yaxes(title='Ventes prédites')
            st.plotly_chart(light_fig(fig,260), use_container_width=True)

            st.dataframe(df_proj.style
                .format({'Prédiction':'{:.0f}','Low':'{:.0f}','High':'{:.0f}'})
                .background_gradient(cmap='Oranges',subset=['Prédiction'])
                .set_properties(**{'font-family':'IBM Plex Mono, monospace','font-size':'12px'}),
                use_container_width=True)
        else:
            st.markdown(f"""
            <div style="background:{SURFACE2};border:1px dashed {BORDER};
                        border-radius:12px;padding:60px 20px;text-align:center;margin-top:10px;">
              <div style="font-size:40px;margin-bottom:14px">🎯</div>
              <div style="color:{MUTED};font-size:14px">Configurez les paramètres<br>
              puis cliquez sur <b style="color:{ORANGE}">Prédire les ventes</b></div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RECOMMANDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Recommandations":
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-badge">Insights · Business Intelligence</div>
      <div class="hero-title">Recommandations <span>Stratégiques</span></div>
      <div class="hero-sub">Élasticité-prix · Optimisation promos · Gestion des stocks · Top semaines</div>
    </div>""", unsafe_allow_html=True)

    elas_res = []
    for cat in df_f['Catégorie'].unique():
        d = df_f[df_f['Catégorie']==cat]
        lo = d[d['Prix']<d['Prix'].median()]; hi = d[d['Prix']>=d['Prix'].median()]
        if len(lo)>0 and len(hi)>0:
            pct_v = (hi['Ventes'].mean()-lo['Ventes'].mean())/lo['Ventes'].mean()*100
            pct_p = (hi['Prix'].mean()-lo['Prix'].mean())/lo['Prix'].mean()*100
            e = round(pct_v/pct_p,3) if pct_p!=0 else 0
            elas_res.append({'Catégorie':cat,'Élasticité':e,'Type':'Élastique' if abs(e)>1 else 'Inélastique'})
    df_elas = pd.DataFrame(elas_res)

    promo_sem = (df_f[df_f['Promotion']==1].groupby('Semaine')['Ventes'].mean()
                 .sort_values(ascending=False).head(10).reset_index())

    df_f2 = df_f.copy()
    df_f2['Stock_tranche'] = pd.cut(df_f2['Stock_début'],bins=[0,30,60,100,500],
        labels=['<30','30–60','60–100','>100'])
    stock_imp = df_f2.groupby('Stock_tranche',observed=False)['Ventes'].mean().reset_index()

    c1,c2,c3 = st.columns(3)
    with c1:
        section("Élasticité-prix", "Sensibilité au prix")
        fig = px.bar(df_elas,x='Catégorie',y='Élasticité',color='Type',
            color_discrete_map={'Élastique':ORANGE,'Inélastique':VIOLET},text='Élasticité')
        fig.add_hline(y=-1,line_dash='dot',line_color=MUTED)
        fig.add_hline(y=0,line_color=BORDER,line_width=0.8)
        fig.update_traces(texttemplate='%{text:.3f}',textposition='outside',
            textfont_size=10,textfont_color=TEXT)
        st.plotly_chart(light_fig(fig,310), use_container_width=True)
    with c2:
        section("Top semaines promotionnelles", "ROI promo")
        fig = px.bar(promo_sem,x='Semaine',y='Ventes',text='Ventes',
            color='Ventes',color_continuous_scale=[[0,SURFACE2],[1,AMBER]],
            labels={'Ventes':'Ventes moy.'})
        fig.update_traces(texttemplate='%{text:.0f}',textposition='outside',
            textfont_size=10,textfont_color=TEXT)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(light_fig(fig,310), use_container_width=True)
    with c3:
        section("Impact du stock sur les ventes", "Gestion stock")
        fig = px.bar(stock_imp,x='Stock_tranche',y='Ventes',
            color='Ventes',color_continuous_scale=[[0,SURFACE2],[1,VIOLET]],
            text='Ventes',labels={'Stock_tranche':'Stock','Ventes':'Ventes moy.'})
        fig.update_traces(texttemplate='%{text:.0f}',textposition='outside',
            textfont_size=10,textfont_color=TEXT)
        fig.add_vline(x=1.5,line_dash='dash',line_color=DANGER,
            annotation_text='Seuil critique',
            annotation=dict(font_color=DANGER,font_size=11))
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(light_fig(fig,310), use_container_width=True)

    section("Recommandations stratégiques", "Actions prioritaires")
    r1,r2,r3,r4 = st.columns(4)
    recs = [
        ("🎯","Promotions ciblées","Concentrer les actions en semaines 10–14 (Ramadan) et 50–52. C'est là que le ROI promotionnel est maximal."),
        ("📦","Stock minimum","Maintenir plus de 60 unités en stock. En dessous, les ventes chutent de ~30% par rupture partielle."),
        ("💰","Stratégie prix","Électroménager et Vêtements sont élastiques : un ajustement de prix a un fort impact sur les volumes."),
        ("📊","Top catégories","Boissons et Alimentaire répondent le mieux aux promotions. Prioriser leur approvisionnement en haute saison."),
    ]
    for col,(icon,title,desc) in zip([r1,r2,r3,r4],recs):
        col.markdown(f"""
        <div class="reco-card">
          <div class="reco-icon">{icon}</div>
          <h4>{title}</h4>
          <p>{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Synthèse du meilleur modèle", best['Modèle'])
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi("Modèle", best['Modèle'], "", "Champion","violet"), unsafe_allow_html=True)
    c2.markdown(kpi("R²", f"{best['R²']:.4f}", f"{best['R²']*100:.1f}% variance","Expliqué","orange"), unsafe_allow_html=True)
    c3.markdown(kpi("MAE", f"{best['MAE']:.1f}", "unités d'écart","Erreur abs.","amber"), unsafe_allow_html=True)
    c4.markdown(kpi("RMSE", f"{best['RMSE']:.1f}", "unités","Erreur RMS","amber"), unsafe_allow_html=True)
    c5.markdown(kpi("MAPE", f"{best['MAPE']:.1f}%", "erreur relative","Précision","violet"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Données brutes (200 premières lignes)"):
        st.dataframe(df_f.head(200), use_container_width=True)

    st.markdown(f"""
    <div style="border-top:1px solid {BORDER};
                background:linear-gradient(90deg,#FDF0E6,#F0E6FA);
                padding:16px 0; text-align:center; color:{MUTED}; font-size:12px; margin-top:2rem; border-radius:0 0 10px 10px;">
      Machine Learning · Commerce / Retail · Python · scikit-learn · Streamlit · 2 000 observations · 17 features
    </div>""", unsafe_allow_html=True)