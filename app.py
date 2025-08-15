# streamlit app generated from your notebook
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from contextlib import redirect_stdout
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import prince

st.set_page_config(page_title="HDHI Admission — Notebook App", layout="wide")

# ---------- Helpers ----------
@st.cache_data
def load_csv_robust():
    """Try to load the dataset from common filenames in repo root."""
    candidates = [
        "HDHI Admission data.csv",
        "HDHI_Admission_data.csv",
        "HDHI Admission data.CSV",
        "HDHI_Admission_data.CSV",
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    # If none found, list directory for hints
    files = "\n".join(os.listdir("."))
    raise FileNotFoundError(
        "No se encontró el archivo de datos. Asegúrate de subirlo al repositorio "
        "con el nombre 'HDHI Admission data.csv' (o variantes). Archivos en el directorio:\n"
        + files
    )

def show_matplotlib(figure=None):
    """Render current or given Matplotlib figure in Streamlit and close it."""
    if figure is None:
        fig = plt.gcf()
    else:
        fig = figure
    st.pyplot(fig)
    plt.close(fig)

def capture_text(fn, *args, **kwargs):
    """Capture stdout from functions like DataFrame.info()."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()

# ---------- App UI ----------
st.title("Base de datos")

# === 1. Cargar base de datos ===
st.header("1. Cargar base de datos")
bd = load_csv_robust()
st.write("Vista previa:")
st.dataframe(bd.head())

st.subheader("Información del DataFrame")
info_text = capture_text(bd.info)
st.code(info_text, language="text")

# Intentar eliminar columna BNP si existe (como en tu notebook)
if "BNP" in bd.columns:
    bd.drop("BNP", axis=1, inplace=True)

# === Descripción de la base de datos (markdown del notebook) ===
st.header("Descripción de la base de datos")
st.markdown(
"""
Este conjunto de datos contiene registros de admisiones hospitalarias con variables demográficas, 
detalles de admisión, antecedentes médicos, parámetros de laboratorio, condiciones clínicas y 
resultado hospitalario. La variable objetivo usada en este proyecto es **DURATION OF STAY**.
"""
)

# === 2. Tratamiento de la base de datos ===
st.header("2. Tratamiento de la base de datos")

# 2.1 Eliminar variables innecesarias (como en el notebook)
df = bd.copy()
for col in ["SNO", "MRD No."]:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)
if "month year" in df.columns:
    df.drop("month year", axis=1, inplace=True)

# 2.2 Fechas a datetime
for date_col in ["D.O.A", "D.O.D"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%Y", errors="coerce")

# 2.3 Variables numéricas que vienen como texto
cols_to_clean = ['HB', 'TLC', 'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'EF', 'CHEST INFECTION']
for col in cols_to_clean:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace(['EMPTY', 'nan', 'NaN', 'None', ''], np.nan)
            .str.replace(r'[<>]', '', regex=True)
            .str.replace(',', '.', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 2.4 Mapear categóricas a numéricas/dummies (según notebook)
if "GENDER" in df.columns:
    df["GENDER"] = df["GENDER"].map({'M': 1, 'F': 0})
if "RURAL" in df.columns:
    df["RURAL"] = df["RURAL"].map({'R': 1, 'U': 0})
if "TYPE OF ADMISSION-EMERGENCY/OPD" in df.columns:
    df["TYPE OF ADMISSION-EMERGENCY/OPD"] = df["TYPE OF ADMISSION-EMERGENCY/OPD"].map({'E': 1, 'O': 0})
if "OUTCOME" in df.columns:
    df = pd.get_dummies(df, columns=["OUTCOME"], drop_first=False)

# 2.5 Cols booleanas a int
bool_cols = df.select_dtypes(include=bool).columns
if len(bool_cols) > 0:
    df[bool_cols] = df[bool_cols].astype(int)

# 2.6 CHEST INFECTION a entero
if "CHEST INFECTION" in df.columns:
    df["CHEST INFECTION"] = df["CHEST INFECTION"].fillna(0).astype("Int64").astype(np.int64)

# 2.7 Strip a nombres
df.columns = df.columns.str.strip()

# === 2.1 Separación en variables categóricas y numéricas (según notebook) ===
st.subheader("2.1 Separación en variables")
cat_features = [
    'GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD',
    'OUTCOME_DAMA', 'OUTCOME_DISCHARGE', 'OUTCOME_EXPIRY',
    'SMOKING', 'ALCOHOL', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD',
    'RAISED CARDIAC ENZYMES', 'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA',
    'ACS', 'STEMI', 'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF',
    'VALVULAR', 'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT',
    'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',
    'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',
    'PULMONARY EMBOLISM'
]
# Dejar solo las que existan realmente en df
cat_features = [c for c in cat_features if c in df.columns]
num_features = [col for col in df.columns if col not in cat_features and col not in ["D.O.A","D.O.D"]]

st.write("**Numéricas (preview):**")
st.dataframe(df[num_features].head())
st.write("**Categóricas (preview):**")
st.dataframe(df[cat_features].head())

# === 3. Train/Test split ===
st.header("3. División de datos y preprocesamiento")
X = df[num_features + cat_features].copy()
y = df["DURATION OF STAY"] if "DURATION OF STAY" in df.columns else None
if y is None:
    st.error("No se encontró la columna objetivo 'DURATION OF STAY'.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocesador
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, num_features), ("cat", categorical_transformer, cat_features)]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# === Spearman para numéricas continuas ===
st.header("4. Selección por filtrado: Spearman (numéricas)")
X_train_num = X_train[num_features].copy()
train_num_with_target = X_train_num.copy()
train_num_with_target["DURATION OF STAY"] = y_train
correlaciones = train_num_with_target.corr(method="spearman")["DURATION OF STAY"]
correlaciones_ordenadas = correlaciones.reindex(correlaciones.abs().sort_values(ascending=False).index)

st.write("**Correlaciones (Spearman) ordenadas:**")
st.code("\n".join([f"{col}: {val:.4f}" for col, val in correlaciones_ordenadas.items()]), language="text")

df_corr = correlaciones_ordenadas.drop("DURATION OF STAY").to_frame(name="correlation")
df_corr["abs_corr"] = df_corr["correlation"].abs()
df_corr["percentage"] = df_corr["abs_corr"] / df_corr["abs_corr"].sum() * 100
df_corr["cum_percentage"] = df_corr["percentage"].cumsum()
umbral_acumulado = 90
numericas_significativas = df_corr[df_corr["cum_percentage"] <= umbral_acumulado].index.tolist()

# Gráfico Spearman
plt.figure(figsize=(10,6))
plt.bar(df_corr.index, df_corr["percentage"], label="Porcentaje individual")
plt.plot(df_corr.index, df_corr["cum_percentage"], marker="o", label="Porcentaje acumulado")
plt.axhline(umbral_acumulado, linestyle="--", label=f"Umbral {umbral_acumulado}%")
plt.xticks(rotation=90)
plt.ylabel("Porcentaje (%)")
plt.title("Importancia por Spearman y acumulado")
plt.legend()
plt.tight_layout()
show_matplotlib()

st.success(f"Variables numéricas seleccionadas por 90% acumulado: {numericas_significativas}")

# === ANOVA (categóricas) ===
st.header("5. Selección por filtrado: ANOVA (categóricas)")
X_train_cat = X_train[cat_features].copy()
significativas = []
for col in X_train_cat.columns:
    grupos = [df[df[col] == categoria]["DURATION OF STAY"].dropna() for categoria in df[col].dropna().unique()]
    # Evitar errores si hay categorías con 0 o 1 elemento
    grupos = [g for g in grupos if len(g) > 1]
    if len(grupos) >= 2:
        f_stat, p_val = stats.f_oneway(*grupos)
        if p_val < 0.05:
            significativas.append(col)

st.write("**Variables categóricas significativas (p < 0.05):**")
st.write(significativas)

# === SelectKBest sobre variables filtradas ===
st.header("6. SelectKBest (sobre variables filtradas)")
var_seleccionadas = numericas_significativas + significativas

X_train_filtrado = pd.DataFrame(X_train_processed, columns=num_features + cat_features)[var_seleccionadas]
X_test_filtrado  = pd.DataFrame(X_test_processed,  columns=num_features + cat_features)[var_seleccionadas]

selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X_train_filtrado, y_train)
scores = selector.scores_
ranking = sorted(zip(var_seleccionadas, scores), key=lambda x: x[1], reverse=True)
df_scores = pd.DataFrame(ranking, columns=["Variable", "Score"])
df_scores["Perc"] = (df_scores["Score"] / df_scores["Score"].sum()) * 100
df_scores["CumPerc"] = df_scores["Perc"].cumsum()

st.dataframe(df_scores)

# Gráfico SelectKBest
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.bar(df_scores["Variable"], df_scores["Score"])
ax1.set_xlabel("Variables")
ax1.set_ylabel("Score F-test")
ax1.set_xticklabels(df_scores["Variable"], rotation=45, ha="right")

ax2 = ax1.twinx()
ax2.plot(df_scores["Variable"], df_scores["CumPerc"], marker="o")
ax2.set_ylabel("Cumulative %")
ax2.set_ylim(0, 110)
ax2.axhline(y=90, linestyle="--", linewidth=1)
plt.title("SelectKBest - Importance & Cumulative %")
plt.tight_layout()
show_matplotlib(fig)

vars_90 = df_scores[df_scores["CumPerc"] <= 90]["Variable"].tolist()
st.info(f"Variables que acumulan el 90%: {vars_90} (Cantidad: {len(vars_90)})")

# === RFE ===
st.header("7. RFE (Linear Regression)")
modelo_rfe = LinearRegression()
selector_rfe = RFE(modelo_rfe, n_features_to_select=min(10, X_train_filtrado.shape[1]))
selector_rfe.fit(X_train_filtrado, y_train)
selected_features_rfe = X_train_filtrado.columns[selector_rfe.support_].tolist()
st.success(f"Variables seleccionadas por RFE: {selected_features_rfe}")

# === Random Forest Importances ===
st.header("8. Random Forest — Importancia de variables")
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_filtrado, y_train)
importances = rf_model.feature_importances_
importances_df = pd.DataFrame({"Variable": X_train_filtrado.columns, "Importancia": importances}).sort_values(
    by="Importancia", ascending=False
)
importances_df["Importancia_Acum"] = importances_df["Importancia"].cumsum()
selected_vars_rf = importances_df[importances_df["Importancia_Acum"] <= 0.90]["Variable"].tolist()

st.dataframe(importances_df)

plt.figure(figsize=(10,6))
plt.bar(importances_df["Variable"], importances_df["Importancia"], alpha=0.7, label="Importancia individual")
plt.plot(importances_df["Variable"], importances_df["Importancia_Acum"], marker="o", label="Importancia acumulada")
plt.axhline(0.90, linestyle="--", label="90% acumulado")
plt.xticks(rotation=90)
plt.ylabel("Importancia")
plt.title("Ranking de importancia de variables - RandomForest")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
show_matplotlib()

# Intersección y DataFrame filtrado final de ejemplo
comunes = list(set(vars_90) & set(selected_features_rfe))
st.write("**Intersección (SelectKBest 90% ∩ RFE):**", comunes)
if len(comunes) > 0:
    df_filtrado = df[comunes]
    st.write("Preview df filtrado (solo comunes):")
    st.dataframe(df_filtrado.head())

# === PCA y MCA ===
st.header("9. PCA y MCA")
# a) PCA en numéricas
num_indices = [i for i, col in enumerate(X_train.columns) if col in num_features]
X_train_numericas = pd.DataFrame(X_train_processed[:, num_indices], columns=num_features)
X_test_numericas  = pd.DataFrame(X_test_processed[:,  num_indices], columns=num_features)

pca = PCA(n_components=0.70, random_state=42)
Xn_train_pca = pca.fit_transform(X_train_numericas)
Xn_test_pca  = pca.transform(X_test_numericas)
pca_names = [f"PCA{i+1}" for i in range(Xn_train_pca.shape[1])]
Xn_train_pca = pd.DataFrame(Xn_train_pca, columns=pca_names, index=X_train.index)
Xn_test_pca  = pd.DataFrame(Xn_test_pca,  columns=pca_names, index=X_test.index)

st.info(f"PCA: {len(pca_names)} componentes, var. explicada acumulada = {pca.explained_variance_ratio_.sum():.3f}")

var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)
plt.figure(figsize=(8,5))
plt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.6, label="Varianza explicada por componente")
plt.step(range(1, len(cum_var_exp)+1), cum_var_exp, where="mid", label="Varianza acumulada")
plt.axhline(y=0.70, linestyle="--", label="70%")
plt.xlabel("Componentes principales")
plt.ylabel("Proporción de varianza explicada")
plt.title("PCA - Varianza explicada y acumulada")
plt.xticks(range(1, len(var_exp)+1))
plt.legend()
plt.grid(True)
plt.tight_layout()
show_matplotlib()

# b) MCA en categóricas
cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]
X_train_categoricas = pd.DataFrame(X_train_processed[:, cat_indices], columns=cat_features)
X_test_categoricas  = pd.DataFrame(X_test_processed[:,  cat_indices], columns=cat_features)

mca = prince.MCA(n_components=5, n_iter=5, random_state=42)
mca.fit(X_train_categoricas)
Xc_train_mca = mca.transform(X_train_categoricas)
Xc_test_mca  = mca.transform(X_test_categoricas)
Xc_train_mca.index = X_train.index
Xc_test_mca.index  = X_test.index

mca_names = [f"MCA{i+1}" for i in range(Xc_train_mca.shape[1])]
Xc_train_mca.columns = mca_names
Xc_test_mca.columns  = mca_names

ev = mca.eigenvalues_summary
# Manejar tanto string con '%' como numérico
perc = ev['% of variance']
if perc.dtype == "object":
    perc = perc.str.replace('%','', regex=False).astype(float)
perc_values = perc.values
cum_perc = np.cumsum(perc_values)

st.write("**MCA inercia por eje (%):**", perc_values)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(perc_values)+1), perc_values, marker="o", label="% por eje")
plt.plot(range(1, len(cum_perc)+1), cum_perc, marker="o", label="Acumulada")
plt.xlabel("Componentes")
plt.ylabel("% de varianza")
plt.title("MCA — % de varianza por eje y acumulada")
plt.legend()
plt.tight_layout()
show_matplotlib()

# c) Concatenar reducciones
X_train_reduced = pd.concat([Xn_train_pca.reset_index(drop=True), Xc_train_mca.reset_index(drop=True)], axis=1)
X_test_reduced  = pd.concat([Xn_test_pca.reset_index(drop=True),  Xc_test_mca.reset_index(drop=True)],  axis=1)
st.write("Shape train reducido:", X_train_reduced.shape)
st.write("Shape test reducido:",  X_test_reduced.shape)

st.success("Listo. El app replica el flujo de tu notebook: texto en orden, transformaciones, selección de variables y gráficas.")
