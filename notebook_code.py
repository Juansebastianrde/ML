# ===== AUTO-GUARD (inserción automática) =====
# Asegura que existan 'cat_features' y 'num_features' antes de su uso,
# incluso si el orden original de celdas del notebook las dejaba después.
cat_features = globals().get('cat_features', [])
num_features = globals().get('num_features', [])

def __ensure_features_from_df():
    global cat_features, num_features
    if 'df' in globals():
        try:
            cat_features_ref = [
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
            if not cat_features:
                cat_features = [c for c in df.columns if c in cat_features_ref]
            if not num_features:
                num_features = [c for c in df.columns if c not in cat_features and c not in ['D.O.A', 'D.O.D']]
        except Exception:
            pass

# Intenta inicializar (no falla si aún no existe df)
__ensure_features_from_df()
# ===== FIN AUTO-GUARD =====

# === Cell 1 ===
# Cargar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import prince


# === Cell 2 ===
# Subir el token
# from google.colab import files
# files.upload()


# === Cell 6 ===
bd = pd.read_csv('HDHI Admission data.csv')
# --- Patch auto: asegurar cat_features/num_features definidos después de cargar df ---
try:
    _cols = list(df.columns)
except NameError:
    _cols = None

if _cols is not None:
    if 'cat_features' not in globals():
        cat_features_ref = [
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
        cat_features = [c for c in cat_features_ref if c in df.columns]
    if 'num_features' not in globals():
        num_features = [c for c in df.columns if c not in cat_features and c not in ['D.O.A', 'D.O.D']]
# --- Fin patch ---
bd.head()


# === Cell 7 ===
bd.info()


# === Cell 8 ===
bd.drop('BNP', axis=1, inplace=True)


# === Cell 9 ===
## Eliminar variables innevesarias

df = bd.drop(['SNO', 'MRD No.'], axis=1)
# --- Patch global definitions for cat_features and num_features ---
cat_features_ref = [
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
cat_features = [c for c in cat_features_ref if c in df.columns]
num_features = [c for c in df.columns if c not in cat_features and c not in ['D.O.A', 'D.O.D']]
# --- End patch ---
df.drop('month year', axis=1, inplace=True)


# === Cell 10 ===
## Transformar las variables de fecha a formate datetime
df['D.O.A'] = pd.to_datetime(df['D.O.A'], format='%m/%d/%Y', errors='coerce')
df['D.O.D'] = pd.to_datetime(df['D.O.D'], format='%m/%d/%Y', errors='coerce')


# === Cell 11 ===
# Tratamiento de aquellas variables que son numéricas pero están como categóricas

cols_to_clean = ['HB', 'TLC', 'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'EF', 'CHEST INFECTION']

for col in cols_to_clean:
    df[col] = (
        df[col]
        .astype(str)                      # aseguramos que todo sea string
        .str.strip()                       # quitamos espacios
        .replace(['EMPTY', 'nan', 'NaN', 'None', ''], np.nan)  # reemplazamos valores no válidos
        .str.replace(r'[<>]', '', regex=True)  # quitamos > y <
        .str.replace(',', '.', regex=False)    # cambiamos coma decimal a punto
    )


# === Cell 12 ===
# Convierte las variables anteriores a numéricas
for col in cols_to_clean:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# === Cell 13 ===
# Transforma las variables categóricas a dummies

df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
df['RURAL'] = df['RURAL'].map({'R': 1, 'U': 0})
df['TYPE OF ADMISSION-EMERGENCY/OPD'] = df['TYPE OF ADMISSION-EMERGENCY/OPD'].map({'E': 1, 'O': 0})
df = pd.get_dummies(df, columns=['OUTCOME'], drop_first=False)


# === Cell 14 ===
# Convierte cualquier columna booleana a int (0 y 1)

bool_cols = df.select_dtypes(include=bool).columns
df[bool_cols] = df[bool_cols].astype(int)


# === Cell 15 ===
#transformar chest infection a numerica

df['CHEST INFECTION'] = df['CHEST INFECTION'].fillna(0).astype(np.int64)


# === Cell 16 ===
df.columns = df.columns.str.strip()
list(df.columns)


# === Cell 17 ===
# Separar categóricas y numéricas
cat_features = binary_cats = [
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
num_features = [col for col in df.columns if col not in cat_features and col not in ['D.O.A', 'D.O.D']]


# === Cell 18 ===
num_features


# === Cell 19 ===
df_numericas = df[num_features]
df_numericas.head()


# === Cell 20 ===
df_categoricas = df[cat_features]
df_categoricas.head()


# === Cell 21 ===
# Separar variables y objetivo
X = df[num_features + cat_features]  # variables
y = df['DURATION OF STAY']  # objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# === Cell 22 ===
# Transformador para numéricas
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Transformador para categóricas
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Combinamos en un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

# Aplicamos el preprocesamiento
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# === Cell 23 ===
# 1. Filtrar solo columnas numéricas continuas de tu lista df_numericas
X_train_num = X_train[num_features].copy()

# 2. Unir con y_train
train_num_with_target = X_train_num.copy()
train_num_with_target["DURATION OF STAY"] = y_train

# 3. Calcular Spearman solo para numéricas continuas
correlaciones = train_num_with_target.corr(method='spearman')["DURATION OF STAY"]

# 4. Ordenar de mayor a menor por valor absoluto
correlaciones_ordenadas = correlaciones.reindex(
    correlaciones.abs().sort_values(ascending=False).index
)

# 5. Imprimir todas
print("=== Correlaciones de Spearman (solo numéricas continuas) ===")
for col, valor in correlaciones_ordenadas.items():
    print(f"{col}: {valor:.4f}")

# 6. Calcular porcentaje y acumulado
df_corr = correlaciones_ordenadas.drop("DURATION OF STAY").to_frame(name="correlation")
df_corr["abs_corr"] = df_corr["correlation"].abs()
df_corr["percentage"] = df_corr["abs_corr"] / df_corr["abs_corr"].sum() * 100
df_corr["cum_percentage"] = df_corr["percentage"].cumsum()


# 7. Filtrar por porcentaje acumulado
umbral_acumulado = 90
numericas_significativas = df_corr[df_corr["cum_percentage"] <= umbral_acumulado].index.tolist()


# === Cell 24 ===
plt.figure(figsize=(10, 6))
plt.bar(df_corr.index, df_corr["percentage"], label="Porcentaje individual")
plt.plot(df_corr.index, df_corr["cum_percentage"], color="red", marker="o", label="Porcentaje acumulado")
plt.axhline(umbral_acumulado, color="green", linestyle="--", label=f"Umbral {umbral_acumulado}%")
plt.xticks(rotation=90)
plt.ylabel("Porcentaje (%)")
plt.title("Importancia por Spearman y acumulado")
plt.legend()
plt.tight_layout()
plt.show()


# === Cell 25 ===
print("\n✅ Variables numéricas seleccionadas (por 90% acumulado):")
print(numericas_significativas)


# === Cell 26 ===
X_train_cat = X_train[cat_features].copy()

significativas = []

for col in X_train_cat:
    grupos = [
        df[df[col] == categoria]['DURATION OF STAY'].dropna()
        for categoria in df[col].unique()
    ]
    f_stat, p_val = stats.f_oneway(*grupos)

    # Verificamos si es estadísticamente significativa
    if p_val < 0.05:
        significativas.append(col)

    print(f"{col}: F={f_stat:.3f}, p-value={p_val:.4f}")

print("\n✅ Variables categóricas significativas (p < 0.05):")
print(significativas)


# === Cell 27 ===
var_seleccionadas= numericas_significativas + significativas

# Filtrar el preprocesado, no el crudo
X_train_filtrado = pd.DataFrame(
    X_train_processed,  # array preprocesado
    columns=num_features + cat_features  # columnas después del preprocesador
)[var_seleccionadas]

X_test_filtrado = pd.DataFrame(
    X_test_processed,
    columns=num_features + cat_features
)[var_seleccionadas]

# Ahora sí, aplicar SelectKBest
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X_train_filtrado, y_train)

scores = selector.scores_
ranking = sorted(zip(var_seleccionadas, scores), key=lambda x: x[1], reverse=True)

df_scores = pd.DataFrame(ranking, columns=["Variable", "Score"])
df_scores["Perc"] = (df_scores["Score"] / df_scores["Score"].sum()) * 100
df_scores["CumPerc"] = df_scores["Perc"].cumsum()

# Mostrar
print(df_scores)


# === Cell 28 ===
fig, ax1 = plt.subplots(figsize=(10,6))

color = "skyblue"
ax1.bar(df_scores["Variable"], df_scores["Score"], color=color)
ax1.set_xlabel("Variables")
ax1.set_ylabel("Score F-test", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_xticklabels(df_scores["Variable"], rotation=45, ha="right")

ax2 = ax1.twinx()
color = "crimson"
ax2.plot(df_scores["Variable"], df_scores["CumPerc"], color=color, marker="o")
ax2.set_ylabel("Cumulative %", color=color)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim(0, 110)

# 5. Línea de referencia al o 90%
ax2.axhline(y=90, color="gray", linestyle="--", linewidth=1)
ax2.text(len(df_scores)-1, 82, "90% threshold", color="gray")

plt.title("SelectKBest - Importance & Cumulative %")
plt.tight_layout()
plt.show()


# === Cell 29 ===
# Filtrar hasta el 90%
vars_90 = df_scores[df_scores["CumPerc"] <= 90]['Variable'].tolist()

print(f"Variables que acumulan el 90%: {vars_90}")
print(f"Cantidad: {len(vars_90)}")


# === Cell 30 ===
# Modelo base para RFE
modelo_rfe = LinearRegression()

# RFE para seleccionar, por ejemplo, las 10 mejores
selector_rfe = RFE(modelo_rfe, n_features_to_select=10)
selector_rfe.fit(X_train_filtrado, y_train)

# Variables seleccionadas
selected_features_rfe = X_train_filtrado.columns[selector_rfe.support_]
print("\n✅ Variables seleccionadas por RFE:")
print(selected_features_rfe.tolist())


# === Cell 31 ===
# Entrenar modelo
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_filtrado, y_train)

# Importancia de características
importances = rf_model.feature_importances_
importances_df = pd.DataFrame({
    'Variable': X_train_filtrado.columns,
    'Importancia': importances
}).sort_values(by='Importancia', ascending=False)

# Calcular porcentaje acumulado
importances_df['Importancia_Acum'] = importances_df['Importancia'].cumsum()

# Seleccionar variables hasta el 90%
selected_vars = importances_df[importances_df['Importancia_Acum'] <= 0.90]['Variable'].tolist()

print("\n🌲 Ranking de importancia con RandomForest:")
print(importances_df)
print("\n✅ Variables seleccionadas hasta el 90% de importancia acumulada:")
print(selected_vars)

# Filtrar dataset
X_train_sel = X_train_filtrado[selected_vars]
X_test_sel = X_test_filtrado[selected_vars]


# === Cell 32 ===
# Gráfica de importancias y acumulado
plt.figure(figsize=(10, 6))

# Barras de importancia
plt.bar(importances_df['Variable'], importances_df['Importancia'], alpha=0.7, label='Importancia individual')

# Línea de importancia acumulada
plt.plot(importances_df['Variable'], importances_df['Importancia_Acum'], marker='o', color='red', label='Importancia acumulada')

# Línea de referencia del 90%
plt.axhline(0.90, color='green', linestyle='--', label='90% acumulado')

# Formato del gráfico
plt.xticks(rotation=90)
plt.ylabel('Importancia')
plt.title('Ranking de importancia de variables - RandomForest')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# === Cell 33 ===
# Convertimos a conjuntos y sacamos la intersección
comunes = list(set(vars_90) & set(selected_features_rfe) )

print("Variables en común:", comunes)


# === Cell 34 ===
# Filtrar DataFrame solo con esas columnas
df_filtrado = df[comunes]
print("\nDataFrame filtrado:")
df_filtrado.head()


# === Cell 35 ===
# 1. Obtener índices de columnas numéricas en el dataset crudo
num_indices = [i for i, col in enumerate(X_train.columns) if col in num_features]

# 2. Filtrar las columnas procesadas usando esos índices
X_train_numericas = pd.DataFrame(
    X_train_processed[:, num_indices],
    columns=num_features
)

X_test_numericas = pd.DataFrame(
    X_test_processed[:, num_indices],
    columns=num_features
)

# PCA (elige 70% de var. explicada automáticamente)
pca = PCA(n_components=0.70, random_state=42)
Xn_train_pca = pca.fit_transform(X_train_numericas)
Xn_test_pca  = pca.transform(X_test_numericas)

pca_names = [f'PCA{i+1}' for i in range(Xn_train_pca.shape[1])]
Xn_train_pca = pd.DataFrame(Xn_train_pca, columns=pca_names, index=X_train.index)
Xn_test_pca  = pd.DataFrame(Xn_test_pca,  columns=pca_names, index=X_test.index)

print(f'PCA: {len(pca_names)} componentes, var. explicada acumulada = {pca.explained_variance_ratio_.sum():.3f}')


# === Cell 36 ===
# Varianza explicada por cada componente
var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

plt.figure(figsize=(8,5))

# Barras de varianza explicada individual
plt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.6, label='Varianza explicada por componente')

# Línea de varianza acumulada
plt.step(range(1, len(cum_var_exp)+1), cum_var_exp, where='mid', color='red', label='Varianza acumulada')

# Línea horizontal del 70% (opcional, ya que usaste 0.70)
plt.axhline(y=0.70, color='green', linestyle='--', label='70%')

plt.xlabel('Componentes principales')
plt.ylabel('Proporción de varianza explicada')
plt.title('PCA - Varianza explicada y acumulada')
plt.xticks(range(1, len(var_exp)+1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# === Cell 37 ===
# 1. Obtener índices de las columnas categóricas en el X_train original
cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]

# 2. Filtrar las columnas categóricas procesadas
X_train_categoricas = pd.DataFrame(
    X_train_processed[:, cat_indices],
    columns=cat_features
)

X_test_categoricas = pd.DataFrame(
    X_test_processed[:, cat_indices],
    columns=cat_features
)

# 3. MCA con prince
import prince

mca = prince.MCA(
    n_components=5,
    n_iter=5,
    random_state=42
)
mca.fit(X_train_categoricas)

Xc_train_mca = mca.transform(X_train_categoricas)
Xc_test_mca  = mca.transform(X_test_categoricas)
Xc_train_mca.index = X_train.index
Xc_test_mca.index  = X_test.index

# 4. Renombrar componentes
mca_names = [f'MCA{i+1}' for i in range(Xc_train_mca.shape[1])]
Xc_train_mca.columns = mca_names
Xc_test_mca.columns  = mca_names

# 5. Porcentaje de varianza (inercia)
ev = mca.eigenvalues_summary
print('MCA inercia por eje:', ev['% of variance'].values)


# === Cell 38 ===
# Convertir a numérico, quitando símbolos si es necesario
ev['% of variance'] = ev['% of variance'].replace('%', '', regex=True)  # elimina el símbolo %
ev['% of variance'] = ev['% of variance'].str.replace(',', '.', regex=False)  # cambia coma por punto
ev['% of variance'] = pd.to_numeric(ev['% of variance'], errors='coerce')

# Ahora sí, en proporción
var_exp_mca = ev['% of variance'].values / 100
cum_var_exp_mca = np.cumsum(var_exp_mca)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(var_exp_mca) + 1), cum_var_exp_mca, marker='o')
plt.xlabel('Dimensiones')
plt.ylabel('Varianza Acumulada')
plt.grid(True)
plt.show()


# === Cell 39 ===
# Asegurar que los valores sean numéricos
ev['% of variance'] = pd.to_numeric(ev['% of variance'], errors='coerce')

# Varianza explicada en proporción y acumulada
var_exp_mca = ev['% of variance'].values / 100
cum_var_exp_mca = np.cumsum(var_exp_mca)
componentes = np.arange(1, len(var_exp_mca) + 1)

# Gráfica
plt.figure(figsize=(8, 5))

# Barras: varianza explicada individual
plt.bar(componentes, var_exp_mca, alpha=0.7, label='Varianza explicada')

# Línea: varianza acumulada
plt.plot(componentes, cum_var_exp_mca, marker='o', color='red', label='Varianza acumulada')

# Formato
plt.xticks(componentes)
plt.xlabel('Componentes MCA')
plt.ylabel('Proporción de varianza')
plt.title('Scree plot - MCA')
plt.legend()
plt.grid(alpha=0.3)

plt.show()


# === Cell 40 ===
X_train_reduced = pd.concat([Xn_train_pca, Xc_train_mca], axis=1)
X_test_reduced  = pd.concat([Xn_test_pca,  Xc_test_mca],  axis=1)

print('Shape train reducido:', X_train_reduced.shape)
print('Shape test  reducido:', X_test_reduced.shape)


# === Cell 41 ===
X_train_reduced = pd.concat(
    [Xn_train_pca.reset_index(drop=True),
     Xc_train_mca.reset_index(drop=True)],
    axis=1
)
X_train_reduced.head(10)