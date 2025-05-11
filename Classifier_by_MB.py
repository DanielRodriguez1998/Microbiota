import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Carga de datos
# -----------------------------
# Ruta del archivo en Kaggle
file_path = '/kaggle/input/microbiota/taxonomic_profiles.tsv'

# Cargar directamente
df = pd.read_csv(file_path, sep='\t')

print(df.shape)
print(df.head())

# -----------------------------
# 2. Preprocesamiento
# -----------------------------
# Transponer si taxones están como filas
if 'taxonomy' in df.columns or 'Taxon' in df.columns:
    df.rename(columns={'taxonomy': 'Taxon', 'Taxon': 'Taxon'}, inplace=True)
    df.set_index('Taxon', inplace=True)
    df = df.transpose()
    df.index.name = 'sample_id'
    df.reset_index(inplace=True)

# Cargar metadata con etiquetas (asegúrate de tenerlo subido a Kaggle)
metadata_path = '/kaggle/input/microbiota/hmp2_metadata_2018-08-20.csv'
metadata = pd.read_csv(metadata_path, low_memory=False)
print(metadata.columns)


# Unir metadata con perfiles bacterianos
df_merged = pd.merge(metadata, df, left_on='External ID', right_on='sample_id')


# ============================
# 1. Definir columnas OTU puras antes del merge
# ============================
otu_columns = [col for col in df.columns if col != 'sample_id' and col != 'Taxon']

# ============================
# 2. Verificar columnas OTU después del merge
# ============================
otu_columns_after_merge = [col for col in otu_columns if col in df_merged.columns]

# ============================
# 3. Calcular varianza global solo para columnas OTU
# ============================
X_global = df_merged[otu_columns_after_merge]
otu_std_global = X_global.std()
otu_columns_nonzero_global = otu_std_global[otu_std_global > 0].index.tolist()

# Eliminar duplicados en la lista
otu_columns_nonzero_global = list(set(otu_columns_nonzero_global))

print(f"Columnas OTU únicas con varianza global > 0: {len(otu_columns_nonzero_global)}")

# ============================
# 4. Filtrar zona anatómica más común
# ============================
zona_mas_comun = df_merged['biopsy_location'].value_counts().idxmax()
df_biopsia = df_merged[df_merged['biopsy_location'] == zona_mas_comun]

# ============================
# 5. Seleccionar X e y sin duplicados
# ============================
X_biopsia = df_biopsia.loc[:, otu_columns_nonzero_global]
y_biopsia = df_biopsia['diagnosis']

# ============================
# 6. Quitar duplicados de columnas en el dataframe (no solo en la lista)
# ============================
X_biopsia = X_biopsia.loc[:, ~X_biopsia.columns.duplicated()]

print(f"Muestras: {X_biopsia.shape[0]}, Columnas OTU finales (sin duplicados): {X_biopsia.shape[1]}")

# ============================
# 7. Escalar
# ============================
if X_biopsia.shape[1] > 0 and X_biopsia.shape[0] > 0:
    scaler = StandardScaler()
    X_scaled_biopsia = scaler.fit_transform(X_biopsia)
    print("Escalado completado.")
else:
    print("No hay columnas útiles para escalar después de filtrar.")

# ============================
# 8. Aplicar PCA
# ============================
if X_biopsia.shape[1] > 0:
    pca = PCA(n_components=50)
    X_pca_biopsia = pca.fit_transform(X_scaled_biopsia)
    print("Shape después de PCA:", X_pca_biopsia.shape)
else:
    print("No se aplicó PCA porque no hay columnas útiles.")

# Seleccionar solo las columnas OTU/genus/species
otu_columns = [col for col in df_merged.columns if 'Otu' in col or col not in metadata.columns]

# Etiquetas de diagnóstico (Crohn, UC, nonIBD)
y = df_merged['diagnosis']
X = df_merged[otu_columns]

print("IDs en metadata:", metadata['External ID'].nunique())
print("IDs en df:", df['sample_id'].nunique())
print("IDs comunes:", len(set(metadata['External ID']).intersection(set(df['sample_id']))))

print("Shape df_merged:", df_merged.shape)
print("Número de NaNs por columna después del merge:")
print(df_merged.isna().sum())

# ============================
# 1. Filtrar solo UC y CD (tanto X como y)
# ============================
mask_uc_cd = df_merged['diagnosis'].isin(['UC', 'CD'])
df_filtered = df_merged[mask_uc_cd]

# ============================
# 2. Codificar labels UC/CD
# ============================
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_filtered['diagnosis'])  # UC=1, CD=0

# ============================
# 3. Codificar biopsy_location (one-hot)
# ============================
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
location_encoded = encoder.fit_transform(df_filtered[['biopsy_location']])
location_encoded_df = pd.DataFrame(location_encoded, 
                                   columns=encoder.get_feature_names_out(['biopsy_location']),
                                   index=df_filtered.index)

# ============================
# 4. Combinar OTU + biopsy_location
# ============================
X = pd.concat([df_filtered[otu_columns_nonzero_global], location_encoded_df], axis=1)
X = X.loc[:, ~X.columns.duplicated()]

# ============================
# 5. Escalar
# ============================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Mostrar cuántas muestras UC y CD quedaron
print("Distribución de diagnóstico (UC vs CD):")
print(df_filtered['diagnosis'].value_counts())

# Mostrar etiquetas codificadas (y verificar correspondencia)
print("\nClases codificadas por LabelEncoder:")
for i, clase in enumerate(label_encoder.classes_):
    print(f"{clase} → {i}")

# Mostrar primeros 5 valores codificados para confirmar
print("\nPrimeras 5 etiquetas codificadas:")
print(y[:5])
# -----------------------------
# Análisis de ceros y NaNs en X (solo OTU)
# -----------------------------

# Seleccionar solo las columnas OTU del dataframe filtrado
X_otu_filtered = df_filtered[otu_columns_nonzero_global]

# Porcentaje de ceros por bacteria (OTU)
zero_fraction = (X_otu_filtered == 0).sum() / X_otu_filtered.shape[0]
print("Media de ceros por bacteria (OTU):", zero_fraction.mean())
print("Número de bacterias con >90% ceros:", sum(zero_fraction > 0.90))

# Porcentaje de NaNs por bacteria (OTU)
nan_fraction = X_otu_filtered.isnull().sum() / X_otu_filtered.shape[0]
print("Media de NaNs por bacteria (OTU):", nan_fraction.mean())
print("Número de bacterias con >90% NaNs:", sum(nan_fraction > 0.90))

# Número de muestras por grupo de diagnóstico
import pandas as pd
y_biopsia_series = pd.Series(label_encoder.inverse_transform(y), name='diagnosis')
print("\nNúmero de muestras por grupo (diagnóstico):")
print(y_biopsia_series.value_counts())


# -----------------------------
# Filtrar OTUs poco informativas (≥90% ceros o NaNs)
# -----------------------------

# Seleccionar solo las columnas OTU del dataframe filtrado
X_otu_filtered = df_filtered[otu_columns_nonzero_global]

# Seleccionar columnas con ≤90% ceros
cols_to_keep = zero_fraction[zero_fraction <= 0.90].index

# Cruce con columnas con ≤90% NaNs
cols_to_keep = nan_fraction[nan_fraction <= 0.90].index.intersection(cols_to_keep)

# Filtrar el dataframe OTU
X_biopsia_filtered = X_otu_filtered[cols_to_keep]

print(f"Shape después de filtrar OTUs poco informativas: {X_biopsia_filtered.shape}")

import numpy as np
import pandas as pd

# -----------------------------
# Limpieza avanzada de X_biopsia_filtered (solo OTUs)
# -----------------------------
️# Seleccionar solo columnas OTU (sin categóricas como biopsy_location)
X_otu_filtered = df_filtered[otu_columns_nonzero_global]

# Eliminar duplicados en columnas (por seguridad)
X_biopsia_clean = X_otu_filtered.loc[:, ~X_otu_filtered.columns.duplicated()]

# Excluir 'sample_id' si existiera
if 'sample_id' in X_biopsia_clean.columns:
    X_biopsia_noid = X_biopsia_clean.drop(columns=['sample_id'])
else:
    X_biopsia_noid = X_biopsia_clean.copy()

# Eliminar duplicados nuevamente (último control)
X_biopsia_noid = X_biopsia_noid.loc[:, ~X_biopsia_noid.columns.duplicated()]

#  Convertir todo a numérico (forzando coerción de errores)
X_biopsia_numeric = X_biopsia_noid.apply(pd.to_numeric, errors='coerce')

# Reemplazar NaNs por cero
X_biopsia_numeric = X_biopsia_numeric.fillna(0)

# Eliminar duplicados finales
X_biopsia_numeric = X_biopsia_numeric.loc[:, ~X_biopsia_numeric.columns.duplicated()]

# Confirmar limpieza
print("Tipos de datos después de convertir:")
print(X_biopsia_numeric.dtypes.value_counts())
print(" Dimensiones finales:", X_biopsia_numeric.shape)



from scipy.stats import kruskal
import pandas as pd

# Convertir etiquetas numéricas a texto (UC, CD)
y_labels = pd.Series(label_encoder.inverse_transform(y), index=X_biopsia_numeric.index)

# Kruskal-Wallis
pvals = []
bacteria_names = []
fc_dict = {}

for bacteria in X_biopsia_numeric.columns:
    groups = [
        X_biopsia_numeric[y_labels == 'CD'][bacteria],
        X_biopsia_numeric[y_labels == 'UC'][bacteria]
    ]

    if all(len(g) >= 2 for g in groups):
        try:
            stat, p = kruskal(*groups)
            pvals.append(p)
            bacteria_names.append(bacteria)
            
            # Calcular diferencia absoluta de medias (fold change robusto)
            mean_cd = groups[0].mean()
            mean_uc = groups[1].mean()
            fc = abs(mean_cd - mean_uc)
            fc_dict[bacteria] = fc
        except:
            pass

# Crear DataFrame resumen
results_df = pd.DataFrame({
    'bacteria': bacteria_names,
    'raw_pval': pvals
})
results_df['fold_change'] = results_df['bacteria'].map(fc_dict)

# Filtrar bacterias con p < 0.05
significant_df = results_df[results_df['raw_pval'] < 0.05]

#Ordenar por fold change (mayor a menor) y seleccionar top 10
top10_df = significant_df.sort_values('fold_change', ascending=False).head(10)
top10_bacteria = top10_df['bacteria'].tolist()

# Imprimir resumen top 10
if len(top10_bacteria) > 0:
    print("\nTop 10 bacterias (p < 0.05, ordenadas por fold change):")
    for _, row in top10_df.iterrows():
        print(f"{row['bacteria']}: p-value={row['raw_pval']:.4f}, FoldChange(abs)={row['fold_change']:.4f}")
    
    # Crear dataframe reducido solo con top 10 bacterias
    X_significant = X_biopsia_numeric[top10_bacteria]
    print("Dimensiones del dataframe reducido (top 10):", X_significant.shape)
else:
    print("\nNo se encontraron bacterias con p < 0.05.")


from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd

# Alinear metadata con muestras actuales (X_significant)
df_metadata_aligned = df_filtered.loc[X_significant.index]

# Codificar 'biopsy_location' en one-hot
encoder = OneHotEncoder(sparse_output=False)  # sin warning
location_encoded = encoder.fit_transform(df_metadata_aligned[['biopsy_location']])
location_encoded_df = pd.DataFrame(
    location_encoded,
    columns=encoder.get_feature_names_out(['biopsy_location']),
    index=X_significant.index
)

#  Escalar solo las bacterias significativas
scaler = StandardScaler()
X_bacteria_scaled = scaler.fit_transform(X_significant)

#Combinar bacterias escaladas + zona codificada
X_final = np.concatenate([X_bacteria_scaled, location_encoded_df.values], axis=1)

#Alinear y codificar etiquetas UC/CD
y_raw = df_metadata_aligned['diagnosis']
y_aligned = pd.Series(label_encoder.transform(y_raw), index=X_significant.index)

# Verificación final
print(" X_final shape:", X_final.shape)
print(" y_aligned shape:", y_aligned.shape)
print(" Distribución de etiquetas:")
print(y_aligned.value_counts())

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Aplicar PCA (2 componentes)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_final)

# Convertir etiquetas a texto
y_labels_named = pd.Series(label_encoder.inverse_transform(y_aligned), index=y_aligned.index)

#  Graficar
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=y_labels_named,
    palette='viridis',
    s=80,
    edgecolor='black'
)
plt.title("PCA - Muestras (Bacterias + Zona)")
plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(title='Diagnóstico')
plt.tight_layout()
plt.savefig("pca_microbiota_zona.png", dpi=300)
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, make_scorer, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_aligned, test_size=0.2, stratify=y_aligned, random_state=42
)
print(f" Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Grids de hiperparámetros
svm_grid = {'svm__C': [1, 10], 'svm__gamma': [0.1], 'svm__kernel': ['rbf']}
rf_grid = {'n_estimators': [100], 'max_depth': [5], 'min_samples_leaf': [2]}
xgb_grid = {'n_estimators': [200], 'max_depth': [3], 'learning_rate': [0.1], 'reg_lambda': [1], 'reg_alpha': [0]}
catboost_grid = {'iterations': [200], 'depth': [6], 'learning_rate': [0.1]}

# Pipelines y modelos base
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, class_weight='balanced', random_state=42))
])
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
catboost = CatBoostClassifier(verbose=0, random_state=42)

️# Scoring
scorer = make_scorer(recall_score, pos_label=1)

# GridSearchCV con cv=3 para SVM/CatBoost y cv=7 para RF/XGB
print("\n Optimizando SVM...")
svm_search = GridSearchCV(svm_pipeline, svm_grid, cv=3, scoring=scorer, n_jobs=-1)
svm_search.fit(X_train, y_train)
svm_best = svm_search.best_estimator_

print("\n Optimizando RandomForest...")
rf_search = GridSearchCV(rf, rf_grid, cv=7, scoring=scorer, n_jobs=-1)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_

print("\n Optimizando XGBoost...")
xgb_search = GridSearchCV(xgb, xgb_grid, cv=7, scoring=scorer, n_jobs=-1)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_

print("\n Optimizando CatBoost...")
catboost_search = GridSearchCV(catboost, catboost_grid, cv=3, scoring=scorer, n_jobs=-1)
catboost_search.fit(X_train, y_train)
catboost_best = catboost_search.best_estimator_

️# Probar diferentes combinaciones de pesos en VotingClassifier
weight_options = [
    [2,2,3,1], [3,2,2,1], [2,3,2,1],
    [2,2,4,1], [2,2,4,0], [1,1,4,1],
    [3,3,2,1], [2,2,5,1], [2,2,3,0]
]

results = []
for weights in weight_options:
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm_best),
            ('rf', rf_best),
            ('xgb', xgb_best),
            ('catboost', catboost_best)
        ],
        voting='soft',
        weights=weights
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    recall = recall_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_proba)
    results.append({
        'weights': weights,
        'recall': recall,
        'auc': auc
    })

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)
best_weights = results_df.sort_values(by='recall', ascending=False).iloc[0]['weights']
print("\nMejores pesos por recall:", best_weights)
️# VotingClassifier final
final_voting = VotingClassifier(
    estimators=[
        ('svm', svm_best),
        ('rf', rf_best),
        ('xgb', xgb_best),
        ('catboost', catboost_best)
    ],
    voting='soft',
    weights=best_weights
)
final_voting.fit(X_train, y_train)
y_pred_voting = final_voting.predict(X_test)
y_proba_voting = final_voting.predict_proba(X_test)[:, 1]

# StackingClassifier
stacking = StackingClassifier(
    estimators=[
        ('svm', svm_best),
        ('rf', rf_best),
        ('xgb', xgb_best),
        ('catboost', catboost_best)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
stacking.fit(X_train, y_train)
y_pred_stack = stacking.predict(X_test)
y_proba_stack = stacking.predict_proba(X_test)[:, 1]

# Comparación de métricas
voting_recall = recall_score(y_test, y_pred_voting)
voting_auc = roc_auc_score(y_test, y_proba_voting)
stacking_recall = recall_score(y_test, y_pred_stack)
stacking_auc = roc_auc_score(y_test, y_proba_stack)

print(f"\n VotingClassifier | Recall: {voting_recall:.3f} | AUC: {voting_auc:.3f}")
print(f" StackingClassifier | Recall: {stacking_recall:.3f} | AUC: {stacking_auc:.3f}")

# Matriz de confusión VotingClassifier
cm = confusion_matrix(y_test, y_pred_voting)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Clase 0', 'Clase 1'])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(cmap='viridis', ax=ax)
plt.title("Matriz de Confusión - VotingClassifier")
plt.tight_layout()
plt.savefig("confusion_matrix_voting_holdout.png", dpi=300)

# Guardar modelos y resultados
joblib.dump(final_voting, 'best_voting_model_holdout.joblib')
joblib.dump(stacking, 'stacking_model_holdout.joblib')
results_df.to_csv("voting_weights_comparison.csv", index=False)

print("\u2705 Modelos y comparación guardados.")
