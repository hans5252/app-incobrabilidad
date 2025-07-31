import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Predicción de Incobrabilidad", layout="wide")
st.title("📊 Predicción de Incobrabilidad por Oficina")

# Cargar archivo directamente desde el repositorio
df = pd.read_excel("Base APP.xlsx", sheet_name="Hoja1", engine="openpyxl")

cols_numericas = ["DEUDA TOTAL", "DEUDA VENCIDA", "MOROSIDAD", "CRÉDITO VIGENTE", "PROMEDIO DE PAGOS"]
for col in cols_numericas:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=cols_numericas + ["DEUDA TOTAL", "DEUDA VENCIDA", "MOROSIDAD"], inplace=True)

df["INCOBRABLE"] = ((df["DEUDA VENCIDA"] / df["DEUDA TOTAL"] > 0.5) & (df["MOROSIDAD"] > 0.5)).astype(int)

X = df[cols_numericas]
y = df["INCOBRABLE"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_scaled, y)

df["SCORE_RIESGO"] = model.predict_proba(X_scaled)[:, 1]
df["PREDICCION_INCOBRABLE"] = model.predict(X_scaled)

df_ultimos = df.sort_values(by=["AÑO", "MES"], ascending=[False, False])
df_unicos = df_ultimos.drop_duplicates(subset=["ID SAP"], keep="first")

top_clientes_por_oficina = (
    df_unicos.sort_values(by=["OFICINA", "SCORE_RIESGO"], ascending=[True, False])
    .groupby("OFICINA")
    .head(10)
    .sort_values(["OFICINA", "SCORE_RIESGO"], ascending=[True, False])
)

st.subheader("🏢 Filtro por Oficina")
oficinas = sorted(top_clientes_por_oficina["OFICINA"].unique())
oficina_seleccionada = st.selectbox("Selecciona una oficina", ["Todas"] + oficinas)

if oficina_seleccionada != "Todas":
    df_filtrado = top_clientes_por_oficina[top_clientes_por_oficina["OFICINA"] == oficina_seleccionada]
else:
    df_filtrado = top_clientes_por_oficina

st.subheader("📋 Top 10 clientes únicos con mayor riesgo por oficina")
st.dataframe(df_filtrado[["OFICINA", "NOMBRE CLIENTE", "SCORE_RIESGO", "PREDICCION_INCOBRABLE"]])

st.subheader("📈 Riesgo promedio por oficina")
riesgo_promedio = df_unicos.groupby("OFICINA")["SCORE_RIESGO"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
riesgo_promedio.plot(kind="bar", ax=ax, color="salmon")
ax.set_ylabel("Riesgo Promedio")
ax.set_title("Riesgo Promedio por Oficina")
st.pyplot(fig)

st.subheader("⬇️ Descargar resultados")
def convertir_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="TopClientes")
    output.seek(0)
    return output

excel_data = convertir_excel(df_filtrado)
st.download_button(
    label="Descargar Excel",
    data=excel_data,
    file_name="top_clientes_riesgo.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
