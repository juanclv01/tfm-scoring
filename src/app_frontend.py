"""
Frontend Streamlit del pipeline de scoring crediticio explicable.

Consume app_backend.py (FastAPI) via HTTP -- no importa nada del pipeline
directamente, solo llama a la API. Dos modos:
  1. "Consultar solicitante": formulario manual con los 19 campos de
     German Credit (POST /score).
  2. "Demo: instancia aleatoria": boton que pide una instancia al azar
     del conjunto de test ya usado en la experimentacion (GET
     /demo/instancia_aleatoria) -- nunca datos inventados.

Uso:
    streamlit run app_frontend.py
"""
import os

import requests
import streamlit as st
import plotly.graph_objects as go

from data_loader import (
    FEATURE_NAME_LABELS_ES, FEATURE_VALUE_LABELS,
    INSTALLMENT_RATE_LABELS, NUM_DEPENDENTS_LABELS,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Puntuación crediticia explicable", page_icon="💳", layout="centered")


@st.cache_data(ttl=60)
def _max_intentos_guardarraiz() -> int:
    """Se obtiene del backend en vez de duplicar el numero aqui -- evita
    que este valor quede desactualizado si MAX_INTENTOS cambia en
    app_backend.py y aqui no. Cache de 60s para no consultar /health en
    cada repintado de Streamlit; 3 es el valor de fallback solo por si
    el backend no responde (no se usa en condiciones normales)."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()["max_intentos_guardarraiz"]
    except requests.exceptions.RequestException:
        return 3

# ---------------------------------------------------------------------
# Estilo -- aproximacion al tema oscuro de la imagen de referencia
# ---------------------------------------------------------------------
st.markdown("""
<style>
.card {
    background-color: #1e2128;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #2c303a;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 0.5rem;
}
.badge-aprobado { background-color: #16332b; color: #4ade80; }
.badge-rechazado { background-color: #3a1a1a; color: #f87171; }
.badge-riesgo { background-color: #2a2d35; color: #d1d5db; }
.score-num { font-size: 2.6rem; font-weight: 700; }
.narrativa-box {
    background-color: #14161b;
    border-left: 3px solid #4ade80;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    line-height: 1.6;
}
.footer-meta { color: #6b7280; font-size: 0.75rem; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# Valores por defecto: el cliente_ejemplo real de graph.py (no inventados)
VALORES_POR_DEFECTO = {
    "checking_status": "A11", "duration": 24, "credit_history": "A32",
    "purpose": "A43", "credit_amount": 3500, "savings_status": "A61",
    "employment": "A73", "installment_rate": 3, "personal_status": "A93",
    "other_parties": "A101", "residence_since": 2,
    "property_magnitude": "A121", "age": 34,
    "other_payment_plans": "A143", "housing": "A152",
    "existing_credits": 1, "job": "A173", "num_dependents": 1,
    "own_telephone": "A192", "foreign_worker": "A201",
}


def _campo_desplegable(nombre_campo: str, etiquetas: dict, valor_por_defecto):
    """Desplegable que muestra la version decodificada en espanol
    (FEATURE_VALUE_LABELS / INSTALLMENT_RATE_LABELS / NUM_DEPENDENTS_LABELS
    de data_loader.py) pero devuelve el codigo/valor crudo real que espera
    el pipeline -- nunca se escribe a mano, se selecciona de las opciones
    posibles."""
    opciones = list(etiquetas.keys())
    return st.selectbox(
        FEATURE_NAME_LABELS_ES.get(nombre_campo, nombre_campo),
        options=opciones,
        index=opciones.index(valor_por_defecto),
        format_func=lambda codigo: etiquetas[codigo],
    )


def _renderizar_resultado(resultado: dict):
    badge_decision = "badge-aprobado" if resultado["aprobado"] else "badge-rechazado"
    texto_decision = "Aprobado" if resultado["aprobado"] else "Rechazado"

    col_izq, col_der = st.columns([3, 1])
    with col_izq:
        st.markdown(f"""
        <div class="card">
            <div style="font-size:0.75rem;color:#9ca3af;letter-spacing:0.05em;">SOLICITANTE</div>
            <div style="font-size:1.1rem;font-weight:600;margin-bottom:0.6rem;">
                ID {resultado['id_solicitante']}
            </div>
            <span class="badge {badge_decision}">{texto_decision}</span>
            <span class="badge badge-riesgo">Riesgo {resultado['nivel_riesgo']}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_der:
        color_score = "#4ade80" if resultado["aprobado"] else "#f87171"
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div style="font-size:0.75rem;color:#9ca3af;">PUNTUACIÓN</div>
            <div class="score-num" style="color:{color_score};">{resultado['score']}</div>
            <div style="font-size:0.75rem;color:#9ca3af;">de 1000</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Factores SHAP ---
    factores = resultado["factores"]
    nombres = [f["feature_name"] for f in factores][::-1]
    valores = [f["shap_value"] for f in factores][::-1]
    colores = ["#4ade80" if v > 0 else "#f87171" for v in valores]
    etiquetas_valor = [f"{'+' if v > 0 else ''}{v:.1f} pts".replace(".", ",") for v in valores]

    # CORRECTED: las etiquetas "-147,3 pts" / "+80,3 pts" se cortaban porque
    # (a) con textposition="outside" el texto de las barras mas largas cae
    # fuera del rango autoajustado del eje X, y (b) Plotly recorta por
    # defecto (cliponaxis=True) todo lo que queda fuera del area de trazado.
    # Se fija un rango explicito con holgura proporcional a la barra mas
    # larga (para que el texto quepa dentro del area) y se desactiva el
    # recorte como red de seguridad.
    max_abs = max((abs(v) for v in valores), default=1) or 1
    holgura = max_abs * 0.45
    rango_x = [min(0, min(valores, default=0)) - holgura,
               max(0, max(valores, default=0)) + holgura]

    fig = go.Figure(go.Bar(
        x=valores, y=nombres, orientation="h",
        marker_color=colores,
        text=etiquetas_valor, textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title="Factores determinantes — importancia SHAP",
        paper_bgcolor="#1e2128", plot_bgcolor="#1e2128",
        font_color="#e5e7eb",
        margin=dict(l=10, r=20, t=40, b=10),
        height=280,
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#3f4451",
                   visible=False, range=rango_x),
        yaxis=dict(automargin=True),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # --- Narrativa ---
    # CORRECTED: cuando el GRADER no aprueba la narrativa tras MAX_INTENTOS,
    # el grafo enruta a node_human_review(), que SOBRESCRIBE "narrative" con
    # una plantilla fija sin LLM (aviso del art. 22 RGPD dirigido al
    # solicitante). Por tanto, en ese caso resultado["narrativa"] nunca es el
    # borrador rechazado y SI debe mostrarse: es el mensaje de escalado. Lo
    # que era incorrecto en la version original es que el pie de la tarjeta
    # afirmaba "Generado por NARRATOR y GRADER via DSPy", atribuyendo a los
    # LLM un texto que es una plantilla fija; ademas la insignia
    # Aprobada/Rechazada (veredicto del GRADER) contradecia visualmente la
    # decision crediticia. Ambas cosas se eliminan en la rama de revision.
    if resultado["requiere_revision_humana"]:
        st.markdown(f"""
        <div class="card" style="border-color:#f59e0b;">
            <div style="font-size:0.75rem;color:#f59e0b;letter-spacing:0.05em;margin-bottom:0.6rem;">
                ⚠️ REVISIÓN HUMANA REQUERIDA — ART. 22 RGPD
            </div>
            <div class="narrativa-box" style="border-left-color:#f59e0b;">{resultado['narrativa']}</div>
            <div class="footer-meta">
                Mensaje fijo, sin LLM · Intentos del NARRATOR: {resultado['intentos_narrator']} ·
                Hash de trazabilidad: {resultado['id_solicitante']} ·
                {resultado['tiempo_generacion_seg']}s
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="card">
        <div style="font-size:0.75rem;color:#9ca3af;letter-spacing:0.05em;margin-bottom:0.6rem;">
            EXPLICACIÓN GENERADA POR LLM — ART. 22 RGPD
        </div>
        <div class="narrativa-box" style="border-left-color:#4ade80;">{resultado['narrativa']}</div>
        <div class="footer-meta">
            Generado por {resultado['modelo_narrator']} (NARRATOR) y {resultado['modelo_grader']} (GRADER) vía DSPy + LangGraph ·
            Configuración NARRATOR: {resultado['configuracion_narrator']} ·
            Hash de trazabilidad: {resultado['id_solicitante']} ·
            {resultado['tiempo_generacion_seg']}s
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# UI principal
# ---------------------------------------------------------------------
st.title("💳 Puntuación crediticia explicable")
st.caption("Nodo 1 (XGBoost) → Nodo 2 (SHAP) → Nodo 3 (NARRATOR, H=1/B=0) → "
           "Nodo 4 (GRADER + reintento, hasta 5 intentos)")

modo = st.radio("Modo", ["Consultar solicitante", "Demo: instancia aleatoria del conjunto de test"],
                 horizontal=True, label_visibility="collapsed")

if modo == "Demo: instancia aleatoria del conjunto de test":
    st.info("Selecciona una instancia al azar de las 20 del conjunto de test ya usado "
            "en la experimentación (nunca datos inventados) y genera su explicación.")
    if st.button("🎲 Obtener instancia aleatoria", type="primary"):
        with st.spinner("Generando narrativa... (puede tardar varios minutos si el "
                         "guardarraíl de calidad necesita reintentar)"):
            try:
                r = requests.get(f"{API_URL}/demo/instancia_aleatoria", timeout=900)
                r.raise_for_status()
                st.session_state["resultado"] = r.json()
            except requests.exceptions.RequestException as exc:
                st.error(f"Error al conectar con la API ({API_URL}): {exc}")

else:
    with st.form("form_solicitante"):
        st.subheader("Datos del solicitante")
        col1, col2 = st.columns(2)
        with col1:
            checking_status = _campo_desplegable(
                "checking_status", FEATURE_VALUE_LABELS["checking_status"],
                VALORES_POR_DEFECTO["checking_status"])
            duration = st.number_input(
                FEATURE_NAME_LABELS_ES["duration"], value=VALORES_POR_DEFECTO["duration"])
            credit_history = _campo_desplegable(
                "credit_history", FEATURE_VALUE_LABELS["credit_history"],
                VALORES_POR_DEFECTO["credit_history"])
            purpose = _campo_desplegable(
                "purpose", FEATURE_VALUE_LABELS["purpose"],
                VALORES_POR_DEFECTO["purpose"])
            credit_amount = st.number_input(
                FEATURE_NAME_LABELS_ES["credit_amount"], value=VALORES_POR_DEFECTO["credit_amount"])
            savings_status = _campo_desplegable(
                "savings_status", FEATURE_VALUE_LABELS["savings_status"],
                VALORES_POR_DEFECTO["savings_status"])
            employment = _campo_desplegable(
                "employment", FEATURE_VALUE_LABELS["employment"],
                VALORES_POR_DEFECTO["employment"])
            installment_rate = _campo_desplegable(
                "installment_rate", INSTALLMENT_RATE_LABELS,
                VALORES_POR_DEFECTO["installment_rate"])
            personal_status = _campo_desplegable(
                "personal_status", FEATURE_VALUE_LABELS["personal_status"],
                VALORES_POR_DEFECTO["personal_status"])
            other_parties = _campo_desplegable(
                "other_parties", FEATURE_VALUE_LABELS["other_parties"],
                VALORES_POR_DEFECTO["other_parties"])
        with col2:
            residence_since = st.number_input(
                FEATURE_NAME_LABELS_ES["residence_since"], value=VALORES_POR_DEFECTO["residence_since"])
            property_magnitude = _campo_desplegable(
                "property_magnitude", FEATURE_VALUE_LABELS["property_magnitude"],
                VALORES_POR_DEFECTO["property_magnitude"])
            age = st.number_input(
                FEATURE_NAME_LABELS_ES["age"], value=VALORES_POR_DEFECTO["age"])
            other_payment_plans = _campo_desplegable(
                "other_payment_plans", FEATURE_VALUE_LABELS["other_payment_plans"],
                VALORES_POR_DEFECTO["other_payment_plans"])
            housing = _campo_desplegable(
                "housing", FEATURE_VALUE_LABELS["housing"],
                VALORES_POR_DEFECTO["housing"])
            existing_credits = st.number_input(
                FEATURE_NAME_LABELS_ES["existing_credits"], value=VALORES_POR_DEFECTO["existing_credits"])
            job = _campo_desplegable(
                "job", FEATURE_VALUE_LABELS["job"],
                VALORES_POR_DEFECTO["job"])
            num_dependents = _campo_desplegable(
                "num_dependents", NUM_DEPENDENTS_LABELS,
                VALORES_POR_DEFECTO["num_dependents"])
            own_telephone = _campo_desplegable(
                "own_telephone", FEATURE_VALUE_LABELS["own_telephone"],
                VALORES_POR_DEFECTO["own_telephone"])
            foreign_worker = _campo_desplegable(
                "foreign_worker", FEATURE_VALUE_LABELS["foreign_worker"],
                VALORES_POR_DEFECTO["foreign_worker"])

        enviado = st.form_submit_button("Calcular puntuación y explicación", type="primary")

    if enviado:
        payload = {
            "checking_status": checking_status, "duration": int(duration),
            "credit_history": credit_history, "purpose": purpose,
            "credit_amount": int(credit_amount), "savings_status": savings_status,
            "employment": employment, "installment_rate": int(installment_rate),
            "personal_status": personal_status, "other_parties": other_parties,
            "residence_since": int(residence_since), "property_magnitude": property_magnitude,
            "age": int(age), "other_payment_plans": other_payment_plans,
            "housing": housing, "existing_credits": int(existing_credits),
            "job": job, "num_dependents": int(num_dependents),
            "own_telephone": own_telephone, "foreign_worker": foreign_worker,
        }
        with st.spinner("Calculando puntuación y generando narrativa... (puede tardar varios "
                         "minutos si el guardarraíl de calidad necesita reintentar)"):
            try:
                r = requests.post(f"{API_URL}/score", json=payload, timeout=900)
                r.raise_for_status()
                st.session_state["resultado"] = r.json()
            except requests.exceptions.RequestException as exc:
                detalle = ""
                if exc.response is not None:
                    detalle = f" -- {exc.response.text}"
                st.error(f"Error al conectar con la API ({API_URL}): {exc}{detalle}")

if "resultado" in st.session_state:
    st.divider()
    _renderizar_resultado(st.session_state["resultado"])
