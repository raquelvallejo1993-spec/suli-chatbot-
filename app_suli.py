"""
Chatbot de FAQs de SULI usando Ollama Cloud
Bootcamp SKALA / NVIDIA AI

Cambio de arquitectura: en vez de NVIDIA NIM (donde el token no tenía
permiso de inferencia), usamos Ollama Cloud, que expone una API
compatible con OpenAI. Mismo patrón de "orquestador -> servicio de
inferencia -> modelo", solo que aquí el servicio es Ollama Cloud
en vez de NIM.

Requisitos:
    pip install streamlit openai

Cómo correrlo:
    streamlit run app_suli.py

Necesitas una cuenta gratuita en https://ollama.com y una API key
(sección "API Keys" de tu cuenta).
"""

import os
import streamlit as st
from openai import OpenAI

FAQ_PATH = os.path.join(os.path.dirname(__file__), "suli_faq.md")
DRAGON_AVATAR = "https://suli.mentalnetwork360.org/suli-hero.png"


@st.cache_data
def cargar_base_de_conocimiento():
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        return f.read()


def construir_system_prompt(base_conocimiento: str) -> str:
    return f"""Eres el asistente virtual de SULI (Salud que Une Liderazgos),
un proyecto de Mental Network 360 (MNT360) de salud mental para
organizaciones y sus colaboradores.

Tu trabajo es responder preguntas frecuentes sobre SULI usando ÚNICAMENTE
la información de la base de conocimiento de abajo. Reglas:

- Responde en español, con un tono cálido, claro y profesional (no clínico
  ni acartonado).
- Si la pregunta no está cubierta en la base de conocimiento, dilo
  honestamente y sugiere escribir a hola@mentalnetwork360.org o usar
  el botón "Solicitar propuesta" del sitio, en vez de inventar una respuesta.
- Nunca dés consejo terapéutico, diagnóstico ni contención clínica: tu rol
  es informativo, sobre cómo funciona el servicio.
- Si detectas que la persona (o alguien que menciona) podría estar en
  riesgo inmediato, indícale con claridad que contacte servicios de
  emergencia locales y/o escriba directamente a SULI por WhatsApp
  (https://wa.me/525534951828) para que un profesional la atienda —
  no intentes manejar tú la situación.

BASE DE CONOCIMIENTO:
{base_conocimiento}
"""


def aplicar_estilo_suli():
    st.markdown("""
    <style>
    :root {
        --suli-verde: #3F6B4E;
        --suli-verde-claro: #6F9C7C;
        --suli-crema: #FAF6F0;
        --suli-terracota: #D98B6C;
        --suli-texto: #2B2E2A;
    }

    .stApp {
        background-color: var(--suli-crema);
        color: var(--suli-texto);
    }
    .stApp .stMarkdown, .stApp .stMarkdown p, .stApp .stMarkdown li,
    .stApp .stMarkdown strong, .stApp .stMarkdown table, .stApp .stMarkdown td, .stApp .stMarkdown th {
        color: var(--suli-texto) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #2F4A38;
    }
    section[data-testid="stSidebar"] * {
        color: #F2EFE9 !important;
    }
    section[data-testid="stSidebar"] input {
        background-color: #3F6B4E !important;
        color: #FFFFFF !important;
        border: 1px solid var(--suli-verde-claro) !important;
    }

    h1, h2, h3 {
        color: var(--suli-verde) !important;
        font-family: -apple-system, "Georgia", serif;
    }

    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 4px 8px;
    }
    [data-testid="stChatMessageContent"] {
        background-color: #FFFFFF;
        border-radius: 14px;
        border: 1px solid #E4DCCF;
        color: var(--suli-texto) !important;
    }
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span,
    [data-testid="stChatMessageContent"] strong,
    [data-testid="stChatMessageContent"] * {
        color: var(--suli-texto) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * {
        background-color: var(--suli-verde);
        color: #FFFFFF !important;
        border: none;
    }

    .stChatInputContainer, [data-testid="stChatInput"] {
        border: 1.5px solid var(--suli-verde) !important;
        border-radius: 12px !important;
    }

    .stButton > button, [data-testid="baseButton-secondary"] {
        background-color: var(--suli-terracota) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div {
        border-color: var(--suli-verde-claro) !important;
        border-radius: 10px !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #5C6B60 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="SULI · Asistente", page_icon="🌿", layout="centered")
    aplicar_estilo_suli()
    st.title("🌿 Asistente de SULI")
    st.caption("Resuelve dudas frecuentes sobre el programa de salud mental para organizaciones")

    with st.sidebar:
        st.header("Configuración")
        api_key = st.text_input("Ollama API Key", type="password", placeholder="pega tu key aquí")

        modelo = None
        if api_key:
            try:
                client_tmp = OpenAI(base_url="https://ollama.com/v1", api_key=api_key)
                modelos_disponibles = [m.id for m in client_tmp.models.list().data]
                # modelos :cloud primero (más ligeros de correr, no requieren GPU tuya)
                modelos_disponibles.sort(key=lambda m: ("cloud" not in m, m))
                modelo = st.selectbox("Modelo", modelos_disponibles)
            except Exception as e:
                st.error(f"No se pudo listar modelos: {e}")

    base_conocimiento = cargar_base_de_conocimiento()

    if "historial" not in st.session_state:
        st.session_state.historial = []

    for msg in st.session_state.historial:
        avatar = DRAGON_AVATAR if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

    pregunta = st.chat_input("Escribe tu pregunta sobre SULI...")
    if pregunta:
        if not api_key or not modelo:
            st.error("Pon tu API key de Ollama en la barra lateral y elige un modelo.")
            return

        with st.chat_message("user"):
            st.write(pregunta)
        st.session_state.historial.append({"role": "user", "content": pregunta})

        with st.chat_message("assistant", avatar=DRAGON_AVATAR):
            with st.spinner("Pensando..."):
                client = OpenAI(base_url="https://ollama.com/v1", api_key=api_key)
                mensajes = [
                    {"role": "system", "content": construir_system_prompt(base_conocimiento)}
                ] + st.session_state.historial

                respuesta = client.chat.completions.create(
                    model=modelo,
                    messages=mensajes,
                    temperature=0.3,
                )
                texto = respuesta.choices[0].message.content
                st.write(texto)

        st.session_state.historial.append({"role": "assistant", "content": texto})


if __name__ == "__main__":
    main()
