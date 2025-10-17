import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# ==============================
# CONFIG
# ==============================
DATA_FILE = "elite_data.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, offline-compatible

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data()

# Prepare corpus
corpus = []
for item in data:
    if item.get("category") == "service":
        corpus.append(item["details"])
    elif item.get("category") == "general":
        corpus.append(item["answer"])

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()
corpus_embeddings = model.encode(corpus)

# ==============================
# CHAT FUNCTION
# ==============================
def chatbot_response(query):
    query_emb = model.encode([query])
    sim_scores = cosine_similarity(query_emb, corpus_embeddings)[0]
    best_idx = int(np.argmax(sim_scores))
    return corpus[best_idx]

# ==============================
# BOOKING FUNCTION
# ==============================
def book_appointment(name, contact, treatment, date, time):
    with open("bookings.csv", "a", encoding="utf-8") as f:
        f.write(f"{name},{contact},{treatment},{date},{time}\n")
    return f"✅ Appointment booked for {name} on {date} at {time} for {treatment}."

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Elite Body Home Clinic Chatbot", page_icon="💬")

tabs = st.tabs(["💬 Chat", "🗓️ Book Appointment"])

# ------------------------------
# CHAT TAB
# ------------------------------
with tabs[0]:
    st.subheader("Chat with Elite Body Home Bot")
    user_input = st.text_input("Ask here:")
    if st.button("Send"):
        if user_input.strip():
            response = chatbot_response(user_input)
            st.write(response)
        else:
            st.warning("Please type something to ask.")

# ------------------------------
# BOOKING TAB
# ------------------------------
with tabs[1]:
    st.subheader("Book an Appointment")
    name = st.text_input("Full Name")
    contact = st.text_input("Contact Info")
    treatment = st.selectbox("Treatment", [d["name"] for d in data if d["category"] == "service"])
    date = st.date_input("Preferred Date", datetime.date.today())
    time = st.time_input("Preferred Time", datetime.time(10, 0))
    
    if st.button("Book Appointment", use_container_width=True):
        if not name or not contact:
            st.error("Please fill all fields.")
        else:
            msg = book_appointment(name, contact, treatment, date, time)
            st.success(msg)
