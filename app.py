# app.py
import streamlit as st
import json
import os
import time
from datetime import datetime
import dateparser
import numpy as np
import random 

# --- Optional imports for better semantic search ---
use_embeddings = True
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    use_embeddings = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

# ---------------------
# CONFIG
# ---------------------
st.set_page_config(page_title="Elite Body Home Clinic Chatbot", page_icon="💬", layout="centered")
DATA_FILE = "elite_data.json"
BOOKINGS_FILE = "bookings.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.45

# ---------------------
# UTILITIES
# ---------------------
def load_json(path, default):
    """Safely load a JSON file or create it if missing/empty."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read().strip()
                if not data:
                    raise ValueError("Empty file")
                return json.loads(data)
        except Exception:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2)
            return default
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

# ---------------------
# Load data
# ---------------------
kb = load_json(DATA_FILE, default=[])
bookings = load_json(BOOKINGS_FILE, default=[])

# ---------------------
# Corpus preparation
# ---------------------
corpus_texts = []
corpus_meta = []
for item in kb:
    if item.get("category") == "service":
        text = f"SERVICE: {item.get('name', '')} -- {item.get('details','')}"
        meta = {"type": "service", "id": item.get("id"), "name": item.get("name")}
    elif item.get("category") == "general":
        text = f"Q: {item.get('question','')} A: {item.get('answer','')}"
        meta = {"type": "general"}
    else:
        text = json.dumps(item)
        meta = {"type": "other"}
    corpus_texts.append(text)
    corpus_meta.append(meta)

# ---------------------
# Embeddings / TF-IDF
# ---------------------
if use_embeddings:
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        corpus_embeddings = embedder.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        st.warning("⚠️ Embedding model failed, switching to TF-IDF.")
        use_embeddings = False

if not use_embeddings:
    tfidf = TfidfVectorizer().fit(corpus_texts)
    corpus_tfidf = tfidf.transform(corpus_texts)

def retrieve_answer(query, top_k=3):
    if len(corpus_texts) == 0:
        return None, []
    if use_embeddings:
        q_emb = embedder.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, corpus_embeddings)[0]
    else:
        q_tfidf = tfidf.transform([query])
        sims = linear_kernel(q_tfidf, corpus_tfidf).flatten()

    idx_sorted = np.argsort(-sims)[:top_k]
    results = [{"score": float(sims[i]), "text": corpus_texts[i], "meta": corpus_meta[i]} for i in idx_sorted]
    return results[0], results

# ---------------------
# Session state
# ---------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "booking_state" not in st.session_state:
    st.session_state.booking_state = {"active": False, "service": None, "date": None, "time": None}
if "typing" not in st.session_state:
    st.session_state.typing = False

# ---------------------
# UI Styling
# ---------------------
BUBBLE_CSS = """
<style>
.chat-container {
  max-width: 760px;
  margin: 0 auto;
  color: black;
  font-size: 15px;
}
.msg-row {
  display: flex;
  margin-bottom: 8px;
}
.msg-row.user {
  justify-content: flex-end;
}
.msg-row.bot {
  justify-content: flex-start;
}
.msg {
  max-width: 78%;
  padding: 10px 14px;
  border-radius: 12px;
  background: #f7f7f7;
  line-height: 1.4;
  white-space: pre-wrap;
  color: black;
  box-shadow: 0 1px 2px rgba(0,0,0,0.08);
}
.msg.user {
  background: #DCF8C6;
}
.time {
  font-size: 11px;
  color: #444;
  margin-top: 4px;
  text-align:right;
}
.typing {
  width: 46px;
  height: 18px;
  border-radius: 12px;
  background: #eee;
  display: inline-block;
  padding: 3px 6px;
}
</style>
"""
st.markdown(BUBBLE_CSS, unsafe_allow_html=True)

# ---------------------
# Chat Helpers
# ---------------------
def append_message(role, text):
    st.session_state.messages.append({"role": role, "text": text, "ts": datetime.now().isoformat()})

def show_messages():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        role = m["role"]
        text = m["text"]
        ts = datetime.fromisoformat(m["ts"]).strftime("%Y-%m-%d %H:%M")
        css_row = 'user' if role == 'user' else 'bot'
        html = f"""
        <div class="msg-row {css_row}">
          <div class="msg">{text}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------
# Bot Logic
# ---------------------
def generate_bot_response(user_text):
    # ----------------- smalltalk check -----------------
    for item in kb:
        if item.get("category") == "smalltalk":
            for example in item.get("examples", []):
                if example.lower() in user_text.strip().lower():
                    response = random.choice(item.get("responses", []))
                    append_message("bot", response)
                    return

    bs = st.session_state.booking_state
    lower = user_text.strip().lower()

    # ----------------- Booking flow -----------------
    if bs["active"]:
        if not bs["service"]:
            bs["service"] = user_text.strip()
            append_message("bot", f"Got it. For {bs['service']}, which date would you prefer? (e.g., 2025-10-22 or 'tomorrow')")
            return
        if not bs["date"]:
            dt = dateparser.parse(user_text, settings={'PREFER_DATES_FROM': 'future'})
            if not dt:
                append_message("bot", "Please provide a valid date like '2025-10-22' or 'tomorrow'.")
                return
            bs["date"] = dt.date().isoformat()
            append_message("bot", f"Nice. What time would you like on {bs['date']}? (e.g., 16:00 or 4pm)")
            return
        if not bs["time"]:
            dt = dateparser.parse(user_text)
            if not dt:
                append_message("bot", "Please give a valid time like '16:00' or '4pm'.")
                return
            bs["time"] = dt.time().strftime("%H:%M")

            booking = {
                "id": f"bk_{int(time.time())}",
                "service": bs["service"],
                "date": bs["date"],
                "time": bs["time"],
                "created_at": datetime.now().isoformat()
            }
            bookings.append(booking)
            save_json(BOOKINGS_FILE, bookings)
            append_message("bot", f"✅ Booking confirmed for *{booking['service']}* on {booking['date']} at {booking['time']}.")
            st.session_state.booking_state = {"active": False, "service": None, "date": None, "time": None}
            return
    # Start booking intent
    booking_keywords = ["book", "appointment", "reserve", "schedule"]
    if any(k in lower for k in booking_keywords):
        possible_service = None
        for item in kb:
            if item.get("category") == "service":
                name = item.get("name", "").lower()
                if name in lower:
                    possible_service = item.get("name")
                    break
        st.session_state.booking_state = {"active": True, "service": possible_service, "date": None, "time": None}
        if possible_service:
            append_message("bot", f"Sure — booking for *{possible_service}*. Which date works for you?")
        else:
            available = [it.get("name") for it in kb if it.get("category") == "service"]
            avail_str = ", ".join(available) if available else "the service you want"
            append_message("bot", f"Sure — which service do you want to book? Available: {avail_str}")
        return

    # Normal question answering
    best, _ = retrieve_answer(user_text)
    if not best or best["score"] < SIMILARITY_THRESHOLD:
        append_message("bot", "I couldn’t find that info right now. Would you like to book an appointment?")
        return

    meta = best["meta"]
    if meta.get("type") == "service":
        reply = best["text"].replace("SERVICE:", "").strip()
        reply += f"\n\nTo book *{meta.get('name')}*, type 'book {meta.get('name')}'."
        append_message("bot", reply)
    else:
        append_message("bot", best["text"])

# ---------------------
# Page Layout
# ---------------------
st.title("💬 Elite Body Home Clinic Chatbot")
st.caption("Talk to this chatbot like WhatsApp — ask questions or book appointments instantly.")

# Display chat

user_input = st.chat_input("Type your message...")

if user_input:
    append_message("user", user_input)
    st.session_state.typing = True
    time.sleep(0.5)
    st.session_state.typing = False

    try:
        generate_bot_response(user_input)
    except Exception as e:
        append_message("bot", f"⚠️ Error: {str(e)}")

show_messages()
