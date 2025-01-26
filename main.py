from flask import Flask, request, jsonify, Response, stream_with_context, render_template
import requests
import json
import logging
import chromadb
import unicodedata
import re
import threading
from langdetect import detect, DetectorFactory
from functools import lru_cache
import spacy
import os
from cryptography.fernet import Fernet
import pandas as pd
import time
# Define the question counter as a global dictionary
question_counter = {}
DetectorFactory.seed = 0  # Ensure consistent language detection

# Load NLP models for question detection
nlp_fr = spacy.blank("fr")
nlp_en = spacy.blank("en")

nlp_fr.add_pipe("sentencizer")
nlp_en.add_pipe("sentencizer")
def load_key():
    """Load the encryption key from a file."""
    return open("secret.key", "rb").read()
def encrypt_data(data):
    """Encrypt data using Fernet symmetric encryption."""
    key = load_key()
    cipher_suite = Fernet(key)
    encrypted_data = cipher_suite.encrypt(data.encode())
    return encrypted_data.decode()  # Convert bytes to string


def read_decrypted_logs():
    """Read and decrypt conversation logs from JSON."""
    log_file_json = "logs/conversations.json"

    # Ensure log file exists
    if not os.path.exists(log_file_json):
        print("❌ No logs found.")
        return

    # Read logs from JSON file
    with open(log_file_json, "r", encoding="utf-8") as file:
        try:
            logs = json.load(file)
        except json.JSONDecodeError:
            print("❌ Error: JSON file is corrupted or empty.")
            return

    decrypted_logs = []

    # Loop through logs and decrypt
    for entry in logs:
        try:
            decrypted_entry = {
                "timestamp": entry["timestamp"],
                "user_query": decrypt_data(entry["user_query"]),
                "response": decrypt_data(entry["response"])
            }
            decrypted_logs.append(decrypted_entry)

            # ✅ Debugging: Print encrypted vs decrypted values
            print(f"\n🔹 Encrypted Query: {entry['user_query']}")
            print(f"✅ Decrypted Query: {decrypted_entry['user_query']}")
            print(f"\n🔹 Encrypted Response: {entry['response']}")
            print(f"✅ Decrypted Response: {decrypted_entry['response']}")
            print("-" * 50)

        except Exception as e:
            print(f"❌ Error decrypting log entry: {e}")

    # Pretty print the decrypted logs
    print("\n📜 **Decrypted Logs:**")
    print(json.dumps(decrypted_logs, indent=4, ensure_ascii=False))

    return decrypted_logs  # Optional: return logs for API usage
def extract_questions(text, lang="fr"):
    """Extract valid questions from user input."""
    doc = nlp_fr(text) if lang == "fr" else nlp_en(text)
    questions = [sent.text.strip() for sent in doc.sents if sent.text.strip().endswith("?")]
    return questions

app = Flask(__name__)

# Configuration
OLLAMA_SERVER = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b-instruct"
DATASET_PATH = "data/dataset_translated_fixed.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

conversation_history = []
faq_data = {}
chroma_client = chromadb.PersistentClient(path="chroma_db")
stop_event = threading.Event()
feedback_data = {"useful": 0, "not_useful": 0}

def detect_language(text):
    """Detect language of the user query."""
    try:
        lang = detect(text)
        return "fr" if lang == "fr" else "en"
    except:
        return "fr"

def sanitize_category_name(name):
    """Sanitize category names for ChromaDB."""
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8")
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")
    return name.lower()

def load_faq_data():
    """Load FAQ dataset into memory."""
    global faq_data, chroma_client
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "dataset" in data and isinstance(data["dataset"], list):
            faq_data = {}
            for entry in data["dataset"]:
                category = sanitize_category_name(entry["category"].strip())
                question_fr, question_en = entry["question"].strip().lower(), entry["question_en"].strip().lower()
                answer_fr, answer_en = entry["answer"].strip(), entry["answer_en"].strip()

                if category not in faq_data:
                    faq_data[category] = {
                        "questions": {},
                        "collection": chroma_client.get_or_create_collection(name=f"faq_{category}")
                    }

                faq_data[category]["questions"][question_fr] = answer_fr
                faq_data[category]["questions"][question_en] = answer_en

                category_collection = faq_data[category]["collection"]
                category_collection.add(
                    ids=[question_fr, question_en],
                    documents=[question_fr, question_en],
                    metadatas=[{"answer": answer_fr}, {"answer_en": answer_en}]
                )

            logging.info(f"✅ Loaded {sum(len(cat['questions']) for cat in faq_data.values())} FAQ entries across {len(faq_data)} categories.")
    except Exception as e:
        faq_data = {}
        logging.error(f"❌ Error loading dataset: {e}")

load_faq_data()

@lru_cache(maxsize=1000)
def find_best_match(user_query, lang="fr", top_k=1):
    """Find the best match from the FAQ database."""
    best_match, best_answer, best_score = None, None, float("inf")
    for category, category_data in faq_data.items():
        results = category_data["collection"].query(query_texts=[user_query], n_results=top_k)
        if results["documents"] and results["documents"][0]:
            score = results["distances"][0][0]
            if score < best_score:
                best_score, best_match, best_answer = score, results["documents"][0][0], results["metadatas"][0][0]["answer"]
    return best_match, best_answer

def qwen_stream(history, faq_answer, user_query, lang="fr"):
    """Stream response from LLM."""
    global stop_event
    instruction = f"""
    {"Vous êtes un assistant académique intelligent." if lang == "fr" else "You are an intelligent academic assistant."}

    Répondez en adoptant un ton similaire à l'exemple suivant :
    📌 **Exemple de réponse :** {faq_answer}

    📌 **Question de l'utilisateur :** {user_query}

    ⚠️ **Important :** Ne fournissez que les informations strictement nécessaires. Évitez toute information superflue.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": instruction}] + history[-3:]
    }
    logging.info(f"📡 Sending request to LLM: {payload}")
    try:
        response = requests.post(f"{OLLAMA_SERVER}/api/chat", json=payload, stream=True)
        response.raise_for_status()
        logging.info("✅ LLM request successful.")
        for line in response.iter_lines(decode_unicode=True):
            if stop_event.is_set():
                logging.info("🛑 Streaming stopped.")
                break
            try:
                data = json.loads(line)
                message_chunk = data.get("message", {}).get("content", "")
                if message_chunk:
                    yield message_chunk + " "
            except json.JSONDecodeError as e:
                logging.error(f"❌ JSON decode error: {e}")
                continue
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error contacting LLM: {e}")
        yield "Erreur de communication avec l'IA" if lang == "fr" else "Communication error with AI"

@app.route("/conversation", methods=["POST"])
def conversation():
    global stop_event, question_counter
    stop_event.clear()

    data = request.get_json()
    logging.info(f"📥 Received request: {data}")

    history = data.get("history", [])
    if not history:
        logging.error("❌ Empty history received!")
        return jsonify({"success": False, "message": "History cannot be empty."}), 400

    user_query = history[-1].get("content", "").strip().lower()
    logging.info(f"🗣️ User query: {user_query}")

    # Track question frequency
    question_counter[user_query] = question_counter.get(user_query, 0) + 1

    retrieved_question, retrieved_answer = find_best_match(user_query)

    if retrieved_answer:
        logging.info(f"📚 Found matching FAQ response: {retrieved_answer}")
    else:
        retrieved_answer = "Je suis désolé, mais je n'ai pas trouvé de réponse dans la FAQ."

    # Log encrypted conversation
    encrypted_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": encrypt_data(user_query),
        "response": encrypt_data(retrieved_answer)
    }
    log_conversation(encrypted_log)

    return Response(
        stream_with_context(qwen_stream(conversation_history, retrieved_answer, user_query)),
        content_type="text/event-stream"
    )
def log_conversation(log_data):
    """Store encrypted logs in a JSON and CSV file."""
    log_file_json = "logs/conversations.json"
    log_file_csv = "logs/conversations.csv"

    os.makedirs("logs", exist_ok=True)  # Ensure logs folder exists

    # Append to JSON file
    if os.path.exists(log_file_json):
        with open(log_file_json, "r", encoding="utf-8") as file:
            try:
                logs = json.load(file)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append(log_data)
    with open(log_file_json, "w", encoding="utf-8") as file:
        json.dump(logs, file, ensure_ascii=False, indent=4)

    # Append to CSV file
    df = pd.DataFrame([log_data])
    if not os.path.exists(log_file_csv):
        df.to_csv(log_file_csv, mode="w", index=False)
    else:
        df.to_csv(log_file_csv, mode="a", header=False, index=False)
def decrypt_data(encrypted_text):
    """Decrypt encrypted text using Fernet."""
    key = load_key()
    cipher_suite = Fernet(key)
    decrypted_data = cipher_suite.decrypt(encrypted_text.encode())
    return decrypted_data.decode()

@app.route("/delete", methods=["POST"])
def delete_user_data():
    """Delete all records related to a user's query."""
    data = request.get_json()
    user_query = data.get("user_query", "").strip().lower()

    if not user_query:
        return jsonify({"success": False, "message": "Query required for deletion."}), 400

    encrypted_query = encrypt_data(user_query)
    log_file_json = "logs/conversations.json"
    log_file_csv = "logs/conversations.csv"

    # Remove from JSON
    if os.path.exists(log_file_json):
        with open(log_file_json, "r", encoding="utf-8") as file:
            logs = json.load(file)

        filtered_logs = [entry for entry in logs if entry["user_query"] != encrypted_query]

        with open(log_file_json, "w", encoding="utf-8") as file:
            json.dump(filtered_logs, file, ensure_ascii=False, indent=4)

    # Remove from CSV
    if os.path.exists(log_file_csv):
        df = pd.read_csv(log_file_csv)
        df = df[df["user_query"] != encrypted_query]
        df.to_csv(log_file_csv, index=False)

    return jsonify({"success": True, "message": "Data deleted successfully."})

@app.route("/feedback", methods=["POST"])
def feedback():
    """Handle user feedback."""
    data = request.get_json()
    if not data:
        return jsonify({"message": "Invalid feedback data"}), 400

    feedback_type = data.get("feedback")
    user_input = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    language = data.get("language", "fr")

    if not user_input or not answer:
        return jsonify({"message": "Missing question or answer"}), 400

    if feedback_type not in ["useful", "not_useful"]:
        return jsonify({"message": "Invalid feedback type"}), 400

    dataset_file = DATASET_PATH
    if os.path.exists(dataset_file):
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = {"dataset": []}

    valid_questions = extract_questions(user_input, lang=language)
    if not valid_questions:
        return jsonify({"message": "No valid question detected, feedback recorded."})

    for question in valid_questions:
        dataset["dataset"].append({
            "category": "user_feedback",
            "question": question if language == "fr" else "",
            "question_en": question if language == "en" else "",
            "answer": answer if language == "fr" else "",
            "answer_en": answer if language == "en" else ""
        })

    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    return jsonify({"message": "Feedback recorded and updated!"}), 200

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
