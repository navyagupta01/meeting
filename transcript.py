import os
import re
import time
import json
import queue
import nltk
import threading
import requests
import numpy as np
import sounddevice as sd
import webrtcvad
import speech_recognition as sr
import pandas as pd
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pylatex import Document, Section, Command
from pylatex.utils import NoEscape
import firebase_admin
from firebase_admin import credentials, firestore
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

# Load Models
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Google Drive Authentication
SCOPES = ['https://www.googleapis.com/auth/drive.file']
creds = None
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
if not creds or not creds.valid:
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
drive_service = build('drive', 'v3', credentials=creds)

# Global Vars
vad = webrtcvad.Vad(3)
transcription_queue = queue.Queue()
unique_words = set()
session_transcriptions = []
processing_active = True
session_start_time = time.time()
SESSION_DURATION = 25 * 60
session_id = str(int(time.time()))

# Utility Functions
def frame_generator(frame_duration_ms, audio, sample_rate):
    frame_size = int(sample_rate * frame_duration_ms / 1000) * 2
    offset = 0
    while offset + frame_size <= len(audio):
        yield audio[offset:offset + frame_size]
        offset += frame_size

def transcribe_audio(audio_data, sample_rate=16000, language='en-US'):
    recognizer = sr.Recognizer()
    audio = sr.AudioData(audio_data, sample_rate, 2)
    try:
        text = recognizer.recognize_google(audio, language=language, show_all=True)
        if isinstance(text, dict) and 'alternative' in text:
            transcript = text['alternative'][0]['transcript']
            confidence = text['alternative'][0].get('confidence', 0.9)
            return transcript, confidence
    except (sr.UnknownValueError, sr.RequestError):
        return None, 0.0
    return None, 0.0

def update_unique_words(text):
    if text:
        words = word_tokenize(text.lower())
        words = [re.sub(r'[^\w\s]', '', word) for word in words]
        stop_words = set(stopwords.words('english'))
        unique_words.update(word for word in words if word and word not in stop_words)
    return sorted(unique_words)

def extract_keywords(text, subject):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    subject_keywords = {
        'mathematics': ['equation', 'theorem', 'function', 'derivative', 'integral'],
        'science': ['experiment', 'hypothesis', 'molecule', 'energy', 'reaction'],
        'history': ['revolution', 'empire', 'treaty', 'era', 'civilization'],
        'literature': ['theme', 'character', 'plot', 'symbolism', 'narrative']
    }
    default_keywords = subject_keywords.get(subject.lower(), [])
    keywords = [word for word in words if word in default_keywords or word not in stop_words]
    keyword_freq = {word: words.count(word) for word in set(keywords)}
    return list(set(keywords))[:10], keyword_freq

def generate_subject_notes(text, language='en'):
    if not text:
        return "No valid text to generate notes.", "Unknown", 0.0

    subject_prompt = f"Identify the academic subject of the following text: {text}"
    inputs = tokenizer(subject_prompt, return_tensors="pt", truncation=True)
    subject_output = model.generate(inputs.input_ids, max_length=50)
    subject = tokenizer.decode(subject_output[0], skip_special_tokens=True)

    notes_prompt = f"""
    Summarize the following {subject} class content into concise notes in {language}.
    Include a clear explanation of the main topic and one practical example.
    Format: Subject, Topic, Explanation, Example. Content: {text}
    """
    inputs = tokenizer(notes_prompt, return_tensors="pt", truncation=True)
    output = model.generate(inputs.input_ids, max_length=512)
    notes = tokenizer.decode(output[0], skip_special_tokens=True)
    return notes, subject, 0.9

def generate_session_summary(transcriptions):
    combined = " ".join(transcriptions)
    prompt = f"Summarize this class session: {combined[:1000]}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = model.generate(inputs.input_ids, max_length=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_tts(notes, session_id, language='en'):
    try:
        tts = gTTS(text=notes, lang=language)
        path = f"tts_{session_id}.mp3"
        tts.save(path)
        drive_id = upload_to_drive(path, path)
        return path, drive_id
    except Exception as e:
        logging.error(f"TTS generation error: {e}")
        return None, None

def generate_pdf(notes, keywords, session_id):
    doc = Document()
    doc.preamble.append(Command('title', f'Session {session_id} Notes'))
    doc.preamble.append(Command('author', 'Grok'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))
    with doc.create(Section("Notes")):
        doc.append(notes.replace('\n', r'\\n'))
    with doc.create(Section("Key Terms")):
        doc.append(NoEscape(r'\begin{itemize}'))
        for kw in keywords:
            doc.append(NoEscape(f'\\item {kw}'))
        doc.append(NoEscape(r'\end{itemize}'))
    doc.generate_pdf(f'session_{session_id}', clean_tex=True)
    path = f'session_{session_id}.pdf'
    drive_id = upload_to_drive(path, path)
    return path, drive_id

def upload_to_drive(file_path, file_name):
    try:
        file_metadata = {'name': file_name}
        media = MediaFileUpload(file_path)
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
    except Exception as e:
        logging.error(f"Drive upload error: {e}")
        return None

def generate_notes_excel(notes, keywords, subject, transcriptions):
    data = {
        "Session ID": [session_id],
        "Subject": [subject],
        "Notes": [notes],
        "Keywords": [", ".join(keywords)],
        "Transcription": [" ".join(transcriptions)],
        "Timestamp": [datetime.now().isoformat()]
    }
    df = pd.DataFrame(data)
    df.to_excel(f"notes_{session_id}.xlsx", index=False)

def send_session_data(transcription=None, notes=None, words=None, subject=None, confidence=None, keywords=None, keyword_freq=None):
    data = {
        'id': session_id,
        'timestamp': datetime.now().isoformat(),
        'transcription': transcription,
        'notes': notes,
        'words': ", ".join(words or []),
        'subject': subject,
        'confidence': confidence,
        'keywords': keywords or [],
        'keyword_freq': keyword_freq or {},
        'language': 'en'
    }
    try:
        db.collection("sessions").document(session_id).set(data, merge=True)
        if notes and keywords and subject:
            generate_notes_excel(notes, keywords, subject, session_transcriptions)
            generate_pdf(notes, keywords, session_id)
    except Exception as e:
        logging.error(f"Firebase write error: {e}")

def audio_processing_thread():
    global processing_active, session_transcriptions, session_start_time
    with sd.RawInputStream(samplerate=16000, channels=1, dtype='int16', callback=audio_stream_callback):
        while True:
            if time.time() - session_start_time > SESSION_DURATION:
                summary = generate_session_summary(session_transcriptions)
                db.collection("sessions").document(session_id).update({"summary": summary})
                session_transcriptions.clear()
                unique_words.clear()
                session_start_time = time.time()
                session_id = str(int(time.time()))  # Update session_id without global declaration

            try:
                audio_data = transcription_queue.get(timeout=1)
                transcription, confidence = transcribe_audio(audio_data)
                if not transcription:
                    continue

                trans_lower = transcription.lower()
                if "stop processing" in trans_lower:
                    processing_active = False
                    continue
                elif "start processing" in trans_lower:
                    processing_active = True
                    continue
                elif "new session" in trans_lower:
                    session_transcriptions.clear()
                    unique_words.clear()
                    session_id = str(int(time.time()))
                    continue

                session_transcriptions.append(transcription)
                words = update_unique_words(transcription)
                notes, subject, confidence = generate_subject_notes(transcription)
                keywords, freq = extract_keywords(transcription, subject)
                send_session_data(transcription, notes, words, subject, confidence, keywords, freq)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Audio thread error: {e}")

def audio_stream_callback(indata, frames, time_info, status):
    if not processing_active:
        return
    audio = np.frombuffer(indata, dtype=np.int16).tobytes()
    try:
        for frame in frame_generator(30, audio, 16000):
            if vad.is_speech(frame, sample_rate=16000):
                transcription_queue.put(frame)
    except webrtcvad.VadError as e:
        logging.error(f"VAD processing error: {e}")

if __name__ == "__main__":
    threading.Thread(target=audio_processing_thread, daemon=True).start()
    while True:
        time.sleep(1)