from fastapi import FastAPI, Form, UploadFile, File, Request, Response, HTTPException
import requests, json
from datetime import datetime
from starlette.responses import JSONResponse
import shutil
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import traceback
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import subprocess
import logging
from passlib.context import CryptContext
import os
import shutil
from datetime import datetime
from pathlib import Path
import uuid
import docx
import PyPDF2
import io
from typing import List, Optional
from bson import ObjectId
import googletrans
from googletrans import Translator
from pydub import AudioSegment
import speech_recognition as sr

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


translator = Translator()
uri = "mongodb://alex:SUhvupjOhe2n72bx@cluster0-shard-00-00.un4il.mongodb.net:27017,cluster0-shard-00-01.un4il.mongodb.net:27017,cluster0-shard-00-02.un4il.mongodb.net:27017/?ssl=true&replicaSet=atlas-8pso80-shard-0&authSource=admin&retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))
db = client["hc_db"]
user_collection = db["users"]
chat_collection = db["chat_history"]

SUPPORTED_LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "Haitian Creole": "ht",
    "Jamaican Patois": "jam"
}
SUPPORTED_SPEECH_LANGUAGES = {
    "English": "en-US",
    "French": "fr-FR",
    "Spanish": "es-ES",
    "Haitian Creole": "ht-HT",
    "Japanese": "ja-JP"
}

SUPPORTED_TTS_LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "Haitian Creole": "ht",
    "Japanese": "ja"
}

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_UPLOAD_DIR = Path("./audio_uploads")
AUDIO_UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_OUTPUT_DIR = Path("./audio_outputs")
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@app.get('/')
def index():
    return {'message':"FastAPI server is running"}

@app.post("/signin")
async def login(request: Request, response: Response, email: str = Form(...), password: str = Form(...)):
    try:
        print(email, password)
        user = user_collection.find_one({"email": email})

        if user:
            if pwd_context.verify(password, user["password"]):  # Verify hashed password
                # Successful login
                response.status_code = 200
                return {"message": "Login successful", "user_id": str(user["_id"]), "username": user["username"]} # Include user details

            else:
                # Incorrect password
                response.status_code = 401  # Unauthorized
                return {"error": "Incorrect password"}

        else:
            # User not found
            response.status_code = 404  # Not Found
            return {"error": "User not found"}

    except Exception as e:
        traceback.print_exc() # Print the exception for debugging
        response.status_code = 500  # Internal Server Error
        return {"error": "An error occurred during login", "error": str(e)}

@app.post("/register")
async def register(
    request: Request,
    response: Response,
    name: str = Form(...),
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirmpass: str = Form(...)  
):
    try:
        if user_collection.find_one({"username": username}):
            return JSONResponse(status_code=400, content={"error": "Username already exists"})  # Consistent use of JSONResponse

        if user_collection.find_one({"email": email}):
            return JSONResponse(status_code=400, content={"error": "Email already exists"})

        hashed_password = pwd_context.hash(password)
        new_user = {
            "fullname": name,
            "username": username,
            "email": email,
            "password": hashed_password,  
            "created_at": datetime.utcnow(),  
            "updated_at": datetime.utcnow(),  
        }
        result = user_collection.insert_one(new_user)

        response.status_code = 201
        return {"message": "User registered successfully", "user_id": str(result.inserted_id), "username": username}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "An error occurred during registration", "error": str(e)})

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_file(file_path):
    """Extract text from different file formats"""
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.doc':
        # For .doc files we might need a different library or external tool
        # This is a placeholder - consider using a doc to docx converter or external tool
        return "DOC format not supported directly. Please convert to DOCX or PDF."
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        return "Unsupported file format"

@app.post("/create-translation-chat")
async def create_translation_chat(
    request: Request,
    response: Response,
    chat_name: str = Form(...),
    user_id: str = Form(...)
):
    try:
        # Create a new chat entry
        chat_id = str(uuid.uuid4())
        new_chat = {
            "chat_id": chat_id,
            "user_id": user_id,
            "chat_name": chat_name,
            "chat_type": "document_translation",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "translations": []
        }
        
        result = chat_collection.insert_one(new_chat)
        
        return {
            "message": "Translation chat created successfully",
            "chat_id": chat_id,
            "chat_name": chat_name
        }
    
    except Exception as e:
        response.status_code = 500
        return {"error": f"Failed to create translation chat: {str(e)}"}

@app.post("/translate-document")
async def translate_document(
    request: Request,
    response: Response,
    chat_id: str = Form(...),
    user_id: str = Form(...),
    input_language: str = Form(...),
    output_language: str = Form(...),
    document: UploadFile = File(...)
):
    try:
        # Validate languages
        if input_language not in SUPPORTED_LANGUAGES or output_language not in SUPPORTED_LANGUAGES:
            response.status_code = 400
            return {"error": "Unsupported language selection"}
        
        # Create a unique filename
        file_extension = os.path.splitext(document.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_location = UPLOAD_DIR / unique_filename
        
        # Save the uploaded file
        with open(file_location, "wb") as file_object:
            shutil.copyfileobj(document.file, file_object)
        
        # Extract text from the file
        original_text = extract_text_from_file(file_location)
        
        if not original_text or original_text.startswith("Unsupported"):
            response.status_code = 400
            return {"error": "Could not extract text from the document or unsupported format"}
        
        # Translate the text
        try:
            translated_text = translator.translate(
                original_text,
                src=SUPPORTED_LANGUAGES[input_language],
                dest=SUPPORTED_LANGUAGES[output_language]
            ).text
        except Exception as e:
            # Fallback message if translation API fails
            translated_text = f"Translation failed: {str(e)}"
        
        # Store the translation in the chat history
        translation_entry = {
            "timestamp": datetime.utcnow(),
            "document_name": document.filename,
            "input_language": input_language,
            "output_language": output_language,
            "original_text": original_text,
            "translated_text": translated_text
        }
        
        # Update the chat history
        chat_collection.update_one(
            {"chat_id": chat_id, "user_id": user_id},
            {
                "$push": {"translations": translation_entry},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        # Clean up the file after processing
        os.remove(file_location)
        
        return {
            "message": "Document translated successfully",
            "original_text": original_text,
            "translated_text": translated_text,
            "input_language": input_language,
            "output_language": output_language
        }
        
    except Exception as e:
        # Clean up in case of error
        if 'file_location' in locals() and os.path.exists(file_location):
            os.remove(file_location)
            
        response.status_code = 500
        return {"error": f"Translation failed: {str(e)}"}

@app.get("/get-translation-history/{user_id}")
async def get_translation_history(user_id: str, response: Response):
    try:
        # Get all translation chats for this user
        chats = list(chat_collection.find(
            {"user_id": user_id, "chat_type": "document_translation"},
            {"translations": 0}  # Exclude the full translations to keep response size manageable
        ))
        
        # Convert ObjectId to string
        for chat in chats:
            if "_id" in chat:
                chat["_id"] = str(chat["_id"])
        
        return {
            "message": "Translation history retrieved successfully",
            "chats": chats
        }
        
    except Exception as e:
        response.status_code = 500
        return {"error": f"Failed to retrieve translation history: {str(e)}"}

@app.get("/get-translation-chat/{chat_id}")
async def get_translation_chat(chat_id: str, user_id: str, response: Response):
    try:
        # Get detailed chat with translations
        chat = chat_collection.find_one({"chat_id": chat_id, "user_id": user_id})
        
        if not chat:
            response.status_code = 404
            return {"error": "Chat not found"}
        
        # Convert ObjectId to string
        if "_id" in chat:
            chat["_id"] = str(chat["_id"])
        
        return {
            "message": "Translation chat retrieved successfully",
            "chat": chat
        }
        
    except Exception as e:
        response.status_code = 500
        return {"error": f"Failed to retrieve translation chat: {str(e)}"}

def convert_audio_to_wav(input_file, output_file):
    """Convert uploaded audio to WAV format for speech recognition"""
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")
    return output_file

def transcribe_audio(audio_file_path, language_code):
    """Transcribe audio file to text using the specified language"""
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(str(audio_file_path)) as source:
        audio_data = recognizer.record(source)
        
    try:
        # Using Google's speech recognition service
        text = recognizer.recognize_google(audio_data, language=language_code)
        return text
    except sr.UnknownValueError:
        return "Speech could not be recognized"
    except sr.RequestError as e:
        return f"Could not request results from speech recognition service: {e}"

@app.post("/create-speech-to-text-chat")
async def create_speech_to_text_chat(
    request: Request,
    response: Response,
    chat_name: str = Form(...),
    user_id: str = Form(...)
):
    try:
        # Create a new chat entry
        chat_id = str(uuid.uuid4())
        new_chat = {
            "chat_id": chat_id,
            "user_id": user_id,
            "chat_name": chat_name,
            "chat_type": "speech_to_text",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "transcriptions": []
        }
        
        result = chat_collection.insert_one(new_chat)
        
        return {
            "message": "Speech to text chat created successfully",
            "chat_id": chat_id,
            "chat_name": chat_name
        }
    
    except Exception as e:
        response.status_code = 500
        return {"error": f"Failed to create speech to text chat: {str(e)}"}

@app.post("/transcribe-speech")
async def transcribe_speech(
    request: Request,
    response: Response,
    chat_id: str = Form(...),
    user_id: str = Form(...),
    input_language: str = Form(...),
    audio_file: UploadFile = File(...)
):
    audio_file_path = None
    wav_file_path = None
    
    try:
        # Validate language
        if input_language not in SUPPORTED_SPEECH_LANGUAGES:
            response.status_code = 400
            return {"error": f"Unsupported language: {input_language}. Supported languages are: {', '.join(SUPPORTED_SPEECH_LANGUAGES.keys())}"}
        
        # Create unique filenames
        original_filename = audio_file.filename
        file_extension = os.path.splitext(original_filename)[1].lower()
        unique_id = str(uuid.uuid4())
        
        # Save the uploaded audio file
        audio_file_path = AUDIO_UPLOAD_DIR / f"{unique_id}{file_extension}"
        with open(audio_file_path, "wb") as file_object:
            shutil.copyfileobj(audio_file.file, file_object)
        
        # Convert to WAV if not already in WAV format
        if file_extension != '.wav':
            wav_file_path = AUDIO_UPLOAD_DIR / f"{unique_id}.wav"
            convert_audio_to_wav(audio_file_path, wav_file_path)
        else:
            wav_file_path = audio_file_path
        
        # Get language code for speech recognition
        language_code = SUPPORTED_SPEECH_LANGUAGES[input_language]
        
        # Transcribe the audio
        transcribed_text = transcribe_audio(wav_file_path, language_code)
        
        # Store the transcription in the chat history
        transcription_entry = {
            "timestamp": datetime.utcnow(),
            "audio_filename": original_filename,
            "input_language": input_language,
            "transcribed_text": transcribed_text
        }
        
        # Update the chat history
        chat_collection.update_one(
            {"chat_id": chat_id, "user_id": user_id},
            {
                "$push": {"transcriptions": transcription_entry},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return {
            "message": "Audio transcribed successfully",
            "transcribed_text": transcribed_text,
            "input_language": input_language
        }
        
    except Exception as e:
        response.status_code = 500
        return {"error": f"Transcription failed: {str(e)}"}
        
    finally:
        # Clean up files
        if audio_file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        if wav_file_path and wav_file_path != audio_file_path and os.path.exists(wav_file_path):
            os.remove(wav_file_path)

@app.get("/get-speech-to-text-history/{user_id}")
async def get_speech_to_text_history(user_id: str, response: Response):
    try:
        # Get all speech to text chats for this user
        chats = list(chat_collection.find(
            {"user_id": user_id, "chat_type": "speech_to_text"},
            {"transcriptions": 0}  # Exclude the full transcriptions to keep response size manageable
        ))
        
        # Convert ObjectId to string
        for chat in chats:
            if "_id" in chat:
                chat["_id"] = str(chat["_id"])
        
        return {
            "message": "Speech to text history retrieved successfully",
            "chats": chats
        }
        
    except Exception as e:
        response.status_code = 500
        return {"error": f"Failed to retrieve speech to text history: {str(e)}"}

@app.get("/get-speech-to-text-chat/{chat_id}")
async def get_speech_to_text_chat(chat_id: str, user_id: str, response: Response):
    try:
        # Get detailed chat with transcriptions
        chat = chat_collection.find_one({"chat_id": chat_id, "user_id": user_id})
        
        if not chat:
            response.status_code = 404
            return {"error": "Chat not found"}
        
        # Convert ObjectId to string
        if "_id" in chat:
            chat["_id"] = str(chat["_id"])
        
        return {
            "message": "Speech to text chat retrieved successfully",
            "chat": chat
        }
        
    except Exception as e:
        response.status_code = 500
        return {"error": f"Failed to retrieve speech to text chat: {str(e)}"}

@app.post("/create-text-to-speech-chat")
async def create_text_to_speech_chat(
    request: Request,
    response: Response,
    chat_name: str = Form(...),
    user_id: str = Form(...)
):
    try:
        # Create a new chat entry
        chat_id = str(uuid.uuid4())
        new_chat = {
            "chat_id": chat_id,
            "user_id": user_id,
            "chat_name": chat_name,
            "chat_type": "text_to_speech",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "conversions": []
        }
        
        result = chat_collection.insert_one(new_chat)
        
        return {
            "message": "Text to speech chat created successfully",
            "chat_id": chat_id,
            "chat_name": chat_name
        }
    
    except Exception as e:
        response.status_code = 500
        return {"error": f"Failed to create text to speech chat: {str(e)}"}

@app.post("/text-to-speech")
async def text_to_speech(
    request: Request,
    response: Response,
    chat_id: str = Form(...),
    user_id: str = Form(...),
    input_text: str = Form(...),
    input_language: str = Form(...),
    output_language: str = Form(...)
):
    try:
        # Validate languages
        if input_language not in SUPPORTED_TTS_LANGUAGES or output_language not in SUPPORTED_TTS_LANGUAGES:
            response.status_code = 400
            return {"error": f"Unsupported language selection. Supported languages are: {', '.join(SUPPORTED_TTS_LANGUAGES.keys())}"}
        
        # Get language codes
        input_lang_code = SUPPORTED_TTS_LANGUAGES[input_language]
        output_lang_code = SUPPORTED_TTS_LANGUAGES[output_language]
        
        # Translate the text if input and output languages are different
        if input_language != output_language:
            try:
                translated_text = translator.translate(
                    input_text,
                    src=input_lang_code,
                    dest=output_lang_code
                ).text
            except Exception as e:
                response.status_code = 500
                return {"error": f"Translation failed: {str(e)}"}
        else:
            translated_text = input_text
        
        # Convert text to speech
        try:
            # Create a unique filename for the audio file
            unique_id = str(uuid.uuid4())
            audio_file_path = AUDIO_OUTPUT_DIR / f"{unique_id}.mp3"
            
            # Generate speech from text
            tts = gTTS(text=translated_text, lang=output_lang_code, slow=False)
            tts.save(str(audio_file_path))
            
            # Read the file and encode to base64 for direct response
            with open(audio_file_path, "rb") as audio_file:
                audio_content = audio_file.read()
            
            audio_base64 = base64.b64encode(audio_content).decode('utf-8')
            
            # Store the conversion in the chat history
            conversion_entry = {
                "timestamp": datetime.utcnow(),
                "input_text": input_text,
                "input_language": input_language,
                "output_language": output_language,
                "translated_text": translated_text,
                "audio_file_id": unique_id
            }
            
            # Update the chat history
            chat_collection.update_one(
                {"chat_id": chat_id, "user_id": user_id},
                {
                    "$push": {"conversions": conversion_entry},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
            return {
                "message": "Text to speech conversion successful",
                "input_text": input_text,
                "input_language": input_language,
                "translated_text": translated_text,
                "output_language": output_language,
                "audio_content": audio_base64,
                "audio_file_id": unique_id
            }
            
        except Exception as e:
            response.status_code = 500
            return {"error": f"Text to speech conversion failed: {str(e)}"}
            
    except Exception as e:
        response.status_code = 500
        return {"error": f"Text to speech process failed: {str(e)}"}
        
    finally:
        # Clean up the file after sending (we've already encoded it to base64)
        if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
            os.remove(audio_file_path)

@app.get("/audio/{audio_file_id}")
async def get_audio_file(audio_file_id: str):
    """
    Alternative endpoint to directly stream audio files instead of base64 encoding
    May be useful for larger audio files
    """
    audio_file_path = AUDIO_OUTPUT_DIR / f"{audio_file_id}.mp3"
    
    if not os.path.exists(audio_file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=audio_file_path, 
        media_type="audio/mpeg", 
        filename=f"{audio_file_id}.mp3"
    )

@app.get("/get-text-to-speech-history/{user_id}")
async def get_text_to_speech_history(user_id: str, response: Response):
    try:
        # Get all text to speech chats for this user
        chats = list(chat_collection.find(
            {"user_id": user_id, "chat_type": "text_to_speech"},
            {"conversions": 0}  # Exclude the full conversions to keep response size manageable
        ))
        
        # Convert ObjectId to string
        for chat in chats:
            if "_id" in chat:
                chat["_id"] = str(chat["_id"])
        
        return {
            "message": "Text to speech history retrieved successfully",
            "chats": chats
        }
        
    except Exception as e:
        response.status_code = 500
        return {"error": f"Failed to retrieve text to speech history: {str(e)}"}

@app.get("/get-text-to-speech-chat/{chat_id}")
async def get_text_to_speech_chat(chat_id: str, user_id: str, response: Response):
    try:
        # Get detailed chat with conversions
        chat = chat_collection.find_one({"chat_id": chat_id, "user_id": user_id})
        
        if not chat:
            response.status_code = 404
            return {"error": "Chat not found"}
        
        # Convert ObjectId to string
        if "_id" in chat:
            chat["_id"] = str(chat["_id"])
        
        # Process the conversions to include audio base64 content if needed
        if "conversions" in chat:
            for conversion in chat["conversions"]:
                audio_file_id = conversion.get("audio_file_id")
                audio_file_path = AUDIO_OUTPUT_DIR / f"{audio_file_id}.mp3"
                
                # If we want to include the audio content directly (might be large)
                # Alternatively, the frontend can use the /audio/{audio_file_id} endpoint
                if os.path.exists(audio_file_path):
                    with open(audio_file_path, "rb") as audio_file:
                        audio_content = audio_file.read()
                    conversion["audio_content"] = base64.b64encode(audio_content).decode('utf-8')
                else:
                    conversion["audio_content"] = None
        
        return {
            "message": "Text to speech chat retrieved successfully",
            "chat": chat
        }
        
    except Exception as e:
        response.status_code = 500
        return {"error": f"Failed to retrieve text to speech chat: {str(e)}"}