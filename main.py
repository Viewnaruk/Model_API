import uvicorn
from fastapi import FastAPI, HTTPException, Request
import numpy as np
import pickle
import pandas as pd
import re
from scipy.sparse import hstack
import joblib
import emoji
import google.generativeai as genai
from pymongo import MongoClient
from bson import ObjectId
import os
import gdown

# Initialize FastAPI app
app = FastAPI(title="Tourist Reviews API", version="1.0.0")

# MongoDB Connection
try:
    client = MongoClient("mongodb+srv://sasipreyas:1234@cluster0.fwzmzgy.mongodb.net/Web_App_Tourist_Reviews?retryWrites=true&w=majority")
    FASTAPI_URL=("https://your-fastapi-service.onrender.com")
    db = client['Web_App_Tourist_Reviews']
    collection = db['Review']
    print("✅ MongoDB connected successfully!")
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
    raise

# Google Drive file IDs
MODEL_ID = "1pbekIy74RNW4w5dmCMKbvvtia2kp5Chf"
VECTORIZER_ID = "1VBNElPcxYwXuVF7uxH-tq4kSrsOq4lZt"
EMOJI_ID = "1sVSv1GfhPaj2c_WZQklMaYJSx48LY_uR"

# Paths
# os.makedirs("APP", exist_ok=True)
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
EMOJI_PATH = "emoji_mapping.pkl"

# Global variables for models
classifier = None
vectorizer = None
emoji_mapping = None

# Download and load models on startup
@app.on_event("startup")
def startup_event():
    global classifier, vectorizer, emoji_mapping
    print("🌍 Environment:", os.environ)  # ดู environment
    port = int(os.getenv('PORT'))
    print(f"🚀 Running on port {port}")  # แสดง port ที่ใช้
    try:
        # download_if_missing(MODEL_ID, MODEL_PATH)
        download_if_missing(VECTORIZER_ID, VECTORIZER_PATH)
        download_if_missing(EMOJI_ID, EMOJI_PATH)
        classifier = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        emoji_mapping = joblib.load(EMOJI_PATH)
        print("🎯 Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {e}")

# Utility function to download files
def download_if_missing(file_id, save_path):
    if not os.path.exists(save_path):
        print(f"📥 Downloading {save_path} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, save_path, quiet=False)
    else:
        print(f"✅ Found {save_path}")

# Regex for emoji extraction
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # Emoticons
    u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    u"\U0001F680-\U0001F6FF"  # Transport & Map
    u"\U0001F1E0-\U0001F1FF"  # Flags
    u"\U00002700-\U000027BF"  # Dingbats
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols & Pictographs
    u"\U0001FA70-\U0001FAFF"  # Symbols & Pictographs Extended-A
    u"\U00002600-\U000026FF"  # Miscellaneous Symbols
    u"\U00002300-\U000023FF"  # Miscellaneous Technical
    u"\U0000FE00-\U0000FE0F"  # Variation Selectors
    u"\U0001F1F2-\U0001F1F4"  # Macau flag etc.
    u"\U0001F1E6-\U0001F1FF"  # Regional Indicator Symbols
    "]", flags=re.UNICODE)

def extract_emoji(text: str):
    emojis = emoji_pattern.findall(text)
    clean_text = emoji_pattern.sub(r'', text)
    return emojis, clean_text

def strip_aspect(aspect: str):
    return aspect.strip() if aspect else "Other"

@app.post('/predict')
async def predict_reviews(request: Request):
    try:
        if classifier is None or vectorizer is None or emoji_mapping is None:
            print("❌ Models not loaded")
            raise HTTPException(status_code=500, detail="Models not loaded")

        body = await request.json()
        print("📥 Received request body:", body)
        review = body.get("review")
        category = body.get("category")

        if not review:
            print("❌ Missing review")
            raise HTTPException(status_code=400, detail="review is required")
        if not category:
            print("❌ Missing category")
            raise HTTPException(status_code=400, detail="category is required")

        emojis, clean_review = extract_emoji(review)
        print(f"📝 Processed review: {clean_review}, Emojis: {emojis}")

        X_text = vectorizer.transform([clean_review]).toarray()
        print("🔢 Text vectorized")

        emoji_label = sum([emoji_mapping.get(e, 0) for e in emojis])
        emoji_label_array = np.array([emoji_label]).reshape(-1, 1)
        print(f"🌐 Emoji label: {emoji_label}")

        X_final = hstack([X_text, emoji_label_array]).toarray()
        print("✅ Features combined")

        score = classifier.decision_function(X_final)[0]
        print("📊 Score:", score)

        threshold = -0.5456117703308974
        sentiment = "Positive" if score > threshold else "Negative"
        print(f"😊 Sentiment: {sentiment}")

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyBHCXD9hQhtNhWnMp1dkd_v9AvdLHD0GGk"))
        model = genai.GenerativeModel("gemma-3-27b-it")
        print("🤖 Generative AI configured")

        if category == "Religious Place":
            prompt = f"""Analyze the following review text: '{review}'. Your task is to classify the single most prominent aspect discussed in the text. You must respond with only one word, chosen from this exact list of categories: Aesthetics, Scenery, Atmosphere, Spirituality, Location. If the review content does not clearly and strongly align with any of these five options, respond with Other."""
        # (โค้ด prompt อื่น ๆ ตามเดิม)

        print(f"📝 Prompt: {prompt}")
        response = model.generate_content(prompt)
        aspect_stripped = strip_aspect(response.text)
        print(f"🌟 Aspect: {aspect_stripped}")

        return {
            "review": review,
            "sentiment": sentiment,
            "score": float(score),
            "emojis": emojis,
            "emoji_label": emoji_label,
            "Aspect": aspect_stripped
        }

    except genai.types.generation_types.StopReason as e:
        print(f"❌ Generation stopped: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Generation stopped: {str(e)}")
    except genai.types.GenerateContentResponse as e:  # จับ response error
        print(f"❌ Generation response error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get('/health')
def health():
    print("🩺 Health check called on port", os.getenv('PORT', 'unknown'))
    return {'status': 'healthy'}

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8080)