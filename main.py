#!/usr/bin/env python3
import sys
print(f"Python version: {sys.version}")  # Verify Python version
import os
import time
import logging
import requests
import torch
from io import BytesIO
from PIL import Image
from flask import Flask, request
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Initialize device with cloud optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fixes lib conflicts on cloud
logger.info(f"Running on: {device.upper()}")

# Load AI model
try:
    logger.info("Loading Moondream model...")
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        trust_remote_code=True,
        token=HUGGINGFACE_TOKEN,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"  # Auto-selects GPU if available
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "vikhyatk/moondream2",
        token=HUGGINGFACE_TOKEN
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise

app = Flask(__name__)

def process_image(image_bytes):
    """Cloud-optimized image processor"""
    try:
        start_time = time.time()
        image = Image.open(image_bytes).convert('RGB')
        
        # Dynamic resizing for cloud efficiency
        max_size = 512 if device == "cuda" else 384
        if max(image.size) > max_size:
            image = image.resize((max_size, max_size))
        
        # Process with timeout
        with torch.inference_mode():
            image_embeds = model.encode_image(image)
            inputs = tokenizer(
                "Analyze this trading chart for Elliott Wave patterns:",
                return_tensors="pt"
            ).to(device)
            
            output = model.generate(
                image_embeds=image_embeds,
                **inputs,
                max_new_tokens=100,
                max_time=8 if device == "cuda" else 15,
                temperature=0.7
            )
            
            caption = tokenizer.decode(output[0], skip_special_tokens=True)
            logger.info(f"Processed in {time.time()-start_time:.2f}s")
            return caption.strip()
            
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return "⚠️ Analysis failed (server-side error)"

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        msg = data.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        
        if not chat_id:
            return {"ok": False}
        
        if "photo" in msg:
            # Get medium quality image (faster download)
            file_id = msg["photo"][1]["file_id"]
            file_info = requests.get(
                f"https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}",
                timeout=5
            ).json()
            
            # Download image
            file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info['result']['file_path']}"
            img_data = requests.get(file_url, timeout=10).content
            
            # Process and respond
            caption = process_image(BytesIO(img_data))
            response = f"📈 *Chart Analysis:*\n{caption}\n\n_Generated on {device.upper()}_"
            
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": response,
                    "parse_mode": "Markdown"
                },
                timeout=5
            )
            
        return {"ok": True}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"ok": False}

@app.route('/')
def home():
    return "✅ Trading Bot Online - Powered by Render"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
