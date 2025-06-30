import os
import sys
import logging
import pandas as pd
import json
import re
import google.generativeai as genai
from google.cloud import translate_v2 as translate
import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Input/Output
INPUT_XLSM = r"C:\Users\SamClissold\MCCF_Git_Repo\MCCF\PDP8\Ninh Binh Gas Model_V10_VN.xlsm"
OUTPUT_FOLDER = "translated_sheets"
DICTIONARY_PATH = "translation_dictionary_gcp.json"

# Gemini Settings
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # Replace with your Gemini API key
TARGET_LANGUAGE = "vi"

# Google Cloud Translation Settings
GOOGLE_CLOUD_PROJECT =  # Your GCP project ID
GOOGLE_APPLICATION_CREDENTIALS =   # Your service account key file

# GPT Settings
openai.api_key = 

# Init clients
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Google Cloud Translation client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
translate_client = translate.Client()

# --- Helpers ---
def is_translatable(value):
    if pd.isnull(value):
        return False
    val = str(value).strip()
    if val == "":
        return False
    try:
        float(val.replace(",", "."))  # skip numbers
        return False
    except ValueError:
        pass
    if re.match(r"^\d{4}-\d{2}-\d{2}", val):  # skip ISO dates
        return False
    return True

def estimate_cost(texts, method="gemini"):
    char_count = sum(len(text) for text in texts)
    
    if method == "google_translate":
        # Google Cloud Translation: $20 per million characters
        cost = (char_count / 1_000_000) * 20
    elif method == "gemini":
        # Gemini pricing: $0.000075 / 1K characters input, $0.0003 / 1K characters output
        estimated_output_chars = char_count * 1.5
        input_cost = (char_count / 1000) * 0.000075
        output_cost = (estimated_output_chars / 1000) * 0.0003
        cost = input_cost + output_cost
    
    logging.info(f"🧮 Est. characters: {char_count:,} | 💰 Est. cost: ${cost:.4f} ({method})")
    return char_count, cost

def translate_with_google_translate(text):
    """Use Google Cloud Translation API - specialized for translation tasks"""
    try:
        result = translate_client.translate(
            text,
            target_language=TARGET_LANGUAGE,
            source_language='en'
        )
        return result['translatedText']
    except Exception as e:
        logging.warning(f"❌ Google Translation failed for '{text}': {e}")
        return text

def translate_with_gemini(text):
    try:
        prompt = f"""Translate the following English text to Vietnamese. 
        If the text contains financial or technical terms, use appropriate Vietnamese terminology.
        Only return the translated text, nothing else.
        
        Text to translate: {text}"""
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.warning(f"❌ Gemini translation failed for '{text}': {e}")
        return text

def translate_with_gpt(text, model="gpt-4", source_lang="vi", target_lang="en"):
    try:
        prompt = f"Translate the following Vietnamese text to English:\n\n{text}"
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logging.warning(f"❌ GPT translation failed for '{text}': {e}")
        return text


# --- Main ---
def main():
    # Check command line arguments for translation method
    if "--google-translate" in sys.argv:
        translation_method = "google_translate"
        logging.info("🧠 Using Google Cloud Translation API (specialized for translation)")
    elif "--gpt" in sys.argv:
        translation_method = "gpt"
        logging.info("🧠 Using GPT-4")
    else:
        translation_method = "gemini"
        logging.info("🧠 Using Google Gemini")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    xls = pd.ExcelFile(INPUT_XLSM, engine="openpyxl")
    all_unique_texts = set()

    logging.info(f"📘 Loaded Excel file with sheets: {xls.sheet_names}")

    sheet_dfs = {}
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name, dtype=str)  # read all as string
        sheet_dfs[sheet_name] = df
        for col in df.columns:
            all_unique_texts.update(
                str(val).strip() for val in df[col].dropna().unique() if is_translatable(val)
            )

    # Estimate cost
    all_unique_texts = sorted(all_unique_texts)
    estimate_cost(all_unique_texts, translation_method)

    # User confirmation
    if len(sys.argv) <= 1 or "--auto" not in sys.argv:
        proceed = input("⚠️ Proceed with translation? (y/n): ").strip().lower()
        if proceed != "y":
            logging.info("❌ Translation cancelled.")
            return

    translation_dict = {}

    for idx, text in enumerate(all_unique_texts):
        if translation_method == "google_translate":
            translation = translate_with_google_translate(text)
        elif translation_method == "gpt":
            translation = translate_with_gpt(text)
        else:
            translation = translate_with_gemini(text)
            
        translation_dict[text] = translation
        logging.info(f"[{idx+1}/{len(all_unique_texts)}] {text} → {translation}")

    # Save dictionary
    with open(DICTIONARY_PATH, "w", encoding="utf-8") as f:
        json.dump(translation_dict, f, ensure_ascii=False, indent=2)

    # Apply and save per sheet
    def apply_translation(val):
        return translation_dict.get(str(val).strip(), val)

    for sheet_name, df in sheet_dfs.items():
        translated_df = df.applymap(lambda x: apply_translation(x) if is_translatable(x) else x)
        out_path = os.path.join(OUTPUT_FOLDER, f"{sheet_name}_translated.csv")
        translated_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logging.info(f"✅ Saved: {out_path}")

    logging.info("🎉 All sheets translated and saved.")


if __name__ == "__main__":
    main()
