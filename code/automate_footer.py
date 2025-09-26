import json
import re
import time
import argparse
from openai import OpenAI
import os

# Colab environment - no need for dotenv
# Set your environment variables in a cell before running this script:
# import os
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
# os.environ["FOOTER_FILE_PATH"] = "/content/footer.json"

# --- 1. CORE UTILITY FUNCTIONS ---

def prompt_gpt(prompt):
    """Pure GPT prompting function - just send and get response"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        # Clean up response - remove markdown code blocks and quotes
        content = response.choices[0].message.content
        content = content.replace('"',"")
        # Remove markdown code blocks
        content = re.sub(r'```html\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        return content.strip()
    except Exception as e:
        print(f"GPT error: {e}")
        return None

def translate_text(text, target_language):
    """Simple translation function - returns only translated text"""
    prompt = f"Translate to {target_language}. Return only the translation, no explanations , IF THE THE Input text has HTML tags like <br> or <p> or any keep them and translate the text and return if no html return just the text : {text}"
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        # Clean up response - remove markdown code blocks and quotes
        content = response.choices[0].message.content
        content = content.replace('"',"")
        # Remove markdown code blocks
        content = re.sub(r'```html\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        return content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def replace_in_file(json_path, placeholder, replacement_content):
    """Replace placeholder with content in any file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()

        updated_content = content.replace(placeholder, replacement_content)

        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        print(f"Replaced {placeholder} in {json_path}")
        return True

    except Exception as e:
        print(f"Error updating {json_path}: {e}")
        return False

def prompt_gpt_html_validated(prompt, expected_tags, max_retries=2):
    """Prompts GPT for an HTML response, validates its structure, and retries if needed."""
    for attempt in range(max_retries + 1):
        print(f"Prompting GPT for HTML (Attempt {attempt + 1})...")
        response = prompt_gpt(prompt)
        is_valid = all(f"<{tag}>" in response and f"</{tag}>" in response for tag in expected_tags)
        if is_valid:
            print("HTML Validation PASSED.")
            return response
        else:
            print("HTML Validation FAILED. Retrying...")
            prompt += "\n\nCRITICAL: The previous response was invalid. Please ensure the output is a valid HTML string and contains the required tags."
            time.sleep(1)
    print("Max retries reached. Returning last invalid response.")
    return ""

# --- 2. PROMPT GENERATION FUNCTIONS ---

def generate_brand_slogan_prompt(original_slogan, brand_name, language):
    return f"The original slogan is: '{original_slogan}'. Generate a new, similar brand slogan for the brand '{brand_name}'. It should be inspiring and concise. Language: {language}. IMPORTANT: Return ONLY the HTML code without any markdown formatting or code blocks. The response must be a valid HTML string with `<p>`, `<strong>`, and `<a>` tags, ending with a call to action like 'Shop now'."

def generate_trust_badge_prompt(original_title, original_text, brand_name, language):
    return f"The original trust badge is '{original_title}' with text '{original_text}'. Generate a new, similar trust badge title and a one-sentence description for the brand '{brand_name}'. The tone should be reassuring and professional. Language: {language}. The response should be in two parts separated by a pipe '|': TITLE|DESCRIPTION. The title must be wrapped in `<strong>` and the description in `<p>`."

def generate_newsletter_headline_prompt(original_headline, brand_name, language):
    return f"The original newsletter headline is: '{original_headline}'. Generate a new, compelling headline for a newsletter signup for the brand '{brand_name}'. It should offer a clear incentive. Language: {language}. Return only the text."

# --- 3. MAIN EXECUTION SCRIPT ---

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate and generate footer content')
    parser.add_argument('--language', '-l', required=True, help='Target language (e.g., fr, es, de)')
    parser.add_argument('--brand', '-b', required=True, help='Brand name (e.g., GlamCurl)')
    
    args = parser.parse_args()
    
    # Configuration from arguments and environment variables
    FOOTER_JSON_PATH = os.getenv("FOOTER_FILE_PATH")
    language = args.language
    brand_name = args.brand

    # Check if required environment variables are available
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it: export OPENAI_API_KEY='your-key-here'")
        return
        
    if not FOOTER_JSON_PATH:
        print("ERROR: FOOTER_FILE_PATH environment variable not set!")
        print("Please set it: export FOOTER_FILE_PATH='/path/to/your/footer.json'")
        return

    print(f"Starting translation for brand: {brand_name}, language: {language}")
    print(f"Using file: {FOOTER_JSON_PATH}")

    # =================================================================
    # =============== 1. TRANSLATION TASKS (Static UI) ================
    # =================================================================
    print("\n--- Starting Footer Translation Tasks ---\n")

    translated = translate_text("Information", language)
    replace_in_file(FOOTER_JSON_PATH, "NEW_FOOTER_H8I9_TRANSLATED", translated)

    translated = translate_text("Butik", language)
    replace_in_file(FOOTER_JSON_PATH, "NEW_FOOTER_J0K1_TRANSLATED", translated)
    
    translated = translate_text("Behöver hjälp?", language)
    replace_in_file(FOOTER_JSON_PATH, "NEW_FOOTER_L2M3_TRANSLATED", translated)
    
    translated = translate_text("<p>💬 Tillgänglig chatt måndag – fredag, 9:00 till 18:00</p>", language)
    replace_in_file(FOOTER_JSON_PATH, "NEW_FOOTER_N4P5_TRANSLATED", translated)

    translated = translate_text("Subscribe to our emails", language)
    replace_in_file(FOOTER_JSON_PATH, "NEW_FOOTER_Q6R7_TRANSLATED", translated)
    
    translated = translate_text("Vi lovar att inte använda din e-post för spam! du kan när som helst avregistrera dig.", language)
    replace_in_file(FOOTER_JSON_PATH, "NEW_NEWSLETTER_S8T9_TRANSLATED", translated)
    
    translated = translate_text("- Theme by <a href=\"https://lumintheme.com/\" target=\"_blank\" title=\"https://lumintheme.com/\">Lumin<em>Theme</em></a> © 2024", language)
    replace_in_file(FOOTER_JSON_PATH, "NEW_FOOTER_U0V1_TRANSLATED", translated)

    # =================================================================
    # =========== 2. GENERATION TASKS (Brand-specific) ==============
    # =================================================================
    print("\n--- Starting Footer Generation Tasks ---\n")

    # Scrolling Text Slogans
    prompt = generate_brand_slogan_prompt("<p>Skönheten arbetar till din fördel.<strong> — </strong><a href=\"/pages/vara-produkter\" title=\"Våra produkter\"><strong>Köp nu</strong></a></p>", brand_name, language)
    result = prompt_gpt_html_validated(prompt, expected_tags=['p', 'strong', 'a'])
    replace_in_file(FOOTER_JSON_PATH, "NEW_SCROLLING_TEXT_W2X3_GENERATED", result)
    replace_in_file(FOOTER_JSON_PATH, "NEW_SCROLLING_TEXT_Y4Z5_GENERATED", result)

    # Trust Badges
    trust_badges = [
        {"title": "<strong>100 % TILLFREDSHET</strong>", "text": "<p>Tusentals nöjda kunder, prova utan risk.</p>", "placeholders": ("NEW_TRUST_BADGES_A6B7_GENERATED", "NEW_TRUST_BADGES_C8D9_GENERATED")},
        {"title": "<strong>ENKEL OCH SNABB BYTNING</strong>", "text": "<p>Förenklad retur utan besvär.</p>", "placeholders": ("NEW_TRUST_BADGES_E0F1_GENERATED", "NEW_TRUST_BADGES_G2H3_GENERATED")},
        {"title": "<strong>EXPRESSLEVERANS</strong>", "text": "<p>Snabb leverans till din adress.</p>", "placeholders": ("NEW_TRUST_BADGES_I4J5_GENERATED", "NEW_TRUST_BADGES_K6L7_GENERATED")},
        {"title": "<strong>SÄKER BETALNING</strong>", "text": "<p>Dina bankuppgifter är krypterade och skyddade...</p>", "placeholders": ("NEW_TRUST_BADGES_M8N9_GENERATED", "NEW_TRUST_BADGES_O0P1_GENERATED")}
    ]

    for badge in trust_badges:
        prompt = generate_trust_badge_prompt(badge["title"], badge["text"], brand_name, language)
        result = prompt_gpt(prompt)
        if result and '|' in result:
            title, desc = result.split('|', 1)  # Split only on first |
            replace_in_file(FOOTER_JSON_PATH, badge["placeholders"][0], title.strip())
            replace_in_file(FOOTER_JSON_PATH, badge["placeholders"][1], desc.strip())
        else:
            print(f"Warning: Invalid trust badge response: {result}")

    # Newsletter Headline
    prompt = generate_newsletter_headline_prompt("Profiter av 10 % rabatt på din första beställning nu !", brand_name, language)
    result = prompt_gpt(prompt)
    if result:
        replace_in_file(FOOTER_JSON_PATH, "NEW_NEWSLETTER_Q2R3_GENERATED", result)

    print(f"\n--- Completed! Check {FOOTER_JSON_PATH} for results ---")

if __name__ == "__main__":
    main()