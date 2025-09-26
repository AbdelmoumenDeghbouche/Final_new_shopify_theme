import json
import re
import time
import argparse
from openai import OpenAI
import os

def replace_in_file(json_path, placeholder, replacement_content):
    """Replaces a placeholder string in a file with new text."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()

        updated_content = content.replace(placeholder, replacement_content)

        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        print(f"Replaced {placeholder}")
        return True

    except Exception as e:
        print(f"Error updating {json_path}: {e}")
        return False

def translate_text(text, target_language):
    """Translates a string of text, preserving any HTML tags."""
    prompt = f"Translate to {target_language}. Return only the translation, no explanations. IF the input text has HTML tags like <br> or <p> or any, keep them and translate the text content only. If no HTML, return just the translated text: {text}"
    
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
        content = response.choices[0].message.content
        content = content.replace('"',"")
        content = re.sub(r'```html\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        return content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def prompt_gpt(prompt):
    """Sends a prompt to a GPT model and returns the text response."""
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
        content = response.choices[0].message.content
        content = re.sub(r'```html\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        return content.strip()
    except Exception as e:
        print(f"GPT error: {e}")
        return None

def validate_html_tags(response, expected_tags):
    """Improved HTML validation with better debugging"""
    print(f"Validating HTML: {response}")
    print(f"Expected tags: {expected_tags}")
    
    for tag in expected_tags:
        # For opening tags, we need to be more flexible to allow attributes
        # Look for <tag> or <tag with attributes>
        opening_pattern = f"<{tag}(?:\\s|>)"
        closing_tag = f"</{tag}>"
        
        import re
        has_opening = bool(re.search(opening_pattern, response))
        has_closing = closing_tag in response
        
        print(f"Tag '{tag}': opening={has_opening}, closing={has_closing}")
        
        if not (has_opening and has_closing):
            return False
    return True

def prompt_gpt_html_validated(prompt, expected_tags, max_retries=2):
    """Prompts GPT for an HTML response, validates its structure, and retries if needed."""
    for attempt in range(max_retries + 1):
        print(f"Prompting GPT for HTML (Attempt {attempt + 1})...")
        response = prompt_gpt(prompt)
        print("HTML RESPONSE OF GPT:", response)
        
        if not response:
            print("No response from GPT, retrying...")
            continue
            
        is_valid = validate_html_tags(response, expected_tags)
        if is_valid:
            print("HTML Validation PASSED.")
            return response
        else:
            print("HTML Validation FAILED. Retrying...")
            prompt += "\n\nCRITICAL: The previous response was invalid. Please ensure the output is a valid HTML string and contains the required tags with proper opening and closing tags. Use SINGLE QUOTES for all HTML attributes (href='...' title='...') not double quotes."
            time.sleep(1)
    
    print("Max retries reached. Returning last response anyway.")
    return response if 'response' in locals() else ""

# --- 2. PROMPT GENERATION FUNCTIONS ---

def generate_product_tagline_prompt(original_text, product_title, product_description, language):
    return f"The original tagline is '{original_text}'. Generate a new, short, and benefit-driven tagline for a product named '{product_title}' with this description: '{product_description}'. Language: {language}. Return only the text."

def generate_product_feature_prompt(original_feature, product_title, product_description, language):
    return f"The original feature is: '{original_feature}'. Write a new, concise feature bullet point (under 8 words) for '{product_title}' (description: '{product_description}'). The new feature should convey a similar benefit. Language: {language}. The response must be valid HTML wrapped in `<p>` tags. Use SINGLE QUOTES for any HTML attributes."

def generate_quantity_option_prompt(original_label, product_title, language):
    return f"The original purchase option is '{original_label}'. Create a new, similar purchase option label for the product '{product_title}'. It should be short and clear. Language: {language}. Return only the text."
    
def generate_badge_text_prompt(original_badge, product_title, language):
    return f"The original promotional badge text is '{original_badge}'. Generate new, exciting badge text for a special offer on '{product_title}'. Keep it short and urgent. Language: {language}. Return only the text."

def generate_unit_label_prompt(original_label, product_title, language):
     return f"The original unit label is '{original_label}'. Create a new, descriptive unit label for a product bundle of '{product_title}'. Mention the supply duration and shipping. Language: {language}. Return only the text."

def generate_save_text_prompt(original_text, product_title, language):
     return f"The original savings text is '{original_text}'. Create a new, compelling text that highlights the savings for a '{product_title}' bundle. Language: {language}. Return only the text."

def generate_promo_html_prompt(original_promo, product_title, language):
    return f"The original promo text is '{original_promo}'. Create a new, similar promotional message for '{product_title}'. Language: {language}. The response must be a valid HTML string with `<strong>` tags. Use SINGLE QUOTES for any HTML attributes."

def generate_bundle_title_prompt(product_title, language):
    return f"Generate a headline for a 'Bundle & Save' section for the product '{product_title}'. Language: {language}. The response must be valid HTML wrapped in `<h3>` tags. Use SINGLE QUOTES for any HTML attributes."

def generate_social_proof_prompt(original_text, product_title, product_description, language):
    return f"The original social proof text is: '{original_text}'. Write a new, similar message for '{product_title}' (description: '{product_description}') that builds trust by mentioning experts or a large number of users. Language: {language}. The response must be valid HTML with `<p>` and `<strong>` tags. Use SINGLE QUOTES for any HTML attributes."

def generate_collapsible_content_prompt(original_content, product_title, product_description, language):
    return f"The original content is: '{original_content}'. Generate a new, detailed description for a '{product_title}' (description: '{product_description}') that is suitable for a 'Product Details' or 'Benefits' tab. The style should be informative. Language: {language}. The response must be valid HTML, using `<p>` for paragraphs and `<ul><li><strong>` for lists where appropriate. Use SINGLE QUOTES for any HTML attributes."

def generate_guarantee_prompt(original_heading, original_text, product_title, language):
    return f"The original guarantee is '{original_heading}' with text '{original_text}'. Generate a new, similar money-back guarantee title and a one-sentence description for '{product_title}'. The tone should be reassuring. Language: {language}. The response should be in two parts separated by a pipe '|': TITLE|DESCRIPTION. The description must be wrapped in `<p>` tags. Use SINGLE QUOTES for any HTML attributes."

def generate_full_review_prompt(original_review_head, original_review_text, product_title, product_description, language):
    return f"The original customer review has the headline '{original_review_head}' and body '{original_review_text}'. Generate a new, realistic customer review for '{product_title}' (description: '{product_description}'). The review should have a title and a body. Language: {language}. The response should be in two parts separated by a pipe '|': TITLE|BODY. The body should be plain text."

def generate_comparison_benefit_prompt(original_benefit, product_title, language):
    return f"The original benefit in a comparison table is '{original_benefit}'. Generate a new, similar but unique benefit for the product '{product_title}'. Keep it short and clear. Language: {language}. Return only the text."

def generate_marketing_hashtag_prompt(product_title, brand_name, language):
    return f"Generate a marketing hashtag or slogan for the product '{product_title}' by brand '{brand_name}'. Language: {language}. Return only the text."

def generate_feature_list_item_prompt(original_title, product_title, product_description, language):
    return f"The original feature title is '{original_title}'. Generate a new feature title and description for '{product_title}' (description: '{product_description}'). Language: {language}. The response should be in two parts separated by a pipe '|': TITLE|DESCRIPTION. Use HTML tags where appropriate and SINGLE QUOTES for any HTML attributes."

def generate_faq_prompt(original_question, original_answer, product_title, product_description, language):
    return f"The original FAQ is question: '{original_question}' with answer: '{original_answer}'. Generate a new, similar FAQ for '{product_title}' (description: '{product_description}'). Language: {language}. The response should be in two parts separated by a pipe '|': QUESTION|ANSWER. The answer should be wrapped in `<p>` tags and use SINGLE QUOTES for any HTML attributes."

# --- 3. MAIN EXECUTION SCRIPT ---

def main():
    parser = argparse.ArgumentParser(description='Translate and generate product page content')
    parser.add_argument('--language', '-l', required=True, help='Target language (e.g., fr, es, de)')
    parser.add_argument('--brand', '-b', required=True, help='Brand name (e.g., GlamCurl)')
    parser.add_argument('--product', '-p', required=True, help='Product title')
    parser.add_argument('--description', '-d', required=True, help='Product description')
    
    args = parser.parse_args()
    
    PRODUCT_JSON_PATH = os.getenv("PRODUCT_FILE_PATH")
    language = args.language
    brand_name = args.brand
    product_title = args.product
    product_description = args.description

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        return
        
    if not PRODUCT_JSON_PATH:
        print("ERROR: PRODUCT_FILE_PATH environment variable not set!")
        return

    print(f"Starting product page generation for: {product_title}")
    print(f"Brand: {brand_name}, Language: {language}")

    print("\n--- Starting Product Page Translation Tasks ---\n")
    
    tasks_to_translate = {
        "NEW_BREADCRUMBS_A1B2_TRANSLATED": "Home",
        "NEW_SHIPPING_C3D4_TRANSLATED": "Beställ inom <strong>(timer)</strong>  och få leveransen före <strong>(date). </strong>",
        "NEW_COLLAPSIBLE_TAB_E5F6_TRANSLATED": "Produktdetaljer",
        "NEW_COLLAPSIBLE_TAB_G7H8_TRANSLATED": "Fördelar och nytta",
        "NEW_RICH_TEXT_I9J0_TRANSLATED": "Sett i pressen",
        "NEW_SHOPPABLE_VIDEO_K1L2_TRANSLATED": "NY",
        "NEW_FAQ_M3N4_TRANSLATED": "FAQ",
        "NEW_REVIEW_O5P6_TRANSLATED": "Kundrecensioner",
        "NEW_REVIEW_Q7R8_TRANSLATED": "Verifierat köp",
        "NEW_REVIEW_S9T0_TRANSLATED": "Var detta till hjälp för dig?",
        "NEW_RELATED_PRODUCTS_U1V2_TRANSLATED": "You may also like"
    }
    for placeholder, text in tasks_to_translate.items():
        translated_text = translate_text(text, language)
        replace_in_file(PRODUCT_JSON_PATH, placeholder, translated_text)
        
    print("\n--- Starting Product Page Generation Tasks ---\n")

    # --- Main Product Block ---
    prompt = generate_product_tagline_prompt("Investera i ett stylingverktyg som kan göra allt !", product_title, product_description, language)
    result = prompt_gpt(prompt)
    replace_in_file(PRODUCT_JSON_PATH, "NEW_TEXT_W3X4_GENERATED", result)

    feature_tasks = {
        "NEW_FEATURE_Y5Z6_GENERATED": "<p>Lockar utan friss som håller hela dagen</p>",
        "NEW_FEATURE_A7B8_GENERATED": "<p>Inget skador från extrem värme</p>",
        "NEW_FEATURE_C9D0_GENERATED": "<p>Perfekt för voluminösa lockar</p>"
    }
    for placeholder, original_text in feature_tasks.items():
        prompt = generate_product_feature_prompt(original_text, product_title, product_description, language)
        result = prompt_gpt_html_validated(prompt, expected_tags=['p'])
        replace_in_file(PRODUCT_JSON_PATH, placeholder, result)

    quantity_tasks = {
        "NEW_QUANTITY_E1F2_GENERATED": ("1 Pack", generate_quantity_option_prompt),
        "NEW_QUANTITY_G3H4_GENERATED": ("Buy 2, Get 1 FREE", generate_quantity_option_prompt),
        "NEW_QUANTITY_I5J6_GENERATED": ("Buy 3, Get 2 FREE", generate_quantity_option_prompt),
        "NEW_QUANTITY_K7L8_GENERATED": ("SPECIAL OFFER - Limited Time", generate_badge_text_prompt),
        "NEW_QUANTITY_M9N0_GENERATED": ("FLASH SALE — Grab It Before It's Gone", generate_badge_text_prompt),
        "NEW_QUANTITY_O1P2_GENERATED": ("6 Months Supply | FREE Shipping", generate_unit_label_prompt),
        "NEW_QUANTITY_Q3R4_GENERATED": ("12 Months Supply | FREE Shipping", generate_unit_label_prompt),
        "NEW_QUANTITY_S5T6_GENERATED": ("Save (%) - GET ONE FREE", generate_save_text_prompt),
        "NEW_QUANTITY_U7V8_GENERATED": ("Save (%) - GET TWO FREE", generate_save_text_prompt)
    }
    for placeholder, (original_text, prompt_func) in quantity_tasks.items():
        prompt = prompt_func(original_text, product_title, language)
        result = prompt_gpt(prompt)
        replace_in_file(PRODUCT_JSON_PATH, placeholder, result)

    prompt = generate_promo_html_prompt("<strong>12</strong> Months Supply | <strong>FREE Shipping</strong>", product_title, language)
    result = prompt_gpt_html_validated(prompt, expected_tags=['strong'])
    replace_in_file(PRODUCT_JSON_PATH, "NEW_QUANTITY_W9X0_GENERATED", result)
    
    prompt = generate_bundle_title_prompt(product_title, language)
    result = prompt_gpt_html_validated(prompt, expected_tags=['h3'])
    replace_in_file(PRODUCT_JSON_PATH, "NEW_QUANTITY_Y1Z2_GENERATED", result)

    prompt = generate_social_proof_prompt("<p><strong>50+ certifierade</strong> hudläkare rekommenderar detta till sina patienter.</p>", product_title, product_description, language)
    result = prompt_gpt_html_validated(prompt, expected_tags=['p', 'strong'])
    replace_in_file(PRODUCT_JSON_PATH, "NEW_TESTIMONIAL_IMAGES_A3B4_GENERATED", result)

    collapsible_tasks = {
        "NEW_COLLAPSIBLE_TAB_E7F8_GENERATED": "<p>Denna multifunktionella locktång har snabb uppvärmning...</p>",
        "NEW_COLLAPSIBLE_TAB_G9H0_GENERATED": "<ul><li><strong>Perfekta lockar på ett ögonblick</strong>...</li></ul>",
    }
    for placeholder, original_text in collapsible_tasks.items():
        prompt = generate_collapsible_content_prompt(original_text, product_title, product_description, language)
        result = prompt_gpt(prompt)
        replace_in_file(PRODUCT_JSON_PATH, placeholder, result)
        
    prompt = generate_guarantee_prompt("30 dagars pengar tillbaka garanti", "<p>Om du inte är nöjd med din upplevelse...</p>", product_title, language)
    result = prompt_gpt(prompt)
    if result and '|' in result:
        title, desc = result.split('|', 1)
        replace_in_file(PRODUCT_JSON_PATH, "NEW_COLLAPSIBLE_TAB_C5D6_GENERATED", title.strip())
        replace_in_file(PRODUCT_JSON_PATH, "NEW_COLLAPSIBLE_TAB_I1J2_GENERATED", desc.strip())

    # --- Other Sections ---
    prompt = generate_product_tagline_prompt("<strong>Se locktång i aktion !</strong>", product_title, product_description, language)
    result = prompt_gpt_html_validated(prompt, expected_tags=['strong'])
    replace_in_file(PRODUCT_JSON_PATH, "NEW_SHOPPABLE_VIDEO_K3L4_GENERATED", result)

    prompt = generate_product_tagline_prompt("<strong>Sublima Lockar med locktång</strong>", product_title, product_description, language)
    result = prompt_gpt_html_validated(prompt, expected_tags=['strong'])
    replace_in_file(PRODUCT_JSON_PATH, "NEW_LUMIN_MEGA_M5N6_GENERATED", result)

    prompt = generate_collapsible_content_prompt("<p>Tänk dig att varje morgon börja med perfekta lockar...</p>", product_title, product_description, language)
    result = prompt_gpt(prompt)
    replace_in_file(PRODUCT_JSON_PATH, "NEW_LUMIN_MEGA_O7P8_GENERATED", result)
    
    lumin_slider_tasks = {
        "NEW_LUMIN_SLIDER_Q9R0_GENERATED": ("Se hur andra älskar sin Locktång !", generate_product_tagline_prompt),
        "NEW_LUMIN_SLIDER_S1T2_GENERATED": ("Äkta recensioner från riktiga människor", generate_product_tagline_prompt),
        "NEW_LUMIN_SLIDER_U3V4_GENERATED": ("Köp din Locktång idag", generate_product_tagline_prompt),
        "NEW_LUMIN_SLIDER_W5X6_GENERATED": ("Betygsatt 4,8/5 av 14 428+ glada kunder", generate_social_proof_prompt)
    }
    for placeholder, (original_text, prompt_func) in lumin_slider_tasks.items():
        if prompt_func == generate_social_proof_prompt:
            prompt = prompt_func(original_text, product_title, product_description, language)
        else:
            prompt = prompt_func(original_text, product_title, product_description, language)
        result = prompt_gpt(prompt)
        replace_in_file(PRODUCT_JSON_PATH, placeholder, result)
        
    prompt = generate_product_tagline_prompt("Kvalitet som du inte hittar någon annanstans !", product_title, product_description, language)
    result = prompt_gpt(prompt)
    replace_in_file(PRODUCT_JSON_PATH, "NEW_COMPARISON_TABLE_Y7Z8_GENERATED", result)
    
    prompt = generate_product_tagline_prompt("Ledande konkurrent", product_title, product_description, language)
    result = prompt_gpt(prompt)
    replace_in_file(PRODUCT_JSON_PATH, "NEW_COMPARISON_TABLE_A9B0_GENERATED", result)

    comparison_benefits = {
        "NEW_COMPARISON_TABLE_C1D2_GENERATED": "Säkerhet (Anti-bränning)",
        "NEW_COMPARISON_TABLE_C1D2_GENERATED_1": "Användarvänlighet (Lätt, trådlös)",
        "NEW_COMPARISON_TABLE_C1D2_GENERATED_2": "Förberedelse & Autonomi (Snabb uppvärmning)",
        "NEW_COMPARISON_TABLE_C1D2_GENERATED_3": "Mångsidighet (Temperaturinställning)",
        "NEW_COMPARISON_TABLE_C1D2_GENERATED_4": "Ekonomi (Ingen frisörbesök)",
        "NEW_COMPARISON_TABLE_C1D2_GENERATED_5": "Hållbarhet (Lockar som håller)"
    }
    for placeholder, original_text in comparison_benefits.items():
        prompt = generate_comparison_benefit_prompt(original_text, product_title, language)
        result = prompt_gpt(prompt)
        replace_in_file(PRODUCT_JSON_PATH, placeholder, result)

    prompt = generate_product_tagline_prompt("Varje morgon, perfekta lockar", product_title, product_description, language)
    result = prompt_gpt(prompt)
    replace_in_file(PRODUCT_JSON_PATH, "NEW_COLLAGE_E3F4_GENERATED", result)

    prompt = generate_marketing_hashtag_prompt(product_title, brand_name, language)
    result = prompt_gpt(prompt)
    replace_in_file(PRODUCT_JSON_PATH, "NEW_RICH_TEXT_G5H6_GENERATED", result)
    
    prompt = generate_collapsible_content_prompt("<p>Vi stödjer ett medvetet och ansvarsfullt förhållningssätt...</p>", product_title, product_description, language)
    result = prompt_gpt(prompt)
    replace_in_file(PRODUCT_JSON_PATH, "NEW_RICH_TEXT_I7J8_GENERATED", result)

    prompt = generate_product_tagline_prompt("Varför locktång?", product_title, product_description, language)
    result = prompt_gpt(prompt)
    replace_in_file(PRODUCT_JSON_PATH, "NEW_RICH_TEXT_K9L0_GENERATED", result)
    
    lumin_mega_features = {
        ("NEW_LUMIN_MEGA_FEATURE_1_GENERATED_TITLE", "NEW_LUMIN_MEGA_FEATURE_1_GENERATED_TEXT"): ("6 Temperaturinställningar", "<p>Justera temperaturen för perfekta lockar...</p>"),
        ("NEW_LUMIN_MEGA_FEATURE_2_GENERATED_TITLE", "NEW_LUMIN_MEGA_FEATURE_2_GENERATED_TEXT"): ("<strong>Justerbar Timer</strong>", "<p>Skapa lösa eller täta lockar...</p>"),
        ("NEW_LUMIN_MEGA_FEATURE_3_GENERATED_TITLE", "NEW_LUMIN_MEGA_FEATURE_3_GENERATED_TEXT"): ("<strong>Snabb Laddning</strong>", "<p>Ladda på bara 4 timmar...</p>"),
        ("NEW_LUMIN_MEGA_FEATURE_4_GENERATED_TITLE", "NEW_LUMIN_MEGA_FEATURE_4_GENERATED_TEXT"): ("<strong>Lång Batteritid</strong>", "<p>Upp till 50 minuters trådlös användning...</p>")
    }
    for (title_ph, text_ph), (original_title, original_text) in lumin_mega_features.items():
        prompt = generate_feature_list_item_prompt(original_title, product_title, product_description, language)
        result = prompt_gpt(prompt)
        if result and '|' in result:
            title, desc = result.split('|', 1)
            replace_in_file(PRODUCT_JSON_PATH, title_ph, title.strip())
            replace_in_file(PRODUCT_JSON_PATH, text_ph, desc.strip())
        
    prompt = generate_product_tagline_prompt("Skäm bort ditt hår med mjukhet", product_title, product_description, language)
    result = prompt_gpt(prompt)
    replace_in_file(PRODUCT_JSON_PATH, "NEW_VIDEO_WITH_TEXT_M1N2_GENERATED", result)

    # FAQs
    faqs = [
        ("Är denna locktång lämplig för alla hårtyper?", "<p>Absolut! Tack vare sina flera temperaturinställningar..."),
        ("Kan man justera temperaturen?", "<p>Absolut! Den här locktång erbjuder <strong>sex..."),
        ("Är locktång farlig för håret?", "<p>Nej, tack vare den <strong>keramiska turmalinen</strong>..."),
        ("Fungerar det med kort hår?", "<p>Ja, så länge locken är tillräckligt lång för att..."),
        ("Hur man rengör järnet?", "<p>För att rengöra locktång, stäng av den och..."),
        ("Är locktång transportabel och lätt att ta med?", "<p>Ja, denna locktång är <strong>transportabel</strong>..."),
        ("Är materialen i locktången verkligen ekologiska?", "<p>Ja, vår locktång är tillverkad av hållbara och..."),
        ("Hur fungerar frakt och returer?", "<p>Vi erbjuder snabb och pålitlig frakt...")
    ]
    faq_placeholders = [("NEW_FAQ_{}_GENERATED_QUESTION", "NEW_FAQ_{}_GENERATED_ANSWER") for i in range(1, 9)]
    
    for i, (q_ph, a_ph) in enumerate(faq_placeholders):
        original_q, original_a = faqs[i]
        prompt = generate_faq_prompt(original_q, original_a, product_title, product_description, language)
        result = prompt_gpt(prompt)
        if result and '|' in result:
            question, answer = result.split('|', 1)
            replace_in_file(PRODUCT_JSON_PATH, q_ph.format(i+1), question.strip())
            replace_in_file(PRODUCT_JSON_PATH, a_ph.format(i+1), answer.strip())

    # Reviews
    prompt = generate_social_proof_prompt("<strong>14428+</strong> Verkliga recensioner, verkliga resultat från <strong>människor som du.</strong>", product_title, product_description, language)
    result = prompt_gpt_html_validated(prompt, expected_tags=['strong'])
    replace_in_file(PRODUCT_JSON_PATH, "NEW_REVIEW_O3P4_GENERATED", result)

    reviews = [
        ("Val av temperatur för att inte bränna håret", "Locktång fyller perfekt sin mission..."),
        ("IMPRESIONANTE !", "Jag har rakt hår och min dröm är att ha lockigt..."),
        ("Pålitlig locktång med enkel hantering", "Jag har använt den automatiska locktången ett tag nu..."),
        ("Fantastisk produkt, bra kvalitet", "Jag är supernöjd med denna sladdlösa locktång!..."),
        ("Ottimo… lika effektiv som de stora märkena", "Presterar bra… systemet är väldigt enkelt..."),
        ("Lätt att använda", "Fyller sin funktion, en lätt apparat som är enkel..."),
        ("Super locktång", "Liten och smidig – perfekt hemma och på resa..."),
        ("🎁 Perfekt present… och billig", "Jag behövde en liten present till min tonåriga systerdotter...")
    ]
    review_placeholders = [("NEW_REVIEW_{}_GENERATED_HEAD", "NEW_REVIEW_{}_GENERATED_TEXT") for i in range(1, 9)]

    for i, (head_ph, text_ph) in enumerate(review_placeholders):
        original_head, original_text = reviews[i]
        prompt = generate_full_review_prompt(original_head, original_text, product_title, product_description, language)
        result = prompt_gpt(prompt)
        if result and '|' in result:
            head, text = result.split('|', 1)
            replace_in_file(PRODUCT_JSON_PATH, head_ph.format(i+1), head.strip())
            replace_in_file(PRODUCT_JSON_PATH, text_ph.format(i+1), text.strip())

    print(f"\nCompleted! Check {PRODUCT_JSON_PATH}")

if __name__ == "__main__":
    main()