import json
import re
import shutil
import os
def get_nested_value(data, path):
    """Get value from nested dictionary using dot notation path"""
    keys = path.split('.')
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def set_nested_value(data, path, value):
    """Set value in nested dictionary using dot notation path"""
    keys = path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value

def find_text_properties(data, parent_path=""):
    """Recursively find all 'text' and 'rating_text' properties and their paths"""
    text_paths = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{parent_path}.{key}" if parent_path else key
            
            if key in ["text", "rating_text","row_content","rating_count"] and isinstance(value, str):
                text_paths.append(current_path)
            elif isinstance(value, (dict, list)):
                text_paths.extend(find_text_properties(value, current_path))
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = f"{parent_path}[{i}]"
            text_paths.extend(find_text_properties(item, current_path))
    
    return text_paths

def is_wrapped_in_p(text):
    """Check if text is wrapped in <p> tags"""
    if not isinstance(text, str):
        return False
    
    stripped = text.strip()
    return stripped.startswith('<p>') and stripped.endswith('</p>')

def wrap_in_p(text):
    """Wrap text in <p> tags"""
    if not text.strip():
        return text
    return f"<p>{text}</p>"

def unwrap_p(text):
    """Remove <p> tags from text"""
    if not is_wrapped_in_p(text):
        return text
    
    # Remove outer <p> and </p> tags
    stripped = text.strip()
    return stripped[3:-4]  # Remove '<p>' and '</p>'

def clean_double_p_tags(text):
    """Clean up double <p> tags like <p>'<p> and </p>'</p>"""
    if not isinstance(text, str):
        return text
    
    # Fix <p>'<p> -> <p>
    text = re.sub(r"<p>'\s*<p>", "<p>", text)
    
    # Fix </p>'</p> -> </p>
    text = re.sub(r"</p>'\s*</p>", "</p>", text)
    
    return text

def fix_text_properties(original_json, target_json):
    """Fix text properties in target JSON to match original's paragraph wrapping"""
    
    # Find all text properties in original
    original_text_paths = find_text_properties(original_json)
    
    changes_made = []
    
    for path in original_text_paths:
        original_value = get_nested_value(original_json, path)
        target_value = get_nested_value(target_json, path)
        
        if target_value is None:
            continue  # Property doesn't exist in target
        
        original_wrapped = is_wrapped_in_p(original_value)
        target_wrapped = is_wrapped_in_p(target_value)
        
        if original_wrapped and not target_wrapped:
            # Original is wrapped, target is not - wrap target
            new_value = wrap_in_p(target_value)
            new_value = clean_double_p_tags(new_value)
            set_nested_value(target_json, path, new_value)
            changes_made.append(f"Wrapped '{path}': '{target_value[:50]}...' -> '<p>...'")
            
        elif not original_wrapped and target_wrapped:
            # Original is not wrapped, target is - unwrap target
            new_value = unwrap_p(target_value)
            new_value = clean_double_p_tags(new_value)
            set_nested_value(target_json, path, new_value)
            changes_made.append(f"Unwrapped '{path}': '<p>...' -> '{new_value[:50]}...'")
        else:
            # Clean existing double tags even if wrapping matches
            cleaned_value = clean_double_p_tags(target_value)
            if cleaned_value != target_value:
                set_nested_value(target_json, path, cleaned_value)
                changes_made.append(f"Cleaned double tags in '{path}'")
    
    return target_json, changes_made


if __name__ == "__main__":
    # Load original JSON
    with open('/content/drive/MyDrive/theme-automation-2/code/original_product.json', 'r', encoding='utf-8') as f:
        original_json = json.load(f)
    
    # Load target JSON
    with open('/content/drive/MyDrive/theme-automation-2/templates/product.json', 'r', encoding='utf-8') as f:
        target_json = json.load(f)
    
    # Fix text properties
    fixed_json, changes = fix_text_properties(original_json, target_json)
    
    # Save fixed JSON
    with open('/content/drive/MyDrive/theme-automation-2/fixed_target.json', 'w', encoding='utf-8') as f:
        json.dump(fixed_json, f, indent=2, ensure_ascii=False)
    
    # Print changes made
    if changes:
        print(f"Made {len(changes)} changes:")
        for change in changes:
            print(f"  - {change}")
    else:
        print("No changes needed")
    os.remove("/content/drive/MyDrive/theme-automation-2/templates/product.json")
    shutil.move(
        "/content/drive/MyDrive/theme-automation-2/templates/fixed_target.json",
        "/content/drive/MyDrive/theme-automation-2/templates/product.json",
    )
    print("✅ Successfully fixed and replaced 'product.json'")
    print("Fixed JSON saved as 'fixed_target.json'")