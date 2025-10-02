import json
import re
import shutil
import os
import sys

def fix_json_quotes_in_html(content):
    """Fix unescaped quotes in HTML attributes by replacing them with single quotes"""
    
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check if line contains HTML attributes
        if 'href=' in line or 'src=' in line or 'title=' in line or 'alt=' in line:
            # Replace double quotes in HTML attributes with single quotes
            # This avoids the escaping issue entirely
            fixed_line = line
            
            # Replace href="..." with href='...'
            if 'href="' in fixed_line:
                parts = fixed_line.split('href="')
                result = [parts[0]]
                for part in parts[1:]:
                    # Find the closing quote
                    end_idx = part.find('"')
                    if end_idx != -1:
                        result.append("href='" + part[:end_idx] + "'" + part[end_idx+1:])
                    else:
                        result.append('href="' + part)
                fixed_line = ''.join(result)
            
            # Replace title="..." with title='...'
            if 'title="' in fixed_line:
                parts = fixed_line.split('title="')
                result = [parts[0]]
                for part in parts[1:]:
                    end_idx = part.find('"')
                    if end_idx != -1:
                        result.append("title='" + part[:end_idx] + "'" + part[end_idx+1:])
                    else:
                        result.append('title="' + part)
                fixed_line = ''.join(result)
            
            # Replace src="..." with src='...'
            if 'src="' in fixed_line:
                parts = fixed_line.split('src="')
                result = [parts[0]]
                for part in parts[1:]:
                    end_idx = part.find('"')
                    if end_idx != -1:
                        result.append("src='" + part[:end_idx] + "'" + part[end_idx+1:])
                    else:
                        result.append('src="' + part)
                fixed_line = ''.join(result)
            
            # Replace alt="..." with alt='...'
            if 'alt="' in fixed_line:
                parts = fixed_line.split('alt="')
                result = [parts[0]]
                for part in parts[1:]:
                    end_idx = part.find('"')
                    if end_idx != -1:
                        result.append("alt='" + part[:end_idx] + "'" + part[end_idx+1:])
                    else:
                        result.append('alt="' + part)
                fixed_line = ''.join(result)
            
            # Replace class="..." with class='...'
            if 'class="' in fixed_line:
                parts = fixed_line.split('class="')
                result = [parts[0]]
                for part in parts[1:]:
                    end_idx = part.find('"')
                    if end_idx != -1:
                        result.append("class='" + part[:end_idx] + "'" + part[end_idx+1:])
                    else:
                        result.append('class="' + part)
                fixed_line = ''.join(result)
            
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def validate_and_fix_json_file(filepath):
    """Try to load JSON, fix if needed, and return the data"""
    
    print(f"📖 Reading: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to parse as-is first
    try:
        data = json.loads(content)
        print(f"✅ Valid JSON: {filepath}")
        return data, False
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON Error at line {e.lineno}, column {e.colno}: {e.msg}")
        print(f"🔧 Attempting to fix...")
        
        # Show context around the error
        lines = content.split('\n')
        if e.lineno <= len(lines):
            print(f"\n📍 Error location:")
            error_line = lines[e.lineno - 1]
            print(f"    {e.lineno:4d} | {error_line}")
            print(f"         {'':>{e.colno}}^")
        
        # Try to fix HTML quotes in JSON
        fixed_content = fix_json_quotes_in_html(content)
        
        try:
            data = json.loads(fixed_content)
            print(f"✅ Fixed JSON successfully!")
            
            # Save backup
            backup_path = filepath + '.backup'
            shutil.copy(filepath, backup_path)
            print(f"💾 Backed up original to: {backup_path}")
            
            # Save fixed version
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"💾 Saved fixed JSON to: {filepath}")
            return data, True
            
        except json.JSONDecodeError as e2:
            print(f"❌ Could not auto-fix JSON.")
            print(f"New error at line {e2.lineno}, column {e2.colno}: {e2.msg}")
            sys.exit(1)

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
    """Recursively find all text properties and their paths"""
    text_paths = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{parent_path}.{key}" if parent_path else key
            
            if key in ["text", "rating_text", "row_content", "rating_count", "Announcement", 
                       "heading", "subtext", "feature_title", "feature_text", "small_text",
                       "newsletter_heading", "copyright", "bottom_text"] and isinstance(value, str):
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
    
    stripped = text.strip()
    return stripped[3:-4]

def clean_double_p_tags(text):
    """Clean up double <p> tags"""
    if not isinstance(text, str):
        return text
    
    text = re.sub(r"<p>'\s*<p>", "<p>", text)
    text = re.sub(r"</p>'\s*</p>", "</p>", text)
    
    return text

def fix_text_properties(original_json, target_json):
    """Fix text properties in target JSON to match original's paragraph wrapping"""
    
    original_text_paths = find_text_properties(original_json)
    changes_made = []
    
    for path in original_text_paths:
        original_value = get_nested_value(original_json, path)
        target_value = get_nested_value(target_json, path)
        
        if target_value is None:
            continue
        
        original_wrapped = is_wrapped_in_p(original_value)
        target_wrapped = is_wrapped_in_p(target_value)
        
        if original_wrapped and not target_wrapped:
            new_value = wrap_in_p(target_value)
            new_value = clean_double_p_tags(new_value)
            set_nested_value(target_json, path, new_value)
            changes_made.append(f"Wrapped '{path}'")
            
        elif not original_wrapped and target_wrapped:
            new_value = unwrap_p(target_value)
            new_value = clean_double_p_tags(new_value)
            set_nested_value(target_json, path, new_value)
            changes_made.append(f"Unwrapped '{path}'")
        else:
            cleaned_value = clean_double_p_tags(target_value)
            if cleaned_value != target_value:
                set_nested_value(target_json, path, cleaned_value)
                changes_made.append(f"Cleaned double tags in '{path}'")
    
    return target_json, changes_made

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <original_json_path> <target_json_path>")
        sys.exit(1)
    
    original_path = sys.argv[1]
    target_path = sys.argv[2]
    
    if not os.path.exists(original_path):
        print(f"❌ Error: Original file not found: {original_path}")
        sys.exit(1)
    
    if not os.path.exists(target_path):
        print(f"❌ Error: Target file not found: {target_path}")
        sys.exit(1)
    
    # Validate and fix both JSONs if needed
    print("\n=== Processing Original JSON ===")
    original_json, _ = validate_and_fix_json_file(original_path)
    
    print("\n=== Processing Target JSON ===")
    target_json, was_fixed = validate_and_fix_json_file(target_path)
    
    # Fix text properties
    print("\n=== Comparing and Fixing Text Properties ===")
    fixed_json, changes = fix_text_properties(original_json, target_json)
    
    # Save result
    target_dir = os.path.dirname(target_path)
    output_path = os.path.join(target_dir, 'fixed_target.json') if target_dir else 'fixed_target.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_json, f, indent=4, ensure_ascii=False)
    
    if changes:
        print(f"\n✅ Made {len(changes)} paragraph wrapping changes:")
        for change in changes[:10]:
            print(f"  - {change}")
        if len(changes) > 10:
            print(f"  ... and {len(changes) - 10} more")
    else:
        print("\n✅ No paragraph wrapping changes needed")
    
    # Replace target file
    os.remove(target_path)
    shutil.move(output_path, target_path)
    
    print(f"\n✅ Successfully processed '{os.path.basename(target_path)}'")