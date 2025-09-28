import json
import re
import shutil
import os
from datetime import datetime
from fix_json_line import fix_json_line

def fix_json(json_string):
    # Fix double quotes in text content - proper escaping
    json_string = re.sub(r': *""([^"]*?)""', r': "\"\1\""', json_string)
    
    # Remove wrapping single quotes from values
    json_string = re.sub(r': *\'([^\']*?)\'(?=\s*[,}])', r': "\1"', json_string)
    
    # Fix escaped single quotes in content
    json_string = json_string.replace("\\'", "'")
    
    return json_string

def main(input_file):
    # Replace with your actual file path
    
    # Create timestamp for backups
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f'/content/drive/MyDrive/theme-automation-2/test_backup_{timestamp}.json'
    
    try:
        # Create backup of original file
        shutil.copy2(input_file, backup_file)
        print(f"Backup created: {backup_file}")
        
        # Read the problematic JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("Original file loaded successfully")
        
        # Apply fixes
        fixed_content = fix_json(content)
        
        # Try to parse to validate
        parsed = json.loads(fixed_content)
        print("JSON is now valid!")
        
        # Update the original file with proper formatting
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        
        print(f"Fixed JSON updated in original file: {input_file}")
        
    except FileNotFoundError:
        print(f"File {input_file} not found. Please check the file path.")
        
    except json.JSONDecodeError as e:
        print(f"JSON is still invalid after fixes: {e}")
        print("Line:", e.lineno, "Column:", e.colno)
        
        # Create backup before saving partially fixed content
        partially_fixed_file = '/content/drive/MyDrive/theme-automation-2/templates/partially_fixed.json'
        
        # If partially_fixed.json already exists, back it up too
        try:
            if open(partially_fixed_file, 'r'):
                partially_backup = f'partially_fixed_backup_{timestamp}.json'
                shutil.copy2(partially_fixed_file, partially_backup)
                print(f"Existing partially_fixed.json backed up to: {partially_backup}")
        except FileNotFoundError:
            pass  # File doesn't exist, no need to backup
        except Exception as backup_error:
            print(f"Warning: Could not backup existing partially_fixed.json: {backup_error}")
        
        # Save the partially fixed version for manual inspection
        with open(partially_fixed_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"Partially fixed content saved to '{partially_fixed_file}' for manual review")
        
        # Try to restore original file since fix failed
        try:
            shutil.copy2(backup_file, input_file)
            print(f"Original file restored from backup due to parsing error.")
        except Exception as restore_error:
            print(f"Warning: Could not restore original file: {restore_error}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
        # Try to restore original file
        try:
            shutil.copy2(backup_file, input_file)
            print(f"Original file restored from backup due to error.")
        except Exception as restore_error:
            print(f"Warning: Could not restore original file: {restore_error}")

if __name__ == "__main__":
    main('/content/drive/MyDrive/theme-automation-2/templates/product.json')
    fix_json_line("/content/drive/MyDrive/theme-automation-2/templates/partially_fixed.json")
    os.remove("/content/drive/MyDrive/theme-automation-2/templates/product.json")
    shutil.move("exported_single_line.json", "/content/drive/MyDrive/theme-automation-2/templates/product.json")

