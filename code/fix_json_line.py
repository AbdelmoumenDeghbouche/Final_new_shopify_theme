import json
import re
from datetime import datetime

# Load the current JSON file
def fix_json_line(input_file='partially_fixed.json'):
    try:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()
            # Clean up the content
            content = re.sub(r'[\x00-\x1F\x7F]', '', content)  # Remove control characters
            data = json.loads(content)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        exit(1)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"exported_single_line.json"

    # Export to new JSON file with proper formatting
    try:
        with open(output_filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        
        print(f"JSON exported successfully to: {output_filename}")
        print(f"File size: {len(json.dumps(data, ensure_ascii=False))} characters")
        
        # Verify the specific section exists and show preview
        section_id = "9ccffc8d-e4c7-404f-8007-8c5162f22285"
        if section_id in data["sections"]["main"]["blocks"]:
            content = data["sections"]["main"]["blocks"][section_id]["settings"]["content"]
            print(f"Content preview (first 100 chars): {content[:100]}...")
        
    except Exception as e:
        print(f"Error exporting JSON: {e}")