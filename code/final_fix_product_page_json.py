import json
import re
import os
import shutil


def fix_json_data(data):
    def process_value(key, value):
        if key == "rating_text" and isinstance(value, (int, float)):
            return str(value)

        elif key == "rating_count" and isinstance(value, str):
            cleaned = re.sub(r"</?p>", "", value)
            return cleaned.strip()

        elif key == "rating_text" and isinstance(value, str):
            cleaned = re.sub(r"</?p>", "", value)
            return cleaned.strip()

        # Remove the automatic <p> wrapping for "text" fields
        # elif key == "text" and isinstance(value, str):
        #     return clean_and_wrap_content(value)

        elif key == "row_content" and isinstance(value, str):
            # Remove outer quotes if present
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            return value

        elif key in [
            "percent_TBmnaV",
            "percent_nJJhny", 
            "percent_UUr9wc",
            "percent_A4eLEH",
        ] and isinstance(value, str):
            # Remove outer quotes if present
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            return value

        return value

    def process_dict(obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if isinstance(value, dict):
                    result[key] = process_dict(value)
                elif isinstance(value, list):
                    result[key] = [
                        process_dict(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    result[key] = process_value(key, value)
            return result
        return obj

    def process_percent_blocks(obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key in [
                    "percent_TBmnaV",
                    "percent_nJJhny",
                    "percent_UUr9wc", 
                    "percent_A4eLEH",
                ] and isinstance(value, dict):
                    if "settings" in value and "text" in value["settings"]:
                        text_value = value["settings"]["text"]
                        if isinstance(text_value, str):
                            # Just remove quotes, don't wrap with <p> tags
                            if text_value.startswith("'") and text_value.endswith("'"):
                                value["settings"]["text"] = text_value[1:-1]

                if isinstance(value, dict):
                    result[key] = process_percent_blocks(value)
                elif isinstance(value, list):
                    result[key] = [
                        process_percent_blocks(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    result[key] = value
            return result
        return obj

    processed = process_dict(data)
    return process_percent_blocks(processed)


if __name__ == "__main__":
    with open(
        "/content/drive/MyDrive/theme-automation-2/templates/product.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    fixed_data = fix_json_data(data)

    with open(
        "/content/drive/MyDrive/theme-automation-2/templates/fixed_product.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)

    os.remove("/content/drive/MyDrive/theme-automation-2/templates/product.json")
    shutil.move(
        "/content/drive/MyDrive/theme-automation-2/templates/fixed_product.json",
        "/content/drive/MyDrive/theme-automation-2/templates/product.json",
    )
    print("✅ Successfully fixed and replaced 'product.json'")