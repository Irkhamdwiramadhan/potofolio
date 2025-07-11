# encode_image.py
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    # Tentukan MIME type berdasarkan ekstensi file
    if image_path.endswith(".png"):
        mime_type = "image/png"
    elif image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        mime_type = "image/jpeg"
    elif image_path.endswith(".webp"):
        mime_type = "image/webp"
    else:
        mime_type = "application/octet-stream" # Default jika tidak dikenal
        print(f"Warning: Unknown image type for {image_path}. Using generic MIME type.")

    return f"data:{mime_type};base64,{encoded_string}"

# Ganti dengan nama file gambar Anda
image_file = "background_dark.png" 
try:
    base64_string = image_to_base64(image_file)
    print("--- COPY THIS ENTIRE STRING AND PASTE INTO YOUR style.css FILE ---")
    print(base64_string)
    print("--- END OF BASE64 STRING ---")
except FileNotFoundError:
    print(f"Error: File '{image_file}' not found. Make sure it's in the same directory as this script.")
except Exception as e:
    print(f"An error occurred during encoding: {e}")