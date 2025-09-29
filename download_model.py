import gdown
import os

def download_model():
    url = "https://drive.google.com/file/d/1Y8XC5LTKqwTJKKsS2X32_YfHGI_PjXRp/view?usp=drive_link"  # apna file id yaha daalo
    output = "garbage_classification_model_inception.keras"

    if not os.path.exists(output):
        print("Downloading model from Google Drive...")
        gdown.download(url, output, quiet=False)
    else:
        print("Model already exists locally.")
    
    return output
