import os
import glob
import torch
from PIL import Image
#from transformers import Blip2Processor, Blip2ForConditionalGeneration

# # Force redownload of model Blip2
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", force_download=True)
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", force_download=True)


# Load the pretrained processor and model
from transformers import AutoProcessor, BlipForConditionalGeneration
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# image directory to the current folder
image_dir = "./images"
image_exts = ["jpg", "jpeg", "png"]

# Open a file to save captions
with open("LocaL_captions.txt", "w", encoding="utf-8") as caption_file:
    for image_ext in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f"*.{image_ext}")):
            try:
                # Load image
                raw_image = Image.open(img_path).convert('RGB')

                # Move input tensors to GPU/CPU
                inputs = processor(raw_image, return_tensors="pt").to(device)

                # Generate caption
                out = model.generate(**inputs, max_new_tokens=50)

                # Decode output
                caption = processor.decode(out[0], skip_special_tokens=True)

                # Write to file
                caption_file.write(f"{os.path.basename(img_path)}: {caption}\n")
                print(f"Captioned: {img_path}")  # Debug message
            except Exception as e:
                print(f"Error processing {img_path}: {e}")  # Handle errors gracefully
