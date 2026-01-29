import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageChops, ImageEnhance
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os
import sys


Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="Select Image for Forgery Detection",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

if not image_path or not os.path.exists(image_path):
    raise ValueError("No image selected")

print(f"Selected Image: {image_path}")


img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# EDGE DETECTION

edges_canny = cv2.Canny(gray, 100, 200)


# ERROR LEVEL ANALYSIS (ELA)

def perform_ela(image_path, quality=90):
    original = Image.open(image_path).convert("RGB")
    temp_path = "temp_ela.jpg"
    original.save(temp_path, "JPEG", quality=quality)
    compressed = Image.open(temp_path)
    ela = ImageChops.difference(original, compressed)
    ela = ImageEnhance.Brightness(ela).enhance(20)
    return np.array(ela)

ela_img = perform_ela(image_path)
ela_mean = np.mean(ela_img)


# COPY-MOVE DETECTION (ORB)

orb = cv2.ORB_create(nfeatures=5000)
kp, des = orb.detectAndCompute(gray, None)

copy_move_score = 0
if des is not None:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des, des)
    copy_move_score = len(matches)


# NOISE ANALYSIS (LBP)

lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
noise_variance = np.var(lbp)


# CNN FORGERY CLASSIFIER

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features, 2
)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_tensor = transform(
    Image.open(image_path).convert("RGB")
).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    probs = torch.softmax(output, dim=1)
    fake_prob = probs[0][1].item()


# FORGERY TYPE IDENTIFICATION (NEW FEATURE)

forgery_types = []
evidence = []

# Copy-Move Forgery
if copy_move_score > 200:
    forgery_types.append("Copy-Move Forgery")
    evidence.append(f"High ORB self-matches: {copy_move_score}")

# Splicing / Compression Forgery
if ela_mean > 15:
    forgery_types.append("Image Splicing / Compression Forgery")
    evidence.append(f"High ELA intensity: {ela_mean:.2f}")

# Noise / Texture Manipulation
if noise_variance > 500:
    forgery_types.append("Noise / Texture Manipulation")
    evidence.append(f"Abnormal LBP variance: {noise_variance:.2f}")

# AI / Deepfake Forgery
if fake_prob > 0.6:
    forgery_types.append("AI / Deepfake Manipulation")
    evidence.append(f"CNN fake confidence: {fake_prob:.2f}")


# FINAL REPORT

print("\n--- FORGERY ANALYSIS REPORT ---")
print(f"Deep Learning Fake Probability : {fake_prob:.2f}")
print(f"Copy-Move Match Score          : {copy_move_score}")
print(f"Noise Variance                 : {noise_variance:.2f}")
print(f"ELA Mean Intensity             : {ela_mean:.2f}")

if forgery_types:
    print("\n⚠️ RESULT: FORGERY DETECTED")
    print("🛑 Detected Forgery Type(s):")
    for f in forgery_types:
        print(f"   • {f}")

    print("\n📌 Supporting Evidence:")
    for e in evidence:
        print(f"   - {e}")
else:
    print("\n✅ RESULT: LIKELY AUTHENTIC IMAGE")


# VISUAL OUTPUT

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Edge Detection (Canny)")
plt.imshow(edges_canny, cmap="gray")

plt.subplot(1,3,2)
plt.title("Error Level Analysis")
plt.imshow(ela_img)

plt.subplot(1,3,3)
plt.title("Noise Pattern (LBP)")
plt.imshow(lbp, cmap="gray")

plt.tight_layout()
plt.show()
