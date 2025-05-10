# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:36:05 2025

@author: sueda
"""

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
from timm import create_model
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import requests
import json
import sqlite3
import uuid
import datetime

image_transforms = transforms.Compose([
    transforms.Resize((384,384)), #görüntü boyutlarını sabitleme
    transforms.ToTensor(),       #tensöre çevirme
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #ImageNet normalizasyonu. fine tune işlemi için gerekli bir standart
])


label = {
    'bone_fracture_detection': 0, 'bone_fracture': 1, 'lung_bnt': 2, 'lung_scc': 3, 'lung_aca': 4,
    'colon_aca': 5, 'colon_bnt': 6, 'all_pro': 7, 'all_pre': 8, 'all_benign': 9, 'all_early': 10,
    'cervix_dyk': 11, 'cervix_sfi': 12, 'cervix_mep': 13, 'cervix_pab': 14, 'cervix_koc': 15,
    'oral_scc': 16, 'oral_normal': 17, 'kidney_tumor': 18, 'kidney_normal': 19, 'breast_benign': 20,
    'breast_malignant': 21, 'lymph_cll': 22, 'lymph_fl': 23, 'lymph_mcl': 24, 'brain_menin': 25,
    'brain_tumor': 26, 'brain_glioma': 27, 'ModerateDemented': 28, 'MildDemented': 29,
    'VeryMildDemented': 30, 'NonDemented': 31, 'NORMAL': 32, 'PNEUMONIA': 33,
    'meningioma_tumor': 34, 'glioma_tumor': 35, 'normal': 36, 'pituitary_tumor': 37
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Model yapısını yeniden oluştur (aynı parametrelerle!)
model = create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=0)
model.load_state_dict(torch.load('swin_transformer_finetuned2.pth', map_location=device), strict = True)
model = model.to(device)
model.eval()

#eğitim transform ile aynı
image_transforms = transforms.Compose([
    transforms.Resize((384,384)), #görüntü boyutlarını sabitleme
    transforms.ToTensor(),       #tensöre çevirme
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #ImageNet normalizasyonu. fine tune işlemi için gerekli bir standart
])


def predict_single_image(model, image_path, transform, class_mapping, device=device, threshold_high=0.75, threshold_low=0.40):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        max_prob, predicted_index = torch.max(probabilities, 1)

    max_prob = max_prob.item()
    predicted_index = predicted_index.item()
    predicted_class = list(class_mapping.keys())[list(class_mapping.values()).index(predicted_index)]

    if max_prob >= threshold_high:
        print(f"✅ {predicted_class} sınıfı yüksek güvenle (%{max_prob*100:.1f}) tahmin edildi.")
        return predicted_class
    elif threshold_low <= max_prob < threshold_high:
        print(f"⚠️ {predicted_class} sınıfı düşük güvenle (%{max_prob*100:.1f}) tahmin edildi. Emin olunmayabilir.")
        return predicted_class
    else:
        print(f"❌ Bu görsel hakkında yorum yapılamamaktadır (güven: %{max_prob*100:.1f}).")
        return None

image_path = "C:\\Users\\sueda\\bitirme\\test\\chest_Xray_test\PNEUMONIA\\person21_virus_53.jpeg"  # Görsel yolu
disease = predict_single_image(model, image_path, image_transforms, label, device)

url = "http://192.168.1.39:1234/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

system_prompt = "You are a medical assistant who helps doctors make diagnoses based on patient complaints. Be detailed, concise, and in paragraphs. Don't repeat yourself. If the question is not medical, explain that you are only a medical assistant and can only answer questions about medical issues.say that 'I cannot answer questions that are not medical.'"
prompt = "What do you know about spider man?"

payload = {
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 800,
    "temperature": 0.8
} 

response = requests.post(url, headers=headers, data=json.dumps(payload)).json()
output = response['choices'][0]['message']['content']
print(output)

# Veritabanı bağlantısını başlat
conn = sqlite3.connect('medical_history.db')
c = conn.cursor()

# Tabloyu oluştur (yoksa)
c.execute('''CREATE TABLE IF NOT EXISTS sessions (
    id TEXT,
    timestamp TEXT,
    user_input TEXT,
    ai_response TEXT,
    disease_prediction TEXT,
    uploaded_files TEXT
)''')

# Veriyi ekle
c.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)", 
          (str(uuid.uuid4()), str(datetime.datetime.now()), prompt, output, disease, image_path))

# Değişiklikleri kaydet ve bağlantıyı kapat
conn.commit()
conn.close()

"""
from googletrans import Translator

# Create a Translator object
translator = Translator()

# Translate a text
result = translator.translate(output, src='en', dest='tr')

# Print translated text
print(result.text)"""
