# 🩺 Digital Diagnosis – AI-Powered Medical Assistant

Digital Diagnosis is an AI-driven medical support system built using **Flask**, **Transformers**, and **PyTorch**.  
The model predicts the most probable disease based on user symptoms and provides relevant medical guidance.

This project uses a fine-tuned Transformer model stored locally in the `final-model/` directory and a clean web interface using HTML/CSS/JS.

---

## 🚀 Features

### ✔ Symptom-to-Disease Prediction  
Uses a fine-tuned Transformer model to classify user symptoms into likely disease categories.

### ✔ Medical Advice Generation  
Uses a curated JSON file (`advice.json`) to provide helpful suggestions for the predicted condition.

### ✔ Modern Web Interface  
Frontend built using HTML, CSS, and JavaScript inside **Flask** templates and static folders.

### ✔ Fast and Lightweight  
No external API calls required. Everything runs locally — ideal for deployment.

---

## 🏗 Tech Stack

### **Backend**
- Python 3.x  
- Flask  
- Transformers (HuggingFace)  
- PyTorch  
- Scikit-Learn  
- Pandas / NumPy  

### **Frontend**
- HTML  
- CSS  
- JavaScript  

---

## 📁 Project Structure

your-project/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── final-model/
│ ├── config.json
│ ├── tokenizer.json
│ ├── pytorch_model.bin
│ └── vocab / merges / model files
│
├── advice.json
│
├── templates/
│ └── index.html
│
└── static/
├── style.css
└── script.js


## ▶️ How to Run the App Locally

### **1️⃣ Install Dependencies**

### **2️⃣ Run Flask App**

### **3️⃣ Open in Browser**


---

## 📦 Model Details

- Model stored locally in `final-model/`  
- Loaded using HuggingFace Transformers  
- Fine-tuned using your symptom–disease dataset  
- Fast inference with PyTorch


---

## 📝 License
This project is for educational and experimental use only.  
Predictions generated should NOT be used as a replacement for professional medical diagnosis.

