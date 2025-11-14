# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import json
# import logging
# from datetime import datetime
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# try:
#     nltk.data.find('corpora/stopwords')
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     logger.warning("NLTK data not found. Downloading...")
#     nltk.download('stopwords', quiet=True)
#     nltk.download('punkt', quiet=True)
#     nltk.download('wordnet', quiet=True)

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)


# class BioBERTDiagnosisSystem:
#     """Disease diagnosis system using fine-tuned BioBERT model"""
    
#     def __init__(self, model_path: str, advice_file: str):
#         self.model_path = model_path
#         self.advice_file = advice_file
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load components
#         self.load_model()
#         self.load_medical_advice()
        
#         logger.info(f"BioBERT Diagnosis System initialized on {self.device}")
    
#     def load_model(self):
#         """Load fine-tuned BioBERT model and tokenizer"""
#         try:
#             logger.info(f"Loading BioBERT model from {self.model_path}...")
            
#             # Load tokenizer and model
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
#             self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
#             self.model.to(self.device)
#             self.model.eval()
            
#             # Get label mappings (id2label, label2id should be in config)
#             self.id2label = self.model.config.id2label
#             # self.label2id = self.model.config.label2id
            
#             logger.info(f"Model loaded successfully with {len(self.id2label)} disease classes")
            
#         except Exception as e:
#             logger.error(f"Failed to load model: {e}")
#             raise Exception(f"Model loading failed: {e}")
    
#     def load_medical_advice(self):
#         """Load medical advice database"""
#         try:
#             with open(self.advice_file, 'r', encoding='utf-8') as f:
#                 self.medical_advice = json.load(f)
#             logger.info(f"Medical advice loaded for {len(self.medical_advice)} diseases")
#         except Exception as e:
#             logger.error(f"Failed to load medical advice: {e}")
#             self.medical_advice = {}

#     def preprocess_symptoms(self, text: str) -> str:
#         """Advanced text preprocessing for symptom description"""
#         try:
#             # 1. Lowercase conversion
#             text = text.lower()
            
#             # 2. Remove special characters but keep important medical punctuation
#             text = re.sub(r'[^a-zA-Z0-9\s\-\.]', ' ', text)
            
#             # 3. Remove extra whitespaces
#             text = re.sub(r'\s+', ' ', text).strip()
            
#             # 4. Tokenization
#             tokens = word_tokenize(text)
            
#             # 5. Remove stopwords (but keep medical stopwords)
#             medical_stopwords = {'pain', 'feeling', 'feel', 'have', 'has', 'had', 
#                                 'severe', 'mild', 'chronic', 'acute', 'sharp', 'dull'}
#             stop_words = set(stopwords.words('english')) - medical_stopwords
#             tokens = [word for word in tokens if word not in stop_words]
            
#             # 6. Lemmatization (convert words to base form)
#             lemmatizer = WordNetLemmatizer()
#             tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
#             # 7. Remove very short tokens (less than 2 characters)
#             tokens = [word for word in tokens if len(word) > 2]
            
#             # 8. Rejoin tokens
#             processed_text = ' '.join(tokens)
            
#             logger.info(f"Preprocessing: '{text[:50]}...' -> '{processed_text[:50]}...'")
#             return processed_text
            
#         except Exception as e:
#             logger.error(f"Preprocessing failed: {e}. Using original text.")
#             return text
    
#     def predict_disease(self, symptoms_text: str, patient_info: dict = None) -> dict:
#         """Predict diseases from symptom text"""
#         try:
#             logger.info(f"Processing symptom text: {len(symptoms_text)} characters")
            
#             # Validate input
#             if not symptoms_text or len(symptoms_text.strip()) < 10:
#                 return {
#                     'success': False,
#                     'error': 'Please provide more detailed symptoms (at least 10 characters)',
#                     'predictions': []
#                 }
            
#             if len(symptoms_text) > 2000:
#                 return {
#                     'success': False,
#                     'error': 'Symptom description too long (max 2000 characters)',
#                     'predictions': []
#                 }
            
#             symptoms_text = symptoms_text.strip()

#             processed_text = self.preprocess_symptoms(symptoms_text)

#             # Tokenize input
#             inputs = self.tokenizer(
#                 processed_text,
#                 return_tensors="pt",
#                 truncation=True,
#                 padding=True,
#                 max_length=128
#             )
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
#             # Get predictions
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 logits = outputs.logits
#                 probabilities = torch.softmax(logits, dim=-1)[0]
            
#             # Get top 3 predictions
#             top_probs, top_indices = torch.topk(probabilities, k=min(3, len(self.id2label)))
            
#             # Build predictions list
#             predictions = []
#             for prob, idx in zip(top_probs, top_indices):
#                 disease_name = self.id2label[idx.item()]
#                 probability = prob.item() * 100  # Convert to percentage
                
#                 # Get medical advice for this disease
#                 advice = self.get_disease_advice(disease_name)
                
#                 prediction = {
#                     'disease': disease_name,
#                     'probability': round(probability, 2),
#                     'confidence': self.get_confidence_level(probability),
#                     **advice
#                 }
                
#                 predictions.append(prediction)
            
#             # Calculate overall confidence
#             max_prob = predictions[0]['probability'] if predictions else 0
#             # overall_confidence = self.calculate_overall_confidence(max_prob, len(symptoms_text))
            
#             # Apply demographic adjustments if provided
#             # if patient_info:
#             #     predictions = self.adjust_for_demographics(predictions, patient_info)
            
#             response = {
#                 'success': True,
#                 'predictions': predictions,
#                 'original_symptoms': symptoms_text,
#                 'processed_symptoms': processed_text,
#                 'extracted_keywords': self.extract_keywords(processed_text),
#                 'analysis_info': {
#                     'model': 'BioBERT (Fine-tuned)',
#                     'approach': 'Transformer-based deep learning',
#                     'total_diseases': len(self.id2label),
#                     'device': str(self.device),
#                     'preprocessing_steps': [
#                         'Lowercase conversion',
#                         'Special character removal',
#                         'Tokenization (NLTK)',
#                         'Stopword removal (excluding medical terms)',
#                         'Lemmatization (WordNet)',
#                         'Short token removal'
#                     ],
#                     'tokens_before': len(symptoms_text.split()),
#                     'tokens_after': len(processed_text.split())
#                 }
#             }
            
#             logger.info(f"Diagnosis completed: Top prediction - {predictions[0]['disease']} ({predictions[0]['probability']}%)")
#             return response
            
#         except Exception as e:
#             logger.error(f"Prediction failed: {e}")
#             return {
#                 'success': False,
#                 'error': f'Diagnosis failed: {str(e)}',
#                 'predictions': [],
#                 'confidence': 0
#             }
    
#     def get_disease_advice(self, disease_name: str) -> dict:
#         """Get medical advice for a disease"""
#         disease_lower = disease_name.lower()
        
#         # Try to find exact match or close match
#         advice_data = None
#         if disease_lower in self.medical_advice:
#             advice_data = self.medical_advice[disease_lower].get('medical_advice', {})
#         else:
#             # Try partial matching
#             for key in self.medical_advice.keys():
#                 if disease_lower in key or key in disease_lower:
#                     advice_data = self.medical_advice[key].get('medical_advice', {})
#                     break
        
#         # Return advice or defaults
#         if advice_data:
#             return {
#                 'primary_symptoms': advice_data.get('primary_symptoms', []),
#                 'immediate_advice': advice_data.get('immediate_advice', []),
#                 'quick_solutions': advice_data.get('quick_solutions', []),
#                 'when_to_seek_help': advice_data.get('when_to_seek_help', 
#                     'Consult a healthcare provider if symptoms persist or worsen.'),
#                 'severity_level': advice_data.get('severity_level', 'moderate'),
#                 'specialist_recommendation': advice_data.get('specialist_recommendation', 
#                     'General Practitioner'),
#                 'symptom_category': advice_data.get('symptom_category', 'general')
#             }
#         else:
#             # Default advice
#             return {
#                 'primary_symptoms': ['Symptom information not available'],
#                 'immediate_advice': [
#                     'Monitor your symptoms carefully',
#                     'Rest and stay hydrated',
#                     'Avoid strenuous activities',
#                     'Keep track of any changes in symptoms'
#                 ],
#                 'quick_solutions': [
#                     'Maintain good hygiene',
#                     'Get adequate rest',
#                     'Follow a healthy diet',
#                     'Consult healthcare provider for proper diagnosis'
#                 ],
#                 'when_to_seek_help': 'Consult a healthcare provider if symptoms persist or worsen.',
#                 'severity_level': 'moderate',
#                 'specialist_recommendation': 'General Practitioner',
#                 'symptom_category': 'general'
#             }
    
#     def extract_keywords(self, text: str) -> list:
#         """Extract medical keywords from text (simple implementation)"""
#         medical_terms = [
#             'pain', 'fever', 'headache', 'cough', 'nausea', 'fatigue', 
#             'dizziness', 'weakness', 'vomiting', 'diarrhea', 'breathing',
#             'chest', 'heart', 'stomach', 'throat', 'muscle', 'joint',
#             'swelling', 'rash', 'bleeding', 'infection', 'chronic', 'acute'
#         ]
        
#         text_lower = text.lower()
#         found_keywords = []
        
#         for term in medical_terms:
#             if term in text_lower:
#                 found_keywords.append(term)
        
#         return found_keywords[:10]  # Return top 10
    
#     def get_confidence_level(self, probability: float) -> str:
#         """Convert probability to confidence level"""
#         if probability >= 70:
#             return 'high'
#         elif probability >= 40:
#             return 'medium'
#         else:
#             return 'low'
    
#     # def calculate_overall_confidence(self, max_prob: float, text_length: int) -> float:
#     #     """Calculate overall confidence score"""
#     #     # Base confidence from probability
#     #     base_confidence = max_prob * 0.8
        
#     #     # Bonus for detailed description
#     #     length_bonus = min(15, (text_length / 50) * 5)
        
#     #     overall = base_confidence + length_bonus
#     #     return round(min(95, max(30, overall)), 1)
    
#     # def adjust_for_demographics(self, predictions: list, patient_info: dict) -> list:
#     #     """Apply demographic adjustments to predictions"""
#     #     # Simple age/gender based adjustments
#     #     age = patient_info.get('age')
#     #     gender = patient_info.get('gender', '').lower()
        
#     #     # You can add disease-specific demographic adjustments here
#     #     # For example, certain diseases are more common in specific age groups or genders
        
#     #     return predictions

# # Global diagnosis system
# diagnosis_system = None

# def initialize_system():
#     """Initialize the diagnosis system"""
#     global diagnosis_system
    
#     try:
#         logger.info("Initializing BioBERT Diagnosis System...")
        
#         # Paths
#         model_path = 'final-model'  # Your model directory
#         advice_file = 'advice.json'  # Your medical advice file
        
#         # Check if files exist
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model directory not found: {model_path}")
#         if not os.path.exists(advice_file):
#             logger.warning(f"Advice file not found: {advice_file}. Using defaults.")
#             # Create empty advice file
#             with open(advice_file, 'w') as f:
#                 json.dump({}, f)
        
#         # Initialize system
#         diagnosis_system = BioBERTDiagnosisSystem(
#             model_path=model_path,
#             advice_file=advice_file
#         )
        
#         logger.info("System initialized successfully!")
#         return True
        
#     except Exception as e:
#         logger.error(f"Failed to initialize system: {e}")
#         return False

# # Flask Routes
# @app.route('/')
# def serve_dashboard():
#     """Serve the main HTML page"""
#     return send_from_directory('.', 'index.html')

# @app.route('/style.css')
# def serve_css():
#     """Serve CSS file"""
#     return send_from_directory('.', 'style.css')

# @app.route('/script.js')
# def serve_js():
#     """Serve JavaScript file"""
#     return send_from_directory('.', 'script.js')

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy' if diagnosis_system else 'unhealthy',
#         'system_loaded': diagnosis_system is not None,
#         'timestamp': datetime.now().isoformat(),
#         'version': '2.0.0 - BioBERT Model',
#         'model_type': 'Fine-tuned BioBERT',
#         'device': str(diagnosis_system.device) if diagnosis_system else 'N/A',
#         'total_diseases': len(diagnosis_system.id2label) if diagnosis_system else 0
#     })

# @app.route('/api/diagnose', methods=['POST'])
# def diagnose():
#     """Main diagnosis endpoint"""
#     try:
#         if not diagnosis_system:
#             return jsonify({
#                 'success': False,
#                 'error': 'Diagnosis system not available'
#             }), 500
        
#         data = request.get_json()
#         if not data:
#             return jsonify({
#                 'success': False,
#                 'error': 'No data provided'
#             }), 400
        
#         symptoms = data.get('symptoms', '').strip()
#         # patient_info = data.get('patient_info', {})
        
#         if not symptoms:
#             return jsonify({
#                 'success': False,
#                 'error': 'Please provide symptom description'
#             }), 400
        
#         logger.info(f"Processing diagnosis request: {len(symptoms)} chars")
        
#         # Process diagnosis
#         start_time = datetime.now()
#         result = diagnosis_system.predict_disease(symptoms)
#         processing_time = (datetime.now() - start_time).total_seconds()
        
#         if result.get('success'):
#             result['processing_time'] = round(processing_time, 2)
#             logger.info(f"Diagnosis completed in {processing_time:.2f}s")
        
#         return jsonify(result)
        
#     except Exception as e:
#         logger.error(f"Diagnosis endpoint error: {e}")
#         return jsonify({
#             'success': False,
#             'error': 'Internal server error'
#         }), 500

# @app.route('/api/diseases', methods=['GET'])
# def get_diseases():
#     """Get list of all diseases the model can predict"""
#     try:
#         if not diagnosis_system:
#             return jsonify({'error': 'System not loaded'}), 500
        
#         diseases = list(diagnosis_system.id2label.values())
#         return jsonify({
#             'success': True,
#             'total_diseases': len(diseases),
#             'diseases': diseases
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Error handlers
# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({'success': False, 'error': 'Internal server error'}), 500

# def main():
#     """Main application entry point"""
#     print("🏥 Medical Diagnosis System - BioBERT Edition")
#     print("=" * 60)
    
#     # Initialize the system
#     if not initialize_system():
#         print("❌ Failed to initialize diagnosis system!")
#         print("Please ensure:")
#         print("  1. 'final-model' directory exists")
#         print("  2. Model files are present in the directory")
#         print("  3. 'advice.json' file exists")
#         return
    
#     print("✅ System Features:")
#     print("   ✓ Fine-tuned BioBERT model")
#     print("   ✓ Top 3 disease predictions")
#     print("   ✓ Medical advice integration")
#     print("   ✓ Confidence scoring")
#     print("   ✓ Patient demographics consideration")
    
#     print(f"\n📊 Model Information:")
#     if diagnosis_system:
#         print(f"   Total Diseases: {len(diagnosis_system.id2label)}")
#         print(f"   Device: {diagnosis_system.device}")
#         print(f"   Model Type: BioBERT (Transformer)")
    
#     print("\n🌐 Available Endpoints:")
#     endpoints = [
#         "GET  /                  - Dashboard UI",
#         "POST /api/diagnose      - Disease prediction",
#         "GET  /api/health        - System health check",
#         "GET  /api/diseases      - List all diseases"
#     ]
#     for endpoint in endpoints:
#         print(f"   {endpoint}")
    
#     print("\n🚀 Starting server on http://localhost:5000")
#     print("=" * 60)
    
#     try:
#         app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
#     except KeyboardInterrupt:
#         print("\n\n🛑 Server stopped")
#     except Exception as e:
#         print(f"\n\n❌ Server error: {e}")

# if __name__ == '__main__':
#     main()


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from datetime import datetime
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -----------------------------------------------------
# Logging
# -----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Pre-download NLTK data (prevents failure on Render)
# -----------------------------------------------------
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

# -----------------------------------------------------
# Flask Setup
# -----------------------------------------------------
app = Flask(
    __name__,
    static_url_path='',
    static_folder='.',
    template_folder='.'
)
CORS(app)


# -----------------------------------------------------
# Diagnosis System Class
# -----------------------------------------------------
class BioBERTDiagnosisSystem:
    def __init__(self, model_path, advice_file):
        self.model_path = model_path
        self.advice_file = advice_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.load_medical_advice()

    def load_model(self):
        logger.info(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def load_medical_advice(self):
        if not os.path.exists(self.advice_file):
            logger.warning("Advice file not found, creating empty file")
            with open(self.advice_file, "w") as f:
                json.dump({}, f)

        with open(self.advice_file, "r", encoding="utf-8") as f:
            self.medical_advice = json.load(f)

    # -----------------------------
    # SAME FUNCTIONS AS YOUR CODE
    # -----------------------------
    def preprocess_symptoms(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s\-\.]", " ", text)
        text = re.sub(r"\s+", " ", text)
        tokens = word_tokenize(text)

        medical_stopwords = {'pain','feeling','feel','have','has','had',
                             'severe','mild','chronic','acute','sharp','dull'}

        stop_words = set(stopwords.words('english')) - medical_stopwords
        tokens = [w for w in tokens if w not in stop_words]

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens if len(w) > 2]

        return " ".join(tokens)

    def predict_disease(self, symptoms_text):
        processed = self.preprocess_symptoms(symptoms_text)

        inputs = self.tokenizer(
            processed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        top_probs, top_idx = torch.topk(probs, k=min(3, len(self.id2label)))

        predictions = []
        for prob, idx in zip(top_probs, top_idx):
            disease = self.id2label[idx.item()]
            predictions.append({
                "disease": disease,
                "probability": round(prob.item() * 100, 2)
            })

        return {
            "success": True,
            "original": symptoms_text,
            "processed": processed,
            "predictions": predictions
        }


# -----------------------------------------------------
# Initialize global model
# -----------------------------------------------------
diagnosis_system = BioBERTDiagnosisSystem(
    model_path="final-model",
    advice_file="advice.json"
)


# -----------------------------------------------------
# ROUTES
# -----------------------------------------------------
@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory('.', 'favicon.ico')


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({"success": False, "error": "No symptoms provided"}), 400

    result = diagnosis_system.predict_disease(data["symptoms"])
    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "device": str(diagnosis_system.device),
    })


# -----------------------------------------------------
# Render will call this through Gunicorn
# -----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
