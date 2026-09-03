# Advanced Phishing Detector - Customization Guide 🚀

For cybersecurity professionals and data scientists extending the model.

---

## 📚 Table of Contents

1. [Custom Feature Engineering](#custom-feature-engineering)
2. [Dataset Integration](#dataset-integration)
3. [Threshold Optimization](#threshold-optimization)
4. [Model Tuning](#model-tuning)
5. [Real-time Deployment](#real-time-deployment)
6. [Advanced Metrics](#advanced-metrics)

---

## 🔧 Custom Feature Engineering

### Adding Email Header Features

```python
class AdvancedEmailFeatureExtractor(EmailFeatureExtractor):
    """Extended feature extractor with header analysis"""
    
    def extract_header_features(self, email_obj):
        """Extract SPF, DKIM, DMARC, Reply-To inconsistencies"""
        features = {
            'spf_pass': 0,
            'dkim_pass': 0,
            'dmarc_pass': 0,
            'reply_to_mismatch': 0,
            'from_domain_age': 0,
            'has_return_path': 1
        }
        
        # SPF Check
        spf_status = email_obj.get('X-SPF', 'unknown')
        features['spf_pass'] = 1 if 'pass' in spf_status.lower() else 0
        
        # DKIM Check
        dkim_status = email_obj.get('X-DKIM', 'unknown')
        features['dkim_pass'] = 1 if 'pass' in dkim_status.lower() else 0
        
        # DMARC Check
        dmarc_status = email_obj.get('X-DMARC', 'unknown')
        features['dmarc_pass'] = 1 if 'pass' in dmarc_status.lower() else 0
        
        # Reply-To vs From mismatch
        from_addr = email_obj.get('From', '')
        reply_to = email_obj.get('Reply-To', '')
        if from_addr and reply_to and from_addr != reply_to:
            features['reply_to_mismatch'] = 1
        
        return features
    
    def extract_domain_reputation_features(self, email_text):
        """Check domain reputation indicators"""
        features = {
            'domain_suspicious': 0,
            'new_domain_indicator': 0,
            'similar_to_popular': 0
        }
        
        # Extract domains from email
        domains = re.findall(r'(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}', 
                            email_text.lower())
        
        # Check for suspicious patterns
        suspicious_patterns = [
            'paypal-', 'amazon-', 'apple-', 'google-',  # Impersonation
            'secure-', 'verify-', 'update-', 'confirm-', # Social engineering
            '-secure', '-verify', '-login'  # Domain mimicking
        ]
        
        for domain in domains:
            for pattern in suspicious_patterns:
                if pattern in domain:
                    features['domain_suspicious'] = 1
        
        return features
    
    def extract_all_features(self, email_obj, text):
        """Extract all features including headers"""
        base_features = super().extract_all_features(text)
        header_features = self.extract_header_features(email_obj)
        domain_features = self.extract_domain_reputation_features(text)
        
        return {**base_features, **header_features, **domain_features}
```

### Adding Attachment Analysis Features

```python
def extract_attachment_features(email_obj):
    """Analyze attachments for phishing indicators"""
    features = {
        'has_attachments': 0,
        'has_executable': 0,
        'has_macro_enabled': 0,
        'has_archive': 0,
        'suspicious_filename': 0,
        'attachment_count': 0
    }
    
    if not hasattr(email_obj, 'attachments'):
        return features
    
    features['has_attachments'] = 1 if email_obj.attachments else 0
    features['attachment_count'] = len(email_obj.attachments)
    
    dangerous_extensions = ['.exe', '.bat', '.scr', '.vbs', '.com', '.pif']
    macro_extensions = ['.docm', '.xlsm', '.pptm']
    archive_extensions = ['.zip', '.rar', '.7z', '.tar']
    
    suspicious_patterns = ['invoice', 'payment', 'wire', 'urgent', 'update']
    
    for attachment in email_obj.attachments:
        filename = attachment.filename.lower()
        
        if any(ext in filename for ext in dangerous_extensions):
            features['has_executable'] = 1
        
        if any(ext in filename for ext in macro_extensions):
            features['has_macro_enabled'] = 1
        
        if any(ext in filename for ext in archive_extensions):
            features['has_archive'] = 1
        
        # Check for suspicious filename patterns
        for pattern in suspicious_patterns:
            if pattern in filename and not any(c.isdigit() for c in filename):
                features['suspicious_filename'] = 1
    
    return features
```

---

## 📊 Dataset Integration

### Loading from CSV

```python
def load_emails_from_csv(csv_path):
    """Load emails from CSV file"""
    df = pd.read_csv(csv_path)
    
    # Expected columns: 'email_text' and 'label' (1=phishing, 0=safe)
    emails = df['email_text'].tolist()
    labels = df['label'].tolist()
    
    print(f"Loaded {len(emails)} emails")
    print(f"Phishing: {sum(labels)}, Legitimate: {len(labels) - sum(labels)}")
    
    # Check class balance
    phishing_ratio = sum(labels) / len(labels)
    if phishing_ratio < 0.3 or phishing_ratio > 0.7:
        print("⚠️ WARNING: Unbalanced dataset detected!")
        print(f"Phishing ratio: {phishing_ratio:.2%}")
    
    return emails, labels
```

### Loading from EML Files (Real Email Format)

```python
import email
from email.parser import Parser

def load_emails_from_eml_directory(directory_path, phishing=True):
    """Load real email files from directory"""
    emails = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.eml'):
            with open(os.path.join(directory_path, filename), 'rb') as f:
                msg = email.message_from_binary_file(f)
                
                # Extract text
                if msg.is_multipart():
                    text_parts = []
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            text_parts.append(part.get_payload())
                    email_text = ' '.join(text_parts)
                else:
                    email_text = msg.get_payload()
                
                emails.append({
                    'text': email_text,
                    'subject': msg.get('Subject', ''),
                    'from': msg.get('From', ''),
                    'label': 1 if phishing else 0
                })
    
    return emails
```

### Balancing Imbalanced Dataset

```python
from sklearn.utils import resample

def balance_dataset(emails, labels, strategy='oversample'):
    """Balance phishing vs legitimate emails"""
    df = pd.DataFrame({'email': emails, 'label': labels})
    
    phishing = df[df['label'] == 1]
    legitimate = df[df['label'] == 0]
    
    print(f"Before balancing: Phishing={len(phishing)}, Legitimate={len(legitimate)}")
    
    if strategy == 'oversample':
        # Oversample minority class
        if len(phishing) < len(legitimate):
            phishing = resample(phishing, n_samples=len(legitimate), random_state=42)
        else:
            legitimate = resample(legitimate, n_samples=len(phishing), random_state=42)
    
    elif strategy == 'undersample':
        # Undersample majority class
        if len(phishing) > len(legitimate):
            phishing = resample(phishing, n_samples=len(legitimate), random_state=42)
        else:
            legitimate = resample(legitimate, n_samples=len(phishing), random_state=42)
    
    balanced_df = pd.concat([phishing, legitimate])
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    
    print(f"After balancing: Phishing={sum(balanced_df['label'])}, "
          f"Legitimate={len(balanced_df) - sum(balanced_df['label'])}")
    
    return balanced_df['email'].tolist(), balanced_df['label'].tolist()
```

---

## 📈 Threshold Optimization

### Dynamic Threshold Finding

```python
from scipy.optimize import minimize_scalar

def find_optimal_threshold(y_true, y_pred_proba, metric='f1', sensitivity_weight=2.0):
    """Find optimal classification threshold"""
    
    def objective(threshold):
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        if metric == 'f1':
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return -f1  # Negative because we minimize
        
        elif metric == 'weighted':
            # Weighted metric: penalize false negatives more
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            weighted_score = (sensitivity_weight * sensitivity + specificity) / (sensitivity_weight + 1)
            return -weighted_score
    
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    optimal_threshold = result.x
    
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    return optimal_threshold
```

### Threshold vs Metric Analysis

```python
def analyze_threshold_tradeoffs(y_true, y_pred_proba):
    """Visualize how metrics change with threshold"""
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = {
        'threshold': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': []
    }
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results['threshold'].append(threshold)
        results['accuracy'].append((tp + tn) / (tp + tn + fp + fn))
        results['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        results['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        results['precision'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 6))
    for metric in ['accuracy', 'sensitivity', 'specificity', 'precision']:
        plt.plot(df['threshold'], df[metric], marker='o', label=metric)
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Metric vs Threshold Tradeoff')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df
```

---

## 🎯 Model Tuning

### Hyperparameter Tuning with Grid Search

```python
from sklearn.model_selection import GridSearchCV

def tune_random_forest(X_train, y_train):
    """Grid search for optimal Random Forest parameters"""
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(rf, param_grid, cv=5, 
                              scoring='f1', n_jobs=-1, verbose=1)
    
    print("Running grid search (this may take a while)...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_
```

### Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

def evaluate_with_cross_validation(model, X, y, cv=5):
    """Robust evaluation using stratified k-fold"""
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    results = cross_validate(model, X, y, cv=skf, scoring=scoring)
    
    print(f"\n{cv}-Fold Cross Validation Results:")
    print("=" * 50)
    for metric in scoring.keys():
        scores = results[f'test_{metric}']
        print(f"{metric.upper():12} | Mean: {scores.mean():.4f} | Std: {scores.std():.4f}")
    
    return results
```

---

## 🚀 Real-time Deployment

### Flask API Deployment

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load pre-trained model
with open('phishing_detector_model.pkl', 'rb') as f:
    detector = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to classify email"""
    data = request.json
    email_text = data.get('email_text', '')
    
    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400
    
    result = detector.predict(email_text)
    
    return jsonify({
        'label': result['label'],
        'prediction': int(result['prediction']),
        'phishing_probability': float(result['phishing_probability']),
        'safe_probability': float(result['safe_probability']),
        'confidence': float(result['confidence'])
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch classification endpoint"""
    data = request.json
    emails = data.get('emails', [])
    
    results = []
    for email_text in emails:
        result = detector.predict(email_text)
        results.append({
            'email': email_text[:50],  # Preview
            'prediction': result['label'],
            'confidence': float(result['confidence'])
        })
    
    return jsonify({'results': results, 'total': len(results)})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'operational'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY phishing_detector.py .
COPY phishing_flask_api.py .
COPY phishing_detector_model.pkl .

EXPOSE 5000

CMD ["python", "phishing_flask_api.py"]
```

### Usage

```bash
# Build and run
docker build -t phishing-detector .
docker run -p 5000:5000 phishing-detector

# Test API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Verify your account now at hxxp://fake.tk"}'
```

---

## 📊 Advanced Metrics

### Per-Class Performance Analysis

```python
def detailed_performance_analysis(y_true, y_pred, y_pred_proba):
    """Detailed per-class analysis"""
    
    from sklearn.metrics import precision_recall_curve
    
    print("PHISHING CLASS PERFORMANCE")
    print("=" * 50)
    
    # For phishing (class 1)
    phishing_precision, phishing_recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    print(f"Precision-Recall curve generated")
    print(f"Max precision: {phishing_precision.max():.4f}")
    print(f"Max recall: {phishing_recall.max():.4f}")
    
    # Confusion at different thresholds
    print("\n" + "=" * 50)
    print("THRESHOLD PERFORMANCE")
    print("=" * 50)
    
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_thresh)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nThreshold: {threshold}")
        print(f"  Sensitivity: {tp/(tp+fn):.4f} (catch phishing)")
        print(f"  Specificity: {tn/(tn+fp):.4f} (avoid false alarms)")
        print(f"  FN Rate: {fn/(tp+fn):.4f} (phishing gets through)")
```

### Calibration Analysis

```python
from sklearn.calibration import calibration_curve

def analyze_prediction_calibration(y_true, y_pred_proba):
    """Check if predicted probabilities match actual frequencies"""
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # If curve deviates from diagonal, model may need calibration
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    print(f"Calibration Error: {calibration_error:.4f}")
    
    if calibration_error > 0.1:
        print("⚠️ Model probabilities may not be well-calibrated")
        print("Consider using Platt scaling or isotonic regression")
```

### Feature Importance Analysis

```python
def analyze_feature_importance(model, feature_names):
    """Show which features matter most"""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Get top features
        top_indices = np.argsort(importances)[-15:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(top_features, top_importances)
        plt.xlabel('Importance')
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Top Features:")
        for feat, imp in zip(top_features, top_importances):
            print(f"  {feat:30} {imp:.4f}")
```

---

## 🔄 Continuous Learning Pipeline

```python
class ContinuousLearningDetector:
    """Detector that learns from new emails"""
    
    def __init__(self, detector):
        self.detector = detector
        self.misclassified = []
        self.user_feedback = []
    
    def predict_with_logging(self, email_text):
        """Predict and log for analysis"""
        result = self.detector.predict(email_text)
        
        return {
            **result,
            'timestamp': pd.Timestamp.now(),
            'email_hash': hash(email_text) % 10**8
        }
    
    def add_user_feedback(self, email_hash, ground_truth, model_prediction):
        """Collect user feedback on predictions"""
        if ground_truth != model_prediction:
            self.misclassified.append({
                'email_hash': email_hash,
                'actual': ground_truth,
                'predicted': model_prediction,
                'timestamp': pd.Timestamp.now()
            })
    
    def retrain_on_feedback(self, emails, labels):
        """Retrain model with feedback"""
        if len(self.misclassified) > 10:
            print(f"Retraining on {len(self.misclassified)} corrected examples...")
            
            # Add corrected examples to training set
            feedback_emails = [e['email'] for e in self.misclassified]
            feedback_labels = [e['actual'] for e in self.misclassified]
            
            combined_emails = emails + feedback_emails
            combined_labels = labels + feedback_labels
            
            self.detector.train(combined_emails, combined_labels)
            
            self.misclassified = []
            print("✓ Model retrained successfully")
```

---

## 🎓 Performance Optimization

### Vectorization Optimization

```python
# SLOW: Extract features one by one
results = []
for email in emails:
    result = feature_extractor.extract_all_features(email)
    results.append(result)

# FAST: Use parallel processing
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(feature_extractor.extract_all_features)(email) 
    for email in emails
)
```

### Model Serialization for Deployment

```python
import pickle
import joblib

# Save model
joblib.dump(detector, 'phishing_detector_model.pkl', compress=3)

# Load model
detector = joblib.load('phishing_detector_model.pkl')

# Predict on new data
result = detector.predict("Verify your account now")
```

---

This advanced guide should help you extend and optimize the phishing detector for production use! 🚀

