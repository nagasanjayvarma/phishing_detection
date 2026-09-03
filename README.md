# Phishing Email Detection Model 🎯

A machine learning solution for detecting phishing emails using Scikit-learn, feature extraction, and ensemble learning methods.

---

## 🚀 Quick Start

### Installation
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Run the Model
```bash
python phishing_detector.py
```

This will:
- ✓ Load sample dataset (20 phishing + 20 legitimate emails)
- ✓ Extract features from emails
- ✓ Train 3 different ML models
- ✓ Evaluate performance with detailed metrics
- ✓ Generate visualizations (confusion matrices, ROC curves)
- ✓ Make predictions on test emails

---

## 📊 Model Architecture

### Three Classification Models

| Model | Type | Best For |
|-------|------|----------|
| **Logistic Regression** | Linear | Fast, interpretable, baseline |
| **Random Forest** | Ensemble | Best overall balance, robustness |
| **Gradient Boosting** | Ensemble | High accuracy, complex patterns |

---

## 🔍 Feature Extraction

### 1. URL Features (URL Analysis)
```
- has_url: Whether email contains any URLs
- num_urls: Count of URLs in email
- has_shortened_url: Detects bit.ly, tinyurl, shorturl
- has_suspicious_tld: Checks for suspicious domains (.tk, .ru, .ml, .ga)
- url_obfuscation: Detects obfuscated URLs (hxxp instead of http)
```

**Why it matters:** Phishing emails often use shortened or suspicious domains to hide malicious links.

### 2. Keyword Features (Content Analysis)
```
- phishing_keyword_count: Number of suspicious keywords detected
- phishing_keyword_score: Weighted score of suspicious terms
- urgency_indicator: Presence of time-sensitive pressure ("urgent", "immediate")
- capitalization_ratio: Excessive use of capital letters
```

**Suspicious Keywords Tracked:**
- Account-related: verify, confirm, update, password, credential
- Action-based: click, authenticate, authorize, reactivate
- Threat-based: urgent, immediate, suspended, limited, compromised
- Emotional: alert, suspicious, unusual, malicious

### 3. Structural Features (Email Format Analysis)
```
- email_length: Total length of email body
- has_multiple_exclamations: More than 2 exclamation marks
- has_all_caps_words: Multiple words in ALL CAPS (3+ letters)
- has_suspicious_email: Noreply, donotreply in sender
- word_count: Number of words in email
```

### 4. Text Features (TF-IDF Vectorization)
```
- TF-IDF (Top 50 features): Numerical representation of word importance
- Unigrams + Bigrams: Single words and word pairs
- Automatically identifies discriminative terms
```

**Total Features: 50+ numerical features per email**

---

## 📈 Evaluation Metrics

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Overall correctness of the model. **Target: >95%**

### Sensitivity (Recall) ⚠️ CRITICAL
```
Sensitivity = TP / (TP + FN)
```
- **What it measures:** Of all phishing emails, how many did we catch?
- **Why it matters:** Missing a phishing email (False Negative) is dangerous
- **Target: >95%** (Catch as many phishing emails as possible)

### Specificity
```
Specificity = TN / (TN + FP)
```
- **What it measures:** Of all legitimate emails, how many were correctly identified?
- **Why it matters:** Too many false positives → users get frustrated
- **Target: >90%** (Minimize false alarms)

### Precision
```
Precision = TP / (TP + FP)
```
- **What it measures:** Of all emails marked as phishing, how many actually are?
- **Why it matters:** Helps understand false positive rate
- **Target: >90%**

### ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
```
AUC measures the model's ability to distinguish between classes
Range: 0 to 1 (1.0 = perfect classification)
Target: >0.95
```

---

## 🎯 Confusion Matrix Explained

```
                 PREDICTED
             PHISHING    SAFE
ACTUAL  PHISHING   TP      FN  ⚠️ Miss! User gets phished
        SAFE       FP      TN
              ⚠️ False alarm!
```

| Metric | Description | Impact |
|--------|-------------|--------|
| **TP (True Positive)** | Correctly identified phishing | ✓ Good |
| **TN (True Negative)** | Correctly identified safe email | ✓ Good |
| **FP (False Positive)** | Legitimate marked as phishing | ⚠️ Annoying (blocks real emails) |
| **FN (False Negative)** | Phishing marked as safe | 🚨 CRITICAL (user gets attacked) |

---

## 🔧 Customization Guide

### Adding Your Own Dataset

```python
# Replace generate_phishing_dataset() with your data:

def load_custom_dataset():
    phishing_emails = [...]  # List of phishing emails
    legitimate_emails = [...] # List of legitimate emails
    
    labels = [1]*len(phishing_emails) + [0]*len(legitimate_emails)
    emails = phishing_emails + legitimate_emails
    
    return emails, labels

# Use it:
emails, labels = load_custom_dataset()
detector = PhishingDetector()
detector.train(emails, labels)
```

### Adding Custom Keywords

Edit the `PHISHING_KEYWORDS` dictionary:

```python
PHISHING_KEYWORDS = {
    'verify': 5,           # keyword: weight (higher = more important)
    'confirm': 4,
    'your_keyword': 6,     # Add yours here
}
```

### Tuning Model Parameters

```python
# RandomForestClassifier parameters
rf_model = RandomForestClassifier(
    n_estimators=200,      # More trees = more robust but slower
    max_depth=15,          # Limit tree depth to prevent overfitting
    min_samples_split=5,   # Min samples per split
    random_state=42
)
```

---

## 📊 Output Files

When you run the model, it generates:

### Console Output
- Dataset statistics
- Feature extraction progress
- Model training status
- Detailed evaluation metrics
- Sample predictions

### Visualization Files

1. **confusion_matrices.png**
   - Heatmaps for all 3 models
   - Shows TP, TN, FP, FN for each
   - Includes accuracy scores

2. **roc_curves.png**
   - ROC curve for each model
   - Shows discrimination ability
   - AUC scores displayed

3. **model_comparison.png**
   - Bar chart comparing all metrics
   - Easy side-by-side comparison
   - Identifies best model

---

## 💡 How to Use in Practice

### Basic Prediction
```python
detector = PhishingDetector()
detector.train(emails, labels)

result = detector.predict("Urgent: Verify your account at hxxp://fake.tk")

print(result['label'])                    # "🚨 PHISHING"
print(result['confidence'])               # 0.95 (95% confident)
print(result['phishing_probability'])     # 0.95
```

### Deployment Workflow
```
1. Train model on historical email data
2. Set classification threshold (default: 0.5)
3. For production, adjust threshold based on business needs:
   - Threshold 0.7: Fewer alerts, might miss some phishing
   - Threshold 0.3: More alerts, catches more phishing but annoying
4. Monitor predictions and retrain periodically
5. Use sensitivity/specificity metrics to tune
```

---

## ⚠️ Important Considerations

### False Negatives are Worse
- **False Negative (FN):** Phishing email gets through = user gets attacked 🚨
- **False Positive (FP):** Legitimate email blocked = user annoyed 😐
- **Always prioritize sensitivity** (catching phishing) over specificity

### Dataset Quality Matters
- Model trained on 20+20 emails is for **demonstration only**
- Production models need:
  - **1000+ samples** minimum
  - **Balanced classes** (equal phishing/legitimate)
  - **Recent examples** (phishing evolves)
  - **Diverse domains** (emails, domains, industries)

### Adversarial Phishing
- Attackers aware of ML detection methods
- They adapt by:
  - Avoiding obvious keywords
  - Using legitimate-looking URLs
  - Mimicking legitimate formatting
- Requires **continuous model updates**

### Feature Limitations
- Text-based features only
- Doesn't analyze:
  - Image-based phishing
  - Zero-day exploits
  - Executive impersonation (requires context)
  - Domain reputation (requires external data)

---

## 🔐 Security Best Practices

### When Using This Model
1. **Never use alone** - Combine with:
   - Spam filters
   - Domain authentication (SPF, DKIM, DMARC)
   - User training
   - Endpoint security
   - Link preview/reputation checking

2. **Regular Updates**
   - Retrain monthly with new phishing samples
   - Monitor model drift
   - A/B test different thresholds
   - Track false positive rates

3. **Explainability**
   - Show users why email was flagged
   - Provide whitelist mechanism
   - Log predictions for audit

---

## 📚 Feature Importance

To see which features matter most (Random Forest):

```python
feature_importance = detector.models['Random Forest'].feature_importances_
# Top 5 most important features
top_indices = np.argsort(feature_importance)[-5:][::-1]
```

Typically most important features:
1. Phishing keyword score
2. Urgency indicators
3. URL characteristics
4. Capitalization ratio
5. TF-IDF terms (specific words)

---

## 🎓 Learning Resources

### Key Concepts
- **Feature Engineering:** Extract meaningful info from raw text
- **Vectorization:** Convert text to numbers (TF-IDF)
- **Ensemble Learning:** Combine multiple models for robustness
- **ROC-AUC:** Evaluate classification at different thresholds

### Improvements to Try
1. **Deep Learning:** Use LSTM/BERT for better text understanding
2. **Header Analysis:** Parse email headers for SPF, DKIM failures
3. **External Data:** Check domain reputation, URL blacklists
4. **Multi-modal:** Analyze images, attachments, metadata
5. **User Context:** Account for user-specific patterns

---

## 📧 Sample Phishing Indicators Detected

```
✓ Urgent language + Shortened URLs = 🚨 High Risk
✓ Password keywords + Suspicious TLD = 🚨 High Risk
✓ Multiple capitalization + Action requests = 🚨 High Risk
✓ Known banking keywords + URL obfuscation = 🚨 High Risk
✓ Formal greetings + Personal content = ✓ Likely Safe
✓ Project updates + No URLs = ✓ Likely Safe
✓ Colleague communication + No urgency = ✓ Likely Safe
```

---

## 🤝 Contributing

To improve this model:
1. Add more training samples
2. Implement new feature extractors
3. Test different algorithms
4. Validate against real-world phishing datasets (PhishTank, APWG)
5. Create industry-specific models

---

## 📝 License & Disclaimer

**For Educational Use Only**

This model is designed for learning cybersecurity concepts. 

- ⚠️ Not suitable for critical production systems without extensive validation
- ⚠️ Phishing attacks are constantly evolving - models need regular updates
- ⚠️ False negatives can have serious security implications
- ⚠️ Always combine with other security measures

---

## 📞 Troubleshooting

### Model Accuracy Low
- **Cause:** Small dataset size
- **Fix:** Collect more training examples (1000+)

### Too Many False Positives
- **Cause:** Threshold too aggressive
- **Fix:** Increase threshold from 0.5 to 0.6-0.7

### Too Many False Negatives
- **Cause:** Missing phishing variants
- **Fix:** Add more phishing samples, retrain

### Slow Training
- **Cause:** Too many features or large dataset
- **Fix:** Reduce `max_features` in TfidfVectorizer

---

## ✨ Happy Phishing Hunting! 🎣🚫

Use this tool responsibly to improve email security! 🔒
