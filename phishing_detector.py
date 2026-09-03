"""
Phishing Email Detection Model
================================
A machine learning model that classifies emails as Phishing or Safe
using feature extraction and Scikit-learn classifiers.

Features extracted:
- Suspicious keywords and phrases
- URL characteristics
- Email header anomalies
- Urgency indicators
- HTML/formatting patterns
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import re
from collections import Counter

# ======================== SAMPLE DATASET ========================

def generate_phishing_dataset():
    """Generate a sample dataset of phishing and legitimate emails"""
    
    # Phishing emails
    phishing_emails = [
        "Urgent: Verify your PayPal account immediately to avoid suspension. Click here: bit.ly/verify",
        "Dear Customer, Update your banking credentials now. Your account has been compromised. Click link below",
        "ALERT: Suspicious activity detected. Confirm identity: hxxp://secure-verify.tk/login",
        "Apple ID Verification Required - Click here to confirm your password",
        "Your Amazon account will be closed. Verify now: shorturl.com/amazon-verify",
        "IRS Tax Refund Pending - Claim your $500 refund at taxrefund-claim.ru",
        "RE: Urgent wire transfer approval needed. Please authorize immediately",
        "Your Netflix subscription has expired. Renew now with updated payment info",
        "Bank Security Alert: Confirm your SSN and credit card details",
        "Congratulations! You've won $1,000,000. Claim your prize at winners-instant.tk",
        "Microsoft Security Alert - Your password will expire. Update now",
        "PayPal Account Limited - Verify account ownership within 24 hours",
        "Executive Wire Transfer Authorization Needed - URGENT",
        "Your LinkedIn profile verification failed. Reactivate immediately",
        "FedEx Delivery Notification - Update shipping information at tracking-verify.ru",
        "Google Account Recovery - Confirm recovery email immediately",
        "Your eBay account has been suspended for unusual activity",
        "Chase Bank: Account security update required within 48 hours",
        "Dropbox Sync Failure - Re-authenticate your credentials now",
        "Instagram Account Verification - Unusual login detected from another location",
    ]
    
    # Legitimate emails
    legitimate_emails = [
        "Hi John, I hope you're doing well. Let's schedule a meeting for next Tuesday.",
        "Thank you for your purchase. Your order #12345 has been shipped.",
        "Meeting Rescheduled: Budget Review moved to Thursday at 2 PM",
        "Welcome to our newsletter! Check out this month's featured content.",
        "Your gym membership renewal is due next month.",
        "Project Update: Design phase completed, moving to development.",
        "Conference Registration Confirmation - See you at TechConf 2024",
        "Birthday reminder: Don't forget Sarah's birthday party this Friday!",
        "Documentation update available for our software",
        "Team lunch planning - What's everyone's preference for Friday?",
        "Quarterly performance review scheduled for March 15th",
        "New feature release: Check out what's new in version 3.0",
        "Subscription renewal - Your annual plan renews on the 15th",
        "Training session reminder: Security best practices workshop tomorrow at 10 AM",
        "Client feedback on last week's presentation",
        "HR Policy Update: New remote work guidelines",
        "Project milestone achieved - Great job everyone!",
        "Weekly status report - See attached for details",
        "Lunch group meeting next week - responding to poll",
        "Thanks for the great collaboration on the proposal",
    ]
    
    # Create labels
    phishing_labels = [1] * len(phishing_emails)
    legitimate_labels = [0] * len(legitimate_emails)
    
    # Combine and return
    emails = phishing_emails + legitimate_emails
    labels = phishing_labels + legitimate_labels
    
    return emails, labels


# ======================== FEATURE EXTRACTION ========================

class EmailFeatureExtractor:
    """Extract features from email content"""
    
    # Suspicious keywords
    PHISHING_KEYWORDS = {
        'verify': 5, 'confirm': 4, 'urgent': 5, 'immediate': 5,
        'suspended': 5, 'limited': 4, 'update': 3, 'click': 3,
        'credential': 5, 'password': 4, 'security': 2, 'unusual': 3,
        'unusual activity': 5, 'account': 2, 'claim': 4, 'alert': 3,
        'expire': 3, 'authorization': 3, 'suspicious': 5, 'compromise': 5,
        'reactivate': 4, 'authenticate': 3, 'malicious': 5, 'restore': 3
    }
    
    URGENCY_KEYWORDS = [
        'urgent', 'immediately', 'asap', 'within 24 hours', 'within 48 hours',
        'don\'t delay', 'limited time', 'act now', 'expire', 'suspension'
    ]
    
    def __init__(self):
        self.features = {}
    
    def extract_url_features(self, text):
        """Extract features from URLs in email"""
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                          text, re.IGNORECASE)
        urls.extend(re.findall(r'bit\.ly|short\.?url|tinyurl|hxxp', text, re.IGNORECASE))
        
        features = {
            'has_url': 1 if urls else 0,
            'num_urls': len(urls),
            'has_shortened_url': 1 if any('bit.ly' in u or 'short' in u.lower() or 'tinyurl' in u.lower() for u in urls) else 0,
            'has_suspicious_tld': 0,
            'url_obfuscation': 1 if 'hxxp' in text else 0
        }
        
        # Check for suspicious TLDs
        suspicious_tlds = ['.tk', '.ru', '.ml', '.ga', '.gq']
        for url in urls:
            for tld in suspicious_tlds:
                if tld in url.lower():
                    features['has_suspicious_tld'] = 1
        
        return features
    
    def extract_keyword_features(self, text):
        """Extract features based on suspicious keywords"""
        text_lower = text.lower()
        
        features = {
            'phishing_keyword_count': 0,
            'phishing_keyword_score': 0,
            'urgency_indicator': 0,
            'capitalization_ratio': 0
        }
        
        # Count phishing keywords
        for keyword, weight in self.PHISHING_KEYWORDS.items():
            if keyword in text_lower:
                features['phishing_keyword_count'] += 1
                features['phishing_keyword_score'] += weight
        
        # Check urgency
        for urgency_word in self.URGENCY_KEYWORDS:
            if urgency_word in text_lower:
                features['urgency_indicator'] = 1
                break
        
        # Capitalization ratio (phishing emails tend to overuse caps)
        if len(text) > 0:
            caps_count = sum(1 for c in text if c.isupper())
            features['capitalization_ratio'] = caps_count / len(text)
        
        return features
    
    def extract_structural_features(self, text):
        """Extract structural features"""
        features = {
            'email_length': len(text),
            'has_multiple_exclamations': 1 if text.count('!') > 2 else 0,
            'has_all_caps_words': 1 if len(re.findall(r'\b[A-Z]{3,}\b', text)) > 2 else 0,
            'has_suspicious_email': self._check_email_patterns(text),
            'word_count': len(text.split())
        }
        return features
    
    def _check_email_patterns(self, text):
        """Check for suspicious email patterns"""
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        suspicious_patterns = ['noreply', 'donotreply', 'fake', 'temp', 'mail']
        
        for email in emails:
            for pattern in suspicious_patterns:
                if pattern in email.lower():
                    return 1
        return 0
    
    def extract_all_features(self, text):
        """Extract all features from email text"""
        url_features = self.extract_url_features(text)
        keyword_features = self.extract_keyword_features(text)
        structural_features = self.extract_structural_features(text)
        
        all_features = {**url_features, **keyword_features, **structural_features}
        return all_features


# ======================== PHISHING DETECTOR MODEL ========================

class PhishingDetector:
    """Complete phishing detection pipeline"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_extractor = EmailFeatureExtractor()
        self.tfidf = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.X_test = None
        self.y_test = None
        
    def extract_features(self, emails):
        """Extract all features from emails"""
        extracted_features = []
        
        for email in emails:
            features = self.feature_extractor.extract_all_features(email)
            extracted_features.append(features)
        
        return pd.DataFrame(extracted_features)
    
    def prepare_data(self, emails, labels):
        """Prepare features and labels for training"""
        # Extract structured features
        feature_df = self.extract_features(emails)
        
        # Extract TF-IDF features
        tfidf_features = self.tfidf.fit_transform(emails)
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        # Combine all features
        X = pd.concat([feature_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        y = np.array(labels)
        
        return X, y
    
    def train(self, emails, labels):
        """Train the phishing detection model"""
        print("=" * 60)
        print("PHISHING EMAIL DETECTION MODEL")
        print("=" * 60)
        print(f"\nDataset size: {len(emails)} emails")
        print(f"Phishing emails: {sum(labels)}")
        print(f"Legitimate emails: {len(labels) - sum(labels)}")
        
        # Prepare data
        print("\n[1/4] Extracting features...")
        X, y = self.prepare_data(emails, labels)
        
        print(f"Total features: {X.shape[1]}")
        print(f"Feature columns: {list(X.columns[:10])}...")
        
        # Split data
        print("\n[2/4] Splitting data (80/20 train/test)...")
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        print("[3/4] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train multiple models
        print("[4/4] Training multiple models...\n")
        
        # Model 1: Logistic Regression
        print("  • Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        lr_model.fit(X_train_scaled, y_train)
        self.models['Logistic Regression'] = lr_model
        
        # Model 2: Random Forest
        print("  • Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        self.models['Random Forest'] = rf_model
        
        # Model 3: Gradient Boosting
        print("  • Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        gb_model.fit(X_train_scaled, y_train)
        self.models['Gradient Boosting'] = gb_model
        
        print("\n✓ Training complete!")
    
    def evaluate(self):
        """Evaluate all models"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        for model_name, model in self.models.items():
            print(f"\n{'─' * 60}")
            print(f"Model: {model_name}")
            print(f"{'─' * 60}")
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            print(f"\nConfusion Matrix:")
            print(f"  True Negatives:  {tn}  (Correctly identified safe)")
            print(f"  False Positives: {fp}  (Legitimate marked as phishing)")
            print(f"  False Negatives: {fn}  (Phishing marked as safe) ⚠")
            print(f"  True Positives:  {tp}  (Correctly identified phishing)")
            
            # Additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            print(f"\nDerived Metrics:")
            print(f"  Sensitivity (Recall):  {sensitivity:.4f}")
            print(f"  Specificity:           {specificity:.4f}")
            print(f"  Precision:             {precision:.4f}")
            
            # Classification Report
            print(f"\nDetailed Classification Report:")
            print(classification_report(self.y_test, y_pred, 
                                       target_names=['Safe', 'Phishing']))
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'cm': cm,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision
            }
    
    def predict(self, email_text):
        """Predict if an email is phishing or safe"""
        # Use the best model (Random Forest)
        model = self.models['Random Forest']
        
        # Extract features
        feature_df = pd.DataFrame([self.feature_extractor.extract_all_features(email_text)])
        tfidf_features = self.tfidf.transform([email_text])
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                               columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        X = pd.concat([feature_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        label = "🚨 PHISHING" if prediction == 1 else "✓ SAFE"
        confidence = probability[prediction]
        
        return {
            'label': label,
            'prediction': prediction,
            'phishing_probability': probability[1],
            'safe_probability': probability[0],
            'confidence': confidence
        }
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, len(self.results), figsize=(15, 4))
        
        if len(self.results) == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = result['cm']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False, annot_kws={'size': 14})
            axes[idx].set_title(f'{model_name}\n(Accuracy: {result["accuracy"]:.2%})')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_xticklabels(['Safe', 'Phishing'])
            axes[idx].set_yticklabels(['Safe', 'Phishing'])
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrices saved to 'confusion_matrices.png'")
        plt.show()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for model_name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            roc_auc = result['roc_auc']
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Phishing Detection Models', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curves saved to 'roc_curves.png'")
        plt.show()
    
    def plot_model_comparison(self):
        """Compare models side by side"""
        metrics_data = {
            'Model': list(self.results.keys()),
            'Accuracy': [v['accuracy'] for v in self.results.values()],
            'Sensitivity': [v['sensitivity'] for v in self.results.values()],
            'Specificity': [v['specificity'] for v in self.results.values()],
            'Precision': [v['precision'] for v in self.results.values()],
            'ROC-AUC': [v['roc_auc'] for v in self.results.values()]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_metrics))
        width = 0.15
        
        for idx, metric in enumerate(['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'ROC-AUC']):
            ax.bar(x + idx*width, df_metrics[metric], width, label=metric, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df_metrics['Model'])
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Model comparison saved to 'model_comparison.png'")
        plt.show()


# ======================== MAIN EXECUTION ========================

if __name__ == "__main__":
    # Generate dataset
    emails, labels = generate_phishing_dataset()
    
    # Initialize and train detector
    detector = PhishingDetector(random_state=42)
    detector.train(emails, labels)
    
    # Evaluate models
    detector.evaluate()
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    detector.plot_confusion_matrices()
    detector.plot_roc_curves()
    detector.plot_model_comparison()
    
    # Test predictions on sample emails
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    test_emails = [
        "Urgent: Verify your account now at hxxp://fake-bank.tk/login",
        "Hi Sarah, Thanks for attending the meeting yesterday. See you next week!",
        "Click here immediately to confirm your PayPal credentials before suspension",
        "Project Update: We've completed the design phase and moving to development"
    ]
    
    for idx, email in enumerate(test_emails, 1):
        result = detector.predict(email)
        print(f"\nEmail {idx}:")
        print(f"  Content: {email[:60]}...")
        print(f"  Prediction: {result['label']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Phishing probability: {result['phishing_probability']:.4f}")
        print(f"  Safe probability: {result['safe_probability']:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete! Check outputs folder for visualizations.")
    print("=" * 60)
