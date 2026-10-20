import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

def load_data(npz_path="data/ravdess_features.npz"):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Feature file not found at {npz_path}. Run extract_features.py first.")
    
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    emotions = data['emotions']
    sentiments = data['sentiments']
    actors = data['actors']
    genders = data['genders']
    
    print(f"Loaded feature matrix: {X.shape}")
    print(f"Total samples: {len(emotions)}")
    return X, emotions, sentiments, actors, genders

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")

def train_and_evaluate(npz_path="data/ravdess_features.npz", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    
    X, emotions, sentiments, actors, genders = load_data(npz_path)
    
    # 1. Encoders & Scaler
    le_emotion = LabelEncoder()
    y_emotion = le_emotion.fit_transform(emotions)
    
    le_sentiment = LabelEncoder()
    y_sentiment = le_sentiment.fit_transform(sentiments)
    
    # Stratified Split (80% train, 20% test)
    X_train, X_test, y_emo_train, y_emo_test, y_sent_train, y_sent_test = train_test_split(
        X, y_emotion, y_sentiment,
        test_size=0.2,
        random_state=42,
        stratify=y_emotion
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------------
    # 2. Train Sentiment Classifier (3 classes: Negative, Neutral, Positive)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("TRAINING SENTIMENT CLASSIFIER (3 Classes)")
    print("="*50)
    
    sentiment_models = {
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
    }
    
    best_sent_acc = 0.0
    best_sent_name = None
    best_sent_model = None
    
    for name, model in sentiment_models.items():
        model.fit(X_train_scaled, y_sent_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_sent_test, preds)
        f1 = f1_score(y_sent_test, preds, average='weighted')
        print(f"[{name}] Test Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")
        if acc > best_sent_acc:
            best_sent_acc = acc
            best_sent_name = name
            best_sent_model = model
            
    # Also evaluate an ensemble voter for sentiment
    print("\nBuilding Sentiment Ensemble (Voting Classifier)...")
    sent_voting = VotingClassifier(
        estimators=[
            ('lgb', sentiment_models['LightGBM']),
            ('xgb', sentiment_models['XGBoost']),
            ('et', sentiment_models['ExtraTrees']),
            ('mlp', sentiment_models['MLP'])
        ],
        voting='soft'
    )
    sent_voting.fit(X_train_scaled, y_sent_train)
    voting_preds = sent_voting.predict(X_test_scaled)
    voting_acc = accuracy_score(y_sent_test, voting_preds)
    voting_f1 = f1_score(y_sent_test, voting_preds, average='weighted')
    print(f"[Ensemble Voting] Test Accuracy: {voting_acc*100:.2f}% | F1-Score: {voting_f1:.4f}")
    
    if voting_acc >= best_sent_acc:
        best_sent_acc = voting_acc
        best_sent_name = "Ensemble_Voting"
        best_sent_model = sent_voting
        sent_final_preds = voting_preds
    else:
        sent_final_preds = best_sent_model.predict(X_test_scaled)
        
    print(f"\n>> Selected Best Sentiment Model: {best_sent_name} ({best_sent_acc*100:.2f}%)")
    sent_report = classification_report(
        y_sent_test, sent_final_preds,
        target_names=le_sentiment.classes_,
        output_dict=True
    )
    print(classification_report(y_sent_test, sent_final_preds, target_names=le_sentiment.classes_))
    
    plot_confusion_matrix(
        y_sent_test, sent_final_preds,
        classes=le_sentiment.classes_,
        title=f"Sentiment Confusion Matrix ({best_sent_name} - {best_sent_acc*100:.1f}%)",
        save_path=os.path.join(models_dir, "confusion_matrix_sentiment.png")
    )
    
    # -------------------------------------------------------------
    # 3. Train Emotion Classifier (8 classes)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("TRAINING EMOTION CLASSIFIER (8 Classes)")
    print("="*50)
    
    emotion_models = {
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=40,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=350,
            max_depth=25,
            random_state=42,
            n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
    }
    
    best_emo_acc = 0.0
    best_emo_name = None
    best_emo_model = None
    
    for name, model in emotion_models.items():
        model.fit(X_train_scaled, y_emo_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_emo_test, preds)
        f1 = f1_score(y_emo_test, preds, average='weighted')
        print(f"[{name}] Test Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")
        if acc > best_emo_acc:
            best_emo_acc = acc
            best_emo_name = name
            best_emo_model = model
            
    print("\nBuilding Emotion Ensemble (Voting Classifier)...")
    emo_voting = VotingClassifier(
        estimators=[
            ('lgb', emotion_models['LightGBM']),
            ('xgb', emotion_models['XGBoost']),
            ('et', emotion_models['ExtraTrees']),
            ('mlp', emotion_models['MLP'])
        ],
        voting='soft'
    )
    emo_voting.fit(X_train_scaled, y_emo_train)
    voting_emo_preds = emo_voting.predict(X_test_scaled)
    voting_emo_acc = accuracy_score(y_emo_test, voting_emo_preds)
    voting_emo_f1 = f1_score(y_emo_test, voting_emo_preds, average='weighted')
    print(f"[Ensemble Voting] Test Accuracy: {voting_emo_acc*100:.2f}% | F1-Score: {voting_emo_f1:.4f}")
    
    if voting_emo_acc >= best_emo_acc:
        best_emo_acc = voting_emo_acc
        best_emo_name = "Ensemble_Voting"
        best_emo_model = emo_voting
        emo_final_preds = voting_emo_preds
    else:
        emo_final_preds = best_emo_model.predict(X_test_scaled)
        
    print(f"\n>> Selected Best Emotion Model: {best_emo_name} ({best_emo_acc*100:.2f}%)")
    emo_report = classification_report(
        y_emo_test, emo_final_preds,
        target_names=le_emotion.classes_,
        output_dict=True
    )
    print(classification_report(y_emo_test, emo_final_preds, target_names=le_emotion.classes_))
    
    plot_confusion_matrix(
        y_emo_test, emo_final_preds,
        classes=le_emotion.classes_,
        title=f"Emotion Confusion Matrix ({best_emo_name} - {best_emo_acc*100:.1f}%)",
        save_path=os.path.join(models_dir, "confusion_matrix_emotion.png")
    )
    
    # -------------------------------------------------------------
    # 4. Save Models, Scaler, Encoders & Metrics
    # -------------------------------------------------------------
    sent_model_path = os.path.join(models_dir, "sentiment_model.joblib")
    emo_model_path = os.path.join(models_dir, "emotion_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    encoders_path = os.path.join(models_dir, "encoders.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")
    
    joblib.dump(best_sent_model, sent_model_path)
    joblib.dump(best_emo_model, emo_model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({
        'emotion_classes': le_emotion.classes_,
        'sentiment_classes': le_sentiment.classes_
    }, encoders_path)
    
    metrics = {
        'sentiment_model': best_sent_name,
        'sentiment_accuracy': float(best_sent_acc),
        'sentiment_report': sent_report,
        'emotion_model': best_emo_name,
        'emotion_accuracy': float(best_emo_acc),
        'emotion_report': emo_report,
        'feature_dim': int(X.shape[1]),
        'num_samples': int(X.shape[0])
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("\n" + "="*50)
    print("TRAINING COMPLETE & ASSETS SAVED SUCCESSFULLY!")
    print(f"- Sentiment Model: {sent_model_path} (Acc: {best_sent_acc*100:.2f}%)")
    print(f"- Emotion Model: {emo_model_path} (Acc: {best_emo_acc*100:.2f}%)")
    print(f"- Scaler: {scaler_path}")
    print(f"- Encoders: {encoders_path}")
    print(f"- Metrics: {metrics_path}")
    print("="*50)

if __name__ == "__main__":
    train_and_evaluate()
