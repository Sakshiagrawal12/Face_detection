# scripts/evaluate_model.py
"""
Script 04: Model Evaluation
Run this after training to evaluate model performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from config import BEST_MODEL_PATH, OUTPUT_PATH, CATEGORIES, MODELS_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_files():
    """Check if required files exist"""
    if not os.path.exists(BEST_MODEL_PATH):
        logging.error(f"Model not found at: {BEST_MODEL_PATH}")
        print("Please train the model first: python main.py --mode train")
        return False
    
    test_files = ['X_test.npy', 'y_test.npy']
    missing_files = [f for f in test_files if not os.path.exists(os.path.join(OUTPUT_PATH, f))]
    
    if missing_files:
        logging.error(f"Missing test files: {missing_files}")
        print("Please run preprocessing first: python main.py --mode preprocess")
        return False
    
    return True

def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_test, y_pred_prob, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc

def plot_per_class_performance(metrics_df, save_path):
    """Plot per-class performance metrics"""
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(metrics_df['Class']))
    width = 0.2
    
    plt.bar(x - width, metrics_df['Precision'], width, label='Precision', color='skyblue')
    plt.bar(x, metrics_df['Recall'], width, label='Recall', color='lightgreen')
    plt.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='lightcoral')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    plt.xticks(x, metrics_df['Class'])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(metrics_df['Precision']):
        plt.text(i - width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(metrics_df['Recall']):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(metrics_df['F1-Score']):
        plt.text(i + width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_misclassifications(X_test, y_test, y_pred, y_pred_prob, save_path):
    """Analyze misclassified examples"""
    misclassified = np.where(y_test != y_pred)[0]
    
    if len(misclassified) == 0:
        print("\n✅ No misclassifications found! Perfect model!")
        return
    
    print(f"\n🔍 Found {len(misclassified)} misclassified samples")
    
    # Analyze by class
    print("\nMisclassification Analysis:")
    for true_class in range(len(CATEGORIES)):
        class_mask = (y_test == true_class)
        misclassified_in_class = np.where((y_test == true_class) & (y_test != y_pred))[0]
        
        if len(misclassified_in_class) > 0:
            # See what they were misclassified as
            pred_classes = y_pred[misclassified_in_class]
            unique, counts = np.unique(pred_classes, return_counts=True)
            
            print(f"\n{ CATEGORIES[true_class]} misclassified as:")
            for u, c in zip(unique, counts):
                percentage = (c / len(misclassified_in_class)) * 100
                print(f"  → {CATEGORIES[u]}: {c} samples ({percentage:.1f}%)")
    
    # Save misclassified indices
    np.save(os.path.join(OUTPUT_PATH, 'misclassified_indices.npy'), misclassified)
    
    # Create a summary DataFrame
    misclassified_summary = []
    for idx in misclassified[:20]:  # Show first 20
        misclassified_summary.append({
            'True Label': CATEGORIES[y_test[idx]],
            'Predicted': CATEGORIES[y_pred[idx]],
            'Confidence': y_pred_prob[idx][y_pred[idx]] * 100
        })
    
    if misclassified_summary:
        df = pd.DataFrame(misclassified_summary)
        df.to_csv(os.path.join(OUTPUT_PATH, 'misclassified_samples.csv'), index=False)
        print(f"\n📊 First 20 misclassified samples saved to: misclassified_samples.csv")

def evaluate_model():
    """Evaluate the trained model with comprehensive metrics"""
    
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    # Check required files
    if not check_files():
        return
    
    try:
        # Load test data
        print("\n📂 Loading test data...")
        X_test = np.load(os.path.join(OUTPUT_PATH, 'X_test.npy'))
        y_test = np.load(os.path.join(OUTPUT_PATH, 'y_test.npy'))
        
        print(f"Test samples: {len(X_test)}")
        print(f"Class distribution: {np.bincount(y_test)}")
        
        # Load model
        print("\n📂 Loading trained model...")
        model = tf.keras.models.load_model(BEST_MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        # Evaluate
        print("\n📊 Evaluating on test set...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        
        # Predictions
        y_pred_prob = model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calculate all metrics
        accuracy_score_val = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Print summary metrics
        print("\n" + "="*60)
        print("📈 MODEL PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision (weighted): {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall (weighted): {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score (weighted): {f1:.4f} ({f1*100:.2f}%)")
        
        # Classification report
        print("\n📊 Classification Report:")
        report = classification_report(y_test, y_pred, target_names=CATEGORIES)
        print(report)
        
        # Save report
        with open(os.path.join(OUTPUT_PATH, 'classification_report.txt'), 'w') as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
            f.write(f"Recall: {recall:.4f} ({recall*100:.2f}%)\n")
            f.write(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, os.path.join(OUTPUT_PATH, 'confusion_matrix.png'))
        
        # ROC curve (for binary classification)
        if len(CATEGORIES) == 2:
            roc_auc = plot_roc_curve(y_test, y_pred_prob, 
                                     os.path.join(OUTPUT_PATH, 'roc_curve.png'))
            print(f"\n📈 ROC-AUC Score: {roc_auc:.4f}")
        
        # Per-class performance
        report_dict = classification_report(y_test, y_pred, 
                                           target_names=CATEGORIES, 
                                           output_dict=True)
        
        per_class_data = []
        for category in CATEGORIES:
            if category in report_dict:
                per_class_data.append({
                    'Class': category,
                    'Precision': report_dict[category]['precision'],
                    'Recall': report_dict[category]['recall'],
                    'F1-Score': report_dict[category]['f1-score'],
                    'Support': report_dict[category]['support']
                })
        
        if per_class_data:
            per_class_df = pd.DataFrame(per_class_data)
            plot_per_class_performance(per_class_df, 
                                      os.path.join(OUTPUT_PATH, 'per_class_performance.png'))
            
            # Save per-class metrics
            per_class_df.to_csv(os.path.join(OUTPUT_PATH, 'per_class_metrics.csv'), 
                               index=False)
        
        # Analyze misclassifications
        analyze_misclassifications(X_test, y_test, y_pred, y_pred_prob, OUTPUT_PATH)
        
        # Save all metrics to JSON
        import json
        metrics = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'accuracy_percent': float(accuracy * 100),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'test_samples': len(X_test),
            'class_distribution': {
                CATEGORIES[0]: int(np.sum(y_test == 0)),
                CATEGORIES[1]: int(np.sum(y_test == 1))
            }
        }
        
        if len(CATEGORIES) == 2:
            metrics['roc_auc'] = float(roc_auc)
        
        with open(os.path.join(OUTPUT_PATH, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n✅ Evaluation completed!")
        print(f"📁 Results saved to: {OUTPUT_PATH}")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_model()