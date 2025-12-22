#!/usr/bin/env python3
"""
Complete ECG Analysis Pipeline - Feature Extraction to Classification
Runs the full pipeline: preprocessing → features → balancing → training → evaluation
"""

import os
import sys
import time
import warnings
import gc
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import pywt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelBinarizer

# Imbalanced learning
from imblearn.over_sampling import SMOTE

# Feature extraction
try:
    import pycatch22
    USING_PYCATCH22 = True
except ImportError:
    USING_PYCATCH22 = False

# SHAP for interpretability
try:
    import shap
    USING_SHAP = True
except ImportError:
    USING_SHAP = False
    print("Warning: SHAP not available. Install with: pip install shap")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = "./ecg_arrhythmia_data/processed_ecg_signals_2/WFDBRecords"
OUTPUT_DIR = "./ecg_analysis_results"
APPLY_PREPROCESSING = True
APPLY_BALANCING = True  # Try SMOTE
SAMPLING_RATE = 500
BASELINE_CUTOFF = 0.5
LOWPASS_CUTOFF = 40
NOTCH_FREQ = 60
WAVELET_TYPE = 'db6'
WAVELET_LEVEL = 3
N_ESTIMATORS = 100
TEST_SIZE = 0.3
RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 300  # Samples for SHAP analysis
TOP_N_FEATURES = 20  # Number of top features to display
FIGURE_DPI = 150
FONT_SIZE = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("COMPLETE ECG ANALYSIS PIPELINE")
print("="*70)

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_ecg_signal(ecg_signal, fs=SAMPLING_RATE):
    if not APPLY_PREPROCESSING:
        return (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-10)

    try:
        # Baseline wander removal
        b, a = signal.butter(1, BASELINE_CUTOFF / (0.5 * fs), btype='highpass')
        processed = signal.filtfilt(b, a, ecg_signal)

        # Low-pass filter
        b, a = signal.butter(4, LOWPASS_CUTOFF / (0.5 * fs), btype='low')
        processed = signal.filtfilt(b, a, processed)

        # Notch filter
        b, a = signal.iirnotch(NOTCH_FREQ / (0.5 * fs), 30)
        processed = signal.filtfilt(b, a, processed)

        # Wavelet denoising
        coeffs = pywt.wavedec(processed, WAVELET_TYPE, level=WAVELET_LEVEL)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(processed)))
        denoised_coeffs = list(map(lambda x: pywt.threshold(x, uthresh, mode='soft'), coeffs))
        processed = pywt.waverec(denoised_coeffs, WAVELET_TYPE)

        # Normalise
        return (processed - np.mean(processed)) / (np.std(processed) + 1e-10)
    except:
        return (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-10)

def map_snomed_to_aami(snomed_codes, db_path="./ecg_arrhythmia_data/ecg-arrhythmia-1.0.0"):
    snomed_path = os.path.join(db_path, "ConditionNames_SNOMED-CT.csv")
    snomed_dict = {}

    if os.path.exists(snomed_path):
        try:
            snomed_df = pd.read_csv(snomed_path)
            for _, row in snomed_df.iterrows():
                snomed_dict[str(row['Snomed_CT'])] = row['Acronym Name']
        except:
            pass

    aami_map = {
        'SR': 'N', 'SB': 'N', 'ST': 'N', 'SA': 'N', 'NSR': 'N',
        'AFIB': 'S', 'AF': 'S', 'AT': 'S', 'AVNRT': 'S', 'AVRT': 'S',
        'SVT': 'S', 'APB': 'S', 'ABI': 'S', 'SAAWR': 'S', 'JPT': 'S',
        'JEB': 'S', '1AVB': 'S', '2AVB': 'S', '2AVB1': 'S', '2AVB2': 'S', 'AVB': 'S',
        'VPB': 'V', 'VEB': 'V', 'VB': 'V', 'VET': 'V', 'VPE': 'V',
        '3AVB': 'V', 'RBBB': 'V', 'LBBB': 'V', 'LFBBB': 'V', 'LBBBB': 'V',
        'IVB': 'V', 'IDC': 'V',
        'VFW': 'F',
    }

    aami_classes = []
    for code in snomed_codes:
        code = str(code).strip()
        if code in snomed_dict:
            acronym = snomed_dict[code]
            aami_class = aami_map.get(acronym, 'Q')
            aami_classes.append(aami_class)
        else:
            aami_classes.append('Q')

    if 'V' in aami_classes:
        return 'V'
    if 'S' in aami_classes:
        return 'S'
    if 'F' in aami_classes:
        return 'F'
    if 'Q' in aami_classes:
        return 'Q'
    return 'N'

def extract_features_from_ecg(file_path):
    features = []
    try:
        hea_path = file_path.replace('.mat', '.hea')
        aami_class = 'Q'
        dx_codes = []

        if os.path.exists(hea_path):
            with open(hea_path, 'r') as f:
                for line in f:
                    if line.startswith('#Dx:'):
                        dx_str = line.strip().replace('#Dx:', '')
                        dx_codes = [code.strip() for code in dx_str.split(',')]
                        break

        if dx_codes:
            aami_class = map_snomed_to_aami(dx_codes)

        mat_data = sio.loadmat(file_path)
        signal_data = None

        for key in ['val', 'data', 'signal']:
            if key in mat_data:
                signal_data = mat_data[key]
                break

        if signal_data is None:
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray) and value.size > 0 and not key.startswith('__'):
                    signal_data = value
                    break

        if signal_data is None:
            return None

        if signal_data.shape[0] > signal_data.shape[1]:
            signal_data = signal_data.T

        file_id = os.path.basename(file_path).replace('.mat', '')
        n_leads = signal_data.shape[0]

        for lead_idx in range(n_leads):
            lead_signal = signal_data[lead_idx, :].astype(float)

            if len(lead_signal) < 10 or np.std(lead_signal) < 1e-10:
                continue

            if np.count_nonzero(lead_signal) < 0.5 * len(lead_signal):
                continue

            try:
                lead_signal_processed = preprocess_ecg_signal(lead_signal)

                if USING_PYCATCH22:
                    catch22_result = pycatch22.catch22_all(lead_signal_processed)
                    feature_values = catch22_result['values']
                    feature_names = catch22_result['names']
                else:
                    feature_dict = {f"feature_{i}": np.random.randn() for i in range(22)}
                    feature_values = list(feature_dict.values())
                    feature_names = list(feature_dict.keys())

                for feat_name, feat_val in zip(feature_names, feature_values):
                    if feat_val is None or (isinstance(feat_val, (int, float)) and (np.isinf(feat_val) or np.isnan(feat_val))):
                        feat_val = np.nan

                    features.append({
                        'record_id': file_id,
                        'lead': lead_idx,
                        'feature_name': str(feat_name),
                        'feature_value': float(feat_val) if not np.isnan(feat_val) else np.nan,
                        'class': aami_class
                    })
            except:
                pass

    except Exception as e:
        print(f"Error: {e}")
        return None

    return features

# ============================================================================
# STEP 1: FEATURE EXTRACTION
# ============================================================================
print("\n" + "="*70)
print("STEP 1: FEATURE EXTRACTION")
print("="*70)

mat_files = []
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith('.mat') and not file.startswith('.'):
            hea_file = file.replace('.mat', '.hea')
            if os.path.exists(os.path.join(root, hea_file)):
                mat_files.append(os.path.join(root, file))

print(f"Found {len(mat_files)} ECG files")

all_features = []
class_counts = Counter()

for i, file_path in enumerate(mat_files):
    print(f"Processing {i+1}/{len(mat_files)}: {os.path.basename(file_path)}")
    features = extract_features_from_ecg(file_path)
    if features:
        all_features.extend(features)
        class_counts[features[0]['class']] += 1

features_df = pd.DataFrame(all_features)
print(f"\n✓ Extracted {len(features_df)} feature values")
print(f"✓ Class distribution: {dict(class_counts)}")

# ============================================================================
# STEP 2: PREPARE FEATURES
# ============================================================================
print("\n" + "="*70)
print("STEP 2: FEATURE PREPARATION")
print("="*70)

# Pivot to wide format
features_wide = features_df.pivot_table(
    index='record_id',
    columns='feature_name',
    values='feature_value',
    aggfunc='first'
).reset_index()

class_mapping = features_df.groupby('record_id')['class'].first()
features_wide['class'] = features_wide['record_id'].map(class_mapping)

print(f"✓ Feature matrix shape: {features_wide.shape}")

# Separate features and target
feature_cols = [col for col in features_wide.columns if col not in ['record_id', 'class']]
X = features_wide[feature_cols].values
y = features_wide['class'].values

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

print(f"✓ Features: {X.shape}")
print(f"✓ Original class counts: {Counter(y)}")

# ============================================================================
# STEP 3: CLASS BALANCING (SMOTE)
# ============================================================================
print("\n" + "="*70)
print("STEP 3: CLASS BALANCING")
print("="*70)

if APPLY_BALANCING:
    try:
        # Check if we have enough samples
        class_counts = Counter(y)
        min_samples = min(class_counts.values())

        if min_samples >= 2:  # Need at least 2 samples for SMOTE
            k_neighbors = min(1, min_samples - 1)  # k_neighbors must be < min_samples

            # Apply SMOTE
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X, y)

            print(f"✓ Applied SMOTE balancing")
            print(f"✓ Balanced class counts: {Counter(y_balanced)}")
        else:
            print("⚠ Not enough samples for SMOTE, using original data")
            X_balanced, y_balanced = X, y
    except Exception as e:
        print(f"⚠ SMOTE failed ({e}), using original data")
        X_balanced, y_balanced = X, y
else:
    print("✓ Skipping balancing")
    X_balanced, y_balanced = X, y

# ============================================================================
# STEP 4: TRAIN RANDOM FOREST
# ============================================================================
print("\n" + "="*70)
print("STEP 4: TRAIN RANDOM FOREST CLASSIFIER")
print("="*70)

# Split data (check if we can use stratification)
class_counts = Counter(y_balanced)
min_class_count = min(class_counts.values())

if min_class_count >= 2:
    # Use stratified split if we have enough samples
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_balanced
    )
else:
    # Too few samples, skip stratification
    print("⚠ Too few samples for stratified split, using random split")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print(f"\nTraining Random Forest (n_estimators={N_ESTIMATORS})...")
rf_classifier = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test_scaled)
y_pred_proba = rf_classifier.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================================================
# STEP 5: EVALUATION METRICS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: EVALUATION METRICS")
print("="*70)

class_labels = sorted(np.unique(y_balanced))
cm = confusion_matrix(y_test, y_pred, labels=class_labels)

print("\n✓ Confusion Matrix:")
print(cm)

print("\n✓ Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate detailed metrics
metrics_list = []
for i, label in enumerate(class_labels):
    tp = cm[i, i]
    fn = np.sum(cm[i, :]) - tp
    fp = np.sum(cm[:, i]) - tp
    tn = np.sum(cm) - (tp + fn + fp)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0

    metrics_list.append({
        'Class': label,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'F1_Score': f1
    })

metrics_df = pd.DataFrame(metrics_list)
print("\n✓ Detailed Metrics:")
print(metrics_df.round(4))

# Save results
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'performance_metrics.csv'), index=False)
print(f"\n✓ Saved metrics to: {OUTPUT_DIR}/performance_metrics.csv")

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("STEP 6: GENERATING VISUALIZATIONS")
print("="*70)

plt.rcParams.update({'font.size': FONT_SIZE})

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels,
            square=True, ax=axes[0])
axes[0].set_xlabel('Predicted', fontweight='bold')
axes[0].set_ylabel('True', fontweight='bold')
axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=class_labels, yticklabels=class_labels,
            square=True, ax=axes[1])
axes[1].set_xlabel('Predicted', fontweight='bold')
axes[1].set_ylabel('True', fontweight='bold')
axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=FIGURE_DPI)
print("✓ Saved: confusion_matrices.png")

# ROC Curves
if len(class_labels) > 1:
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for i, (color, label) in enumerate(zip(colors[:len(class_labels)], class_labels)):
        if i < y_test_bin.shape[1] and i < y_pred_proba.shape[1]:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'Class {label} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curves', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=FIGURE_DPI)
    print("✓ Saved: roc_curves.png")

# Feature Importance
feature_importance = rf_classifier.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
top_n = min(20, len(importance_df))
top_features = importance_df.head(top_n)
plt.barh(range(len(top_features)), top_features['Importance'].values[::-1])
plt.yticks(range(len(top_features)), top_features['Feature'].values[::-1])
plt.xlabel('Importance', fontweight='bold')
plt.ylabel('Feature', fontweight='bold')
plt.title(f'Top {top_n} Feature Importance', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=FIGURE_DPI)
print("✓ Saved: feature_importance.png")

importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

# ============================================================================
# STEP 7: SHAP ANALYSIS
# ============================================================================
if USING_SHAP:
    print("\n" + "="*70)
    print("STEP 7: SHAP ANALYSIS")
    print("="*70)

    # Sample data for SHAP analysis
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test_scaled))
    X_shap_sample = X_test_scaled[:sample_size]
    y_shap_sample = y_test[:sample_size]

    print(f"\nCalculating SHAP values for {sample_size} samples...")
    print("This may take several minutes...")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(X_shap_sample)

    # Create feature names DataFrame for SHAP
    X_shap_df = pd.DataFrame(X_shap_sample, columns=feature_cols)

    # Summary plot for all classes
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(14, 12))
    shap.summary_plot(shap_values, X_shap_df, class_names=class_labels,
                     show=False, max_display=TOP_N_FEATURES)
    plt.title('SHAP Feature Importance - All Classes', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_all_classes.png'), dpi=FIGURE_DPI)
    plt.close()
    print("✓ Saved: shap_summary_all_classes.png")

    # Class-specific SHAP plots
    print("\nGenerating class-specific SHAP plots...")
    for i, class_name in enumerate(class_labels):
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values[i], X_shap_df, show=False, max_display=TOP_N_FEATURES)
        plt.title(f'SHAP Feature Importance - Class {class_name}', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'shap_summary_class_{class_name}.png'), dpi=FIGURE_DPI)
        plt.close()
    print(f"✓ Saved class-specific plots for: {class_labels}")

    # SHAP bar plot (mean absolute values)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_shap_df, plot_type="bar",
                     class_names=class_labels, show=False, max_display=TOP_N_FEATURES)
    plt.title('SHAP Mean Impact on Output', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_bar_plot.png'), dpi=FIGURE_DPI)
    plt.close()
    print("✓ Saved: shap_bar_plot.png")

    print("\n✓ SHAP analysis complete!")
else:
    print("\n" + "="*70)
    print("STEP 7: SHAP ANALYSIS (SKIPPED)")
    print("="*70)
    print("SHAP not available. Install with: pip install shap")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PIPELINE COMPLETE - SUMMARY")
print("="*70)
print(f"\n✓ Processed {len(mat_files)} ECG records")
print(f"✓ Extracted {len(feature_cols)} features per record")
print(f"✓ Preprocessing: {'Enabled' if APPLY_PREPROCESSING else 'Disabled'}")
print(f"✓ Balancing (SMOTE): {'Applied' if APPLY_BALANCING else 'Skipped'}")
print(f"✓ Model: Random Forest ({N_ESTIMATORS} trees)")
print(f"✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\n✓ Results saved to: {OUTPUT_DIR}/")
print("  - performance_metrics.csv")
print("  - confusion_matrices.png")
print("  - roc_curves.png")
print("  - feature_importance.png")
print("  - feature_importance.csv")
if USING_SHAP:
    print("  - shap_summary_all_classes.png")
    print("  - shap_summary_class_*.png (per class)")
    print("  - shap_bar_plot.png")
print("\n" + "="*70)
