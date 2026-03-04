# HEART DISEASE PREDICTION - COMPLETE CODE
# Comparing Logistic Regression vs Decision Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HEART DISEASE PREDICTION - LOGISTIC REGRESSION vs DECISION TREE")
print("="*80)

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# ============================================
# 1. LOAD DATA
# ============================================

print("\n--- Loading Data ---")
# Load your heart disease dataset
# Tries common locations first, then falls back to sample data
df = None
csv_candidates = [
    Path('/mnt/user-data/uploads/heart.csv'),
    script_dir / 'heart.csv',
    Path.cwd() / 'heart.csv',
]

for csv_path in csv_candidates:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"✓ Data loaded from: {csv_path}")
        break

if df is None:
    print("✗ Could not find heart.csv - using sample data")
    # If no file, create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(30, 80, 1000),
        'sex': np.random.randint(0, 2, 1000),
        'cp': np.random.randint(0, 4, 1000),
        'trestbps': np.random.randint(90, 200, 1000),
        'chol': np.random.randint(120, 400, 1000),
        'fbs': np.random.randint(0, 2, 1000),
        'restecg': np.random.randint(0, 3, 1000),
        'thalach': np.random.randint(70, 200, 1000),
        'exang': np.random.randint(0, 2, 1000),
        'oldpeak': np.random.uniform(0, 6, 1000),
        'slope': np.random.randint(0, 3, 1000),
        'ca': np.random.randint(0, 4, 1000),
        'thal': np.random.randint(0, 4, 1000),
        'target': np.random.randint(0, 2, 1000)
    })

print(f"Dataset shape: {df.shape}")
print(f"Features: {df.shape[1]-1}")
print(f"Samples: {df.shape[0]}")

# ============================================
# 2. DATA EXPLORATION
# ============================================

print("\n--- Data Overview ---")
print(df.head())
print("\n--- Data Info ---")
print(df.info())
print("\n--- Target Distribution ---")
print(df['target'].value_counts())
print(f"Disease cases: {df['target'].sum()}")
print(f"No disease cases: {len(df) - df['target'].sum()}")

# ============================================
# 3. PREPARE DATA
# ============================================

print("\n--- Preparing Data ---")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature Scaling (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Feature scaling completed")

# ============================================
# 4. MODEL 1: LOGISTIC REGRESSION
# ============================================

print("\n" + "="*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*80)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"\n✓ Logistic Regression Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

print("\nConfusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['No Disease', 'Disease']))

# ============================================
# 5. MODEL 2: DECISION TREE
# ============================================

print("\n" + "="*80)
print("MODEL 2: DECISION TREE")
print("="*80)

# Train Decision Tree (no scaling needed!)
dt_model = DecisionTreeClassifier(
    max_depth=5,           # Prevent overfitting
    min_samples_split=20,  # Min samples to split node
    min_samples_leaf=10,   # Min samples in leaf
    random_state=42
)

dt_model.fit(X_train, y_train)  # Note: Using original data, not scaled!

# Predictions
y_pred_dt = dt_model.predict(X_test)
y_pred_dt_proba = dt_model.predict_proba(X_test)[:, 1]

# Evaluate
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"\n✓ Decision Tree Accuracy: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")

print("\nConfusion Matrix:")
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, target_names=['No Disease', 'Disease']))

# Feature Importance
print("\n--- Feature Importance (Decision Tree) ---")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)
print(f"\nMost important feature: {feature_importance.iloc[0]['Feature']}")

# ============================================
# 6. MODEL COMPARISON
# ============================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

# Create comparison dataframe
comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [lr_accuracy, dt_accuracy],
    'True Positives': [cm_lr[1,1], cm_dt[1,1]],
    'True Negatives': [cm_lr[0,0], cm_dt[0,0]],
    'False Positives': [cm_lr[0,1], cm_dt[0,1]],
    'False Negatives': [cm_lr[1,0], cm_dt[1,0]]
})

print("\n" + comparison.to_string(index=False))

# Calculate additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score

lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

dt_precision = precision_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)

print("\n--- Detailed Metrics ---")
print(f"{'Metric':<20} {'Logistic Reg':<15} {'Decision Tree':<15}")
print("-" * 50)
print(f"{'Accuracy':<20} {lr_accuracy:<15.4f} {dt_accuracy:<15.4f}")
print(f"{'Precision':<20} {lr_precision:<15.4f} {dt_precision:<15.4f}")
print(f"{'Recall':<20} {lr_recall:<15.4f} {dt_recall:<15.4f}")
print(f"{'F1-Score':<20} {lr_f1:<15.4f} {dt_f1:<15.4f}")

# Determine winner
if lr_accuracy > dt_accuracy:
    winner = "Logistic Regression"
    diff = lr_accuracy - dt_accuracy
else:
    winner = "Decision Tree"
    diff = dt_accuracy - lr_accuracy

print(f"\n🏆 WINNER: {winner}")
print(f"   Better by: {diff*100:.2f}%")

# ============================================
# 7. VISUALIZATIONS
# ============================================

print("\n--- Creating Visualizations ---")

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Confusion Matrices
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Logistic Regression\nConfusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

ax2 = plt.subplot(2, 3, 2)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Decision Tree\nConfusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 2. Accuracy Comparison
ax3 = plt.subplot(2, 3, 3)
models = ['Logistic\nRegression', 'Decision\nTree']
accuracies = [lr_accuracy * 100, dt_accuracy * 100]
colors = ['#3498db', '#2ecc71']
bars = plt.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim([0, 100])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# 3. Feature Importance (Top 10)
ax4 = plt.subplot(2, 3, 4)
top_features = feature_importance.head(10)
plt.barh(top_features['Feature'], top_features['Importance'], color='coral', edgecolor='black')
plt.xlabel('Importance')
plt.title('Top 10 Important Features\n(Decision Tree)')
plt.gca().invert_yaxis()

# 4. ROC Curves
ax5 = plt.subplot(2, 3, 5)
# Logistic Regression ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr_proba)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, 
         label=f'Logistic Reg (AUC = {roc_auc_lr:.2f})')

# Decision Tree ROC
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt_proba)
roc_auc_dt = auc(fpr_dt, tpr_dt)
plt.plot(fpr_dt, tpr_dt, color='green', lw=2,
         label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# 5. Metrics Comparison
ax6 = plt.subplot(2, 3, 6)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
lr_metrics = [lr_accuracy, lr_precision, lr_recall, lr_f1]
dt_metrics = [dt_accuracy, dt_precision, dt_recall, dt_f1]

x = np.arange(len(metrics_names))
width = 0.35

plt.bar(x - width/2, lr_metrics, width, label='Logistic Reg', 
        color='#3498db', alpha=0.7, edgecolor='black')
plt.bar(x + width/2, dt_metrics, width, label='Decision Tree',
        color='#2ecc71', alpha=0.7, edgecolor='black')

plt.ylabel('Score')
plt.title('All Metrics Comparison')
plt.xticks(x, metrics_names, rotation=45)
plt.ylim([0, 1])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
comparison_plot_path = output_dir / 'heart_disease_comparison.png'
plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
print("✓ Visualizations saved to heart_disease_comparison.png")

# ============================================
# 8. SAVE MODELS
# ============================================

print("\n--- Saving Models ---")
import joblib

joblib.dump(lr_model, output_dir / 'logistic_regression_model.pkl')
joblib.dump(dt_model, output_dir / 'decision_tree_model.pkl')
joblib.dump(scaler, output_dir / 'scaler.pkl')

print("✓ Models saved:")
print("  - logistic_regression_model.pkl")
print("  - decision_tree_model.pkl")
print("  - scaler.pkl")

# ============================================
# 9. TEST ON NEW DATA
# ============================================

print("\n" + "="*80)
print("TESTING ON NEW PATIENT DATA")
print("="*80)

# Example: New patient data (adjust values as needed)
new_patient = pd.DataFrame({
    'age': [55],
    'sex': [1],
    'cp': [2],
    'trestbps': [140],
    'chol': [250],
    'fbs': [0],
    'restecg': [1],
    'thalach': [150],
    'exang': [0],
    'oldpeak': [1.5],
    'slope': [1],
    'ca': [0],
    'thal': [2]
})

print("\nNew Patient Data:")
print(new_patient)

# Predict with Logistic Regression
new_patient_scaled = scaler.transform(new_patient)
lr_prediction = lr_model.predict(new_patient_scaled)[0]
lr_probability = lr_model.predict_proba(new_patient_scaled)[0]

print("\n--- Logistic Regression Prediction ---")
print(f"Prediction: {'Heart Disease' if lr_prediction == 1 else 'No Heart Disease'}")
print(f"Probability of Disease: {lr_probability[1]*100:.2f}%")

# Predict with Decision Tree
dt_prediction = dt_model.predict(new_patient)[0]
dt_probability = dt_model.predict_proba(new_patient)[0]

print("\n--- Decision Tree Prediction ---")
print(f"Prediction: {'Heart Disease' if dt_prediction == 1 else 'No Heart Disease'}")
print(f"Probability of Disease: {dt_probability[1]*100:.2f}%")



