import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = "/content/cStick.csv"
df = pd.read_csv(file_path)

# Rename incorrect column name
df.rename(columns={'ï»¿Distance': 'Distance'}, inplace=True)

# Explore data
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['Decision'].value_counts())

# Check for missing values and outliers
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nFeature statistics:")
print(df.describe())

# Select features
features = ['HRV', 'SpO2', 'Accelerometer', 'Distance', 'Sugar level', 'Pressure']
target = 'Decision'

# Extensive Feature Engineering

# 1. Time-series features (critical for fall detection)
window_sizes = [3, 5, 7, 10]
for window in window_sizes:
    # Rolling statistics for accelerometer
    df[f'Acc_roll_mean_{window}'] = df['Accelerometer'].rolling(window=window).mean().fillna(df['Accelerometer'])
    df[f'Acc_roll_std_{window}'] = df['Accelerometer'].rolling(window=window).std().fillna(0)
    df[f'Acc_roll_max_{window}'] = df['Accelerometer'].rolling(window=window).max().fillna(df['Accelerometer'])
    df[f'Acc_roll_min_{window}'] = df['Accelerometer'].rolling(window=window).min().fillna(df['Accelerometer'])
    df[f'Acc_roll_range_{window}'] = df[f'Acc_roll_max_{window}'] - df[f'Acc_roll_min_{window}']

    # Rolling statistics for pressure (important for altitude changes during falls)
    df[f'Pressure_roll_mean_{window}'] = df['Pressure'].rolling(window=window).mean().fillna(df['Pressure'])
    df[f'Pressure_roll_std_{window}'] = df['Pressure'].rolling(window=window).std().fillna(0)

    # Rolling statistics for HRV (heart rate variability can change during falls)
    df[f'HRV_roll_mean_{window}'] = df['HRV'].rolling(window=window).mean().fillna(df['HRV'])
    df[f'HRV_roll_std_{window}'] = df['HRV'].rolling(window=window).std().fillna(0)

# 2. Derivatives (rate of change is critical for detecting sudden movements)
for feature in ['Accelerometer', 'Pressure', 'HRV']:
    # First derivative
    df[f'{feature}_diff1'] = df[feature].diff().fillna(0)
    # Second derivative (acceleration)
    df[f'{feature}_diff2'] = df[f'{feature}_diff1'].diff().fillna(0)

    # Smoothed derivatives
    for window in [3, 5]:
        df[f'{feature}_diff1_smooth_{window}'] = df[f'{feature}_diff1'].rolling(window=window).mean().fillna(0)

# 3. Interaction features
df['Acc_Pressure'] = df['Accelerometer'] * df['Pressure']
df['Acc_HRV'] = df['Accelerometer'] * df['HRV']
df['Pressure_HRV'] = df['Pressure'] * df['HRV']
df['Acc_SpO2'] = df['Accelerometer'] * df['SpO2']

# 4. Lag features (important for time sequence)
for lag in [1, 2, 3, 5, 10]:
    df[f'Acc_lag_{lag}'] = df['Accelerometer'].shift(lag).fillna(0)
    df[f'Pressure_lag_{lag}'] = df['Pressure'].shift(lag).fillna(0)

    # Changes over different time horizons
    df[f'Acc_change_{lag}'] = (df['Accelerometer'] - df[f'Acc_lag_{lag}']).fillna(0)
    df[f'Pressure_change_{lag}'] = (df['Pressure'] - df[f'Pressure_lag_{lag}']).fillna(0)

# 5. Threshold based features
acc_mean = df['Accelerometer'].mean()
acc_std = df['Accelerometer'].std()
df['Acc_high'] = (df['Accelerometer'] > (acc_mean + 2*acc_std)).astype(int)
df['Acc_low'] = (df['Accelerometer'] < (acc_mean - 2*acc_std)).astype(int)

# Get all feature columns after feature engineering
all_features = df.columns.drop(['Decision'])
all_features = [col for col in all_features if col != target]
print(f"\nTotal number of features after engineering: {len(all_features)}")

# Create sequence features (for each sample, include features from previous time steps)
seq_length = 5
X, y = [], []

for i in range(seq_length, len(df)):
    # For each time point, include features from previous 'seq_length' time points
    current_features = []
    for j in range(seq_length, 0, -1):
        current_features.extend(df.loc[i-j, all_features].values)

    # Add current time point features
    current_features.extend(df.loc[i, all_features].values)
    X.append(current_features)
    y.append(df.loc[i, target])

X = np.array(X)
y = np.array(y)

print(f"Feature vector shape after sequence creation: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Check class distribution
print("\nClass distribution in prepared data:")
unique, counts = np.unique(y, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} samples")

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define preprocessing pipeline
preprocessing = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))  # Retain 95% of variance
])

# Apply preprocessing
X_train_processed = preprocessing.fit_transform(X_train)
X_test_processed = preprocessing.transform(X_test)

print(f"Reduced feature dimensions with PCA: {X_train_processed.shape[1]}")

# Apply SMOTEENN (combined over and undersampling)
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_processed, y_train)

print("\nClass distribution after resampling:")
unique, counts = np.unique(y_train_resampled, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} samples")

# Compute class weights
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train_resampled), y=y_train_resampled
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("\nClass weights:", class_weights_dict)

# Define multiple models for ensemble
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        tree_method='hist',
        random_state=42
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    ),
    'SVM': SVC(
        C=10,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=300,
        random_state=42
    )
}

# Train and evaluate each model
results = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred = model.predict(X_test_processed)
    predictions[name] = y_pred

    # Evaluate
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }

    # Print results
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Fall', 'Fall Prediction', 'Definite Fall'],
                yticklabels=['No Fall', 'Fall Prediction', 'Definite Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.show()

# Create ensemble prediction (voting)
ensemble_pred = np.zeros((len(y_test), len(np.unique(y))))
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        ensemble_pred += model.predict_proba(X_test_processed)
    else:
        # For models like SVM that have decision_function instead
        proba = np.zeros((len(y_test), len(np.unique(y))))
        decisions = model.decision_function(X_test_processed)
        if decisions.ndim == 1:  # Binary classification
            proba[:, 0] = 1 / (1 + np.exp(decisions))
            proba[:, 1] = 1 - proba[:, 0]
        else:  # Multi-class
            proba = np.exp(decisions) / np.sum(np.exp(decisions), axis=1)[:, np.newaxis]
        ensemble_pred += proba

# Get final prediction
ensemble_pred = np.argmax(ensemble_pred, axis=1)

# Evaluate ensemble
print("\nClassification Report for Ensemble:")
print(classification_report(y_test, ensemble_pred))

# Plot confusion matrix for ensemble
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, ensemble_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Fall', 'Fall Prediction', 'Definite Fall'],
            yticklabels=['No Fall', 'Fall Prediction', 'Definite Fall'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Ensemble')
plt.tight_layout()
plt.show()

# Compare model performances
accuracies = [results[model]['accuracy'] for model in results]
macro_f1s = [results[model]['macro_f1'] for model in results]
weighted_f1s = [results[model]['weighted_f1'] for model in results]

plt.figure(figsize=(12, 6))
x = np.arange(len(results))
width = 0.25

plt.bar(x - width, accuracies, width, label='Accuracy')
plt.bar(x, macro_f1s, width, label='Macro F1')
plt.bar(x + width, weighted_f1s, width, label='Weighted F1')

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, list(results.keys()), rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance analysis from the Random Forest model
if 'RandomForest' in models:
    rf_model = models['RandomForest']

    # We need to map back to original feature names
    # Each original feature appears seq_length+1 times (for past and current time points)
    feature_names = []
    for t in range(seq_length, -1, -1):
        time_point = "t" if t == 0 else f"t-{t}"
        for feature in all_features:
            feature_names.append(f"{feature}_{time_point}")

    # Get top 30 features
    feature_importances = rf_model.feature_importances_

    # Map PCA components back to original features (approximate)
    pca = preprocessing.named_steps['pca']

    # Get the absolute value of PCA components
    abs_components = np.abs(pca.components_)

    # For each PCA component, find the original feature with highest contribution
    pca_feature_importance = np.zeros(len(feature_names))

    for i, importance in enumerate(rf_model.feature_importances_):
        # Get the PCA component's contribution to original features
        contributions = abs_components[i, :]

        # Distribute this component's importance to original features
        pca_feature_importance += importance * contributions / np.sum(contributions)

    # Get top 30 features
    top_indices = np.argsort(pca_feature_importance)[-30:]

    plt.figure(figsize=(15, 10))
    plt.title('Top 30 Feature Importances for Fall Detection')
    plt.barh(range(len(top_indices)), pca_feature_importance[top_indices], color='b', align='center')
    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

# Save the best model
best_model_name = max(results, key=lambda x: results[x]['macro_f1'])
print(f"\nThe best model is {best_model_name} with macro F1 score: {results[best_model_name]['macro_f1']:.4f}")

import joblib
# Save preprocessing pipeline
joblib.dump(preprocessing, 'fall_detection_preprocessing.pkl')
# Save best model
joblib.dump(models[best_model_name], f'fall_detection_{best_model_name}.pkl')

# Function to predict on new data
def predict_fall(new_data, preprocessing_pipeline, model):
    """
    Predict fall category on new data.

    Parameters:
    -----------
    new_data : array-like
        New data with the same features as the training set
    preprocessing_pipeline : sklearn Pipeline
        The preprocessing pipeline used for training
    model : sklearn model
        The trained model

    Returns:
    --------
    predictions : array
        Predicted fall categories
    """
    # Apply preprocessing
    processed_data = preprocessing_pipeline.transform(new_data)

    # Make predictions
    return model.predict(processed_data)

# Example usage
print("\nExample usage for future predictions:")
print("predictions = predict_fall(new_data, preprocessing, models[best_model_name])")
