import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = "C:/Users/Sparsh/Documents/AI-Threat-Detector/featured_network_data.csv"
df = pd.read_csv(file_path)

# Define input features (X) and target labels (y)
X = df.drop(columns=["Length"])  # Drop Length (modify if needed)
y = (df["Length"] > df["Length"].median()).astype(int)  # Classify packets based on size

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print new class distribution
print("\n✅ After SMOTE - Training class distribution:")
print(y_resampled.value_counts())

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    # Create RandomForest model with suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    # Perform cross-validation (5 folds)
    scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
    
    # Return the mean accuracy
    return scores.mean()

# Run the hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # Run 10 trials

# Print the best parameters
print("\n✅ Best Hyperparameters Found:")
print(study.best_params)

# Train final model with best parameters
best_params = study.best_params
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_resampled, y_resampled)

print("\n✅ Final Model Trained with Optimized Parameters!")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Split dataset into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train final model on training data
final_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = final_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print results
print("\n✅ Model Evaluation Results:")
print(f"🔹 Accuracy: {accuracy:.2f}")
print(f"🔹 Precision: {precision:.2f}")
print(f"🔹 Recall: {recall:.2f}")
print(f"🔹 F1 Score: {f1:.2f}")

# Print Confusion Matrix
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib  # Import joblib to save the model

# Save the trained model
model_filename = "C:/Users/Sparsh/Documents/AI-Threat-Detector/trained_model.pkl"
joblib.dump(final_model, model_filename)

print(f"\n✅ Model saved successfully at: {model_filename}")
