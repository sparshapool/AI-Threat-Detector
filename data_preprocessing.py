import pandas as pd
import numpy as np

# Load the dataset
file_path = "C:/Users/Sparsh/Documents/AI-Threat-Detector/network_data.csv"

try:
    df = pd.read_csv(file_path)
    print("✅ Dataset Loaded Successfully!")
except FileNotFoundError:
    print("❌ Error: File not found. Please check the file path.")
    exit()

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Drop unnecessary columns
columns_to_drop = ["No.", "Time"]  # Adjust based on your dataset
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")

# Handle missing values
df = df.fillna(0)  # Replace NaN values with 0

# Convert categorical data to numerical if required
df = pd.get_dummies(df, drop_first=True)

# Save the cleaned dataset
cleaned_file_path = "C:/Users/Sparsh/Documents/AI-Threat-Detector/cleaned_network_data.csv"
df.to_csv(cleaned_file_path, index=False)

print("✅ Preprocessing complete. Cleaned data saved at:", cleaned_file_path)

# --------------------- Feature Engineering ---------------------

# Load the cleaned dataset
df = pd.read_csv(cleaned_file_path)

# Print the available columns for debugging
print("Columns in dataset:", df.columns.tolist())

# Identify the one-hot encoded Source and Destination columns
source_columns = [col for col in df.columns if col.startswith("Source_")]
destination_columns = [col for col in df.columns if col.startswith("Destination_")]


# Calculate Total Source Frequency as a count per source IP
df["Total_Source_Freq"] = df[source_columns].sum(axis=1) * len(source_columns)

df["Total_Destination_Freq"] = df[destination_columns].sum(axis=1)

# Normalize Packet Length
df["Length"] = (df["Length"] - df["Length"].min()) / (df["Length"].max() - df["Length"].min())

# Drop the original one-hot encoded IP columns (optional)
df = df.drop(columns=source_columns + destination_columns, errors="ignore")

# Save the feature-engineered dataset
feature_file_path = "C:/Users/Sparsh/Documents/AI-Threat-Detector/featured_network_data.csv"
df.to_csv(feature_file_path, index=False)

print("✅ Feature Engineering complete. Data saved at:", feature_file_path)
