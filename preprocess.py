# %%
import pandas as pd

# Load the dataset
df = pd.read_csv("Veremi_final_dataset.csv")   

print(df.info())  # Check column data types
print(df.head())  # Preview first few rows

# Check unique attack types
print(df["attack_type"].unique())


# %%
print(df['attack_type'].value_counts()) 
print(df['type'].unique())
print(df.groupby('type')['attack'].value_counts())


# %%
total_entries = df.shape[0]
print("Total entries in filtered_data:", total_entries)

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)



# %%
# Check for duplicates in the filtered dataset
duplicates = df[df.duplicated()]

# Count the number of duplicate rows
duplicate_count = duplicates.shape[0]

print("Number of duplicate rows:", duplicate_count)
print("Duplicate rows:\n", duplicates)

# %%
print(df.head())

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Count benign (0) and attack (1) messages
attack_counts = df['attack'].value_counts()

# Plot attack distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=attack_counts.index, y=attack_counts.values, palette=['blue', 'red'])
plt.xticks([0, 1], ['Benign (0)', 'Attack (1)'])
plt.ylabel('Number of Messages')
plt.xlabel('Attack Status')
plt.title('Attack Distribution')
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
sns.countplot(data=df, x='attack_type', order=df['attack_type'].value_counts().index, palette='coolwarm')
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.title("Attack Type Distribution")
plt.show()

# %%
df['attack_type'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8, 8), cmap='coolwarm')
plt.title("Attack Type Distribution")
plt.ylabel("")  # Hide y-axis label
plt.show()

# %%
#Normalization
from sklearn.preprocessing import MinMaxScaler

# Select all numerical features (excluding categorical and time-related ones)
features_to_normalize = [
    'pos_0', 'pos_1', 'pos_noise_0', 'pos_noise_1', 
    'spd_0', 'spd_1', 'spd_noise_0', 'spd_noise_1', 
    'acl_0', 'acl_1', 'acl_noise_0', 'acl_noise_1', 
    'hed_0', 'hed_1', 'hed_noise_0', 'hed_noise_1'
]

# Initialize Min-Max Scaler
scaler = MinMaxScaler()

# Apply Min-Max Scaling only to numerical features
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Verify normalization
print(df.head())


# %%
# Noise reduction
# Apply a moving average filter to smooth noise
def moving_average(series, window_size=3):
    return series.rolling(window=window_size, min_periods=1).mean()

# Apply to selected numerical features
features_to_smooth = ['pos_0', 'pos_1', 'spd_0', 'spd_1', 'acl_0', 'acl_1']
for feature in features_to_smooth:
    df[feature] = moving_average(df[feature])

print(df.head())
print(df.shape)



# %%
#Data Augmentation - Random Perturbation

import numpy as np

# Add random noise to suitable features
noise_factor = 0.01
noisy_features = ['pos_0', 'pos_1', 'spd_0', 'spd_1', 'acl_0', 'acl_1']
df[noisy_features] += noise_factor * np.random.normal(size=df[noisy_features].shape)

print(df.head())  # Verify augmented dataset




# %%
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform `attack_type`
df['attack_type'] = label_encoder.fit_transform(df['attack_type'])

# Print mapping of original attack types to encoded values
attack_type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

print("Attack Type Encoding Mapping:")
for attack, encoded_value in attack_type_mapping.items():
    print(f"{attack} → {encoded_value}")



# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# %%
print(df.shape)

# Feature Selection
# Drop index or identifier columns
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(df.head())

# %%
from sklearn.model_selection import train_test_split

# Define features (X) and labels (y)
X = df.drop(columns=['attack'])
y = df['attack']

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# %%
# Save the preprocessed dataset
df.to_csv("preprocessed_dataset.csv", index=False)
print("Preprocessed dataset saved successfully.")


