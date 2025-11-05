import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# -----------------------------
# Load dataset
# -----------------------------
print("Loading UCI HAR Dataset...")

X_train = pd.read_csv("./UCI HAR Dataset/train/X_train.txt", delim_whitespace=True, header=None)
y_train = pd.read_csv("./UCI HAR Dataset/train/y_train.txt", delim_whitespace=True, header=None)
X_test = pd.read_csv("./UCI HAR Dataset/test/X_test.txt", delim_whitespace=True, header=None)
y_test = pd.read_csv("./UCI HAR Dataset/test/y_test.txt", delim_whitespace=True, header=None)

# Load feature names
features = pd.read_csv("./UCI HAR Dataset/features.txt", delim_whitespace=True, header=None)[1].tolist()
X_train.columns = features
X_test.columns = features

# Map activity labels
activity_labels = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}
y_train = y_train[0].map(activity_labels)
y_test = y_test[0].map(activity_labels)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Classes:", y_train.unique())

# -----------------------------
# Class distribution
# -----------------------------
plt.figure(figsize=(8, 5))
y_train.value_counts().plot(kind="bar", title="Training Set Class Distribution")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close()

# -----------------------------
# Feature statistics
# -----------------------------
summary = X_train.describe().T
print("\nFeature summary (first 10):")
print(summary.head(10))

# -----------------------------
# Correlation heatmap (subset)
# -----------------------------
plt.figure(figsize=(10, 8))
corr = X_train.corr().iloc[:20, :20]  # smaller subset for readability
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (first 20 features)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# -----------------------------
# Distribution of one feature across activities
# -----------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x=y_train, y=X_train["tBodyAcc-mean()-X"])
plt.xticks(rotation=30)
plt.title("tBodyAcc-mean()-X by Activity")
plt.tight_layout()
plt.savefig("feature_distribution.png")
plt.close()

# -----------------------------
# PCA for visualization
# -----------------------------
print("\nRunning PCA for visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette="tab10", alpha=0.6)
plt.title("PCA of UCI HAR (first 2 components)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("pca_plot.png")
plt.close()

print("\n EDA complete! Saved plots:")
print(" - class_distribution.png")
print(" - correlation_heatmap.png")
print(" - feature_distribution.png")
print(" - pca_plot.png")
