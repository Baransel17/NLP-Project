import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('punkt')

print("All libraries loaded successfully!")


df = pd.read_csv('mtsamples.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
df.head()


print("Missing values:\n", df.isnull().sum())
print("\nNumber of unique specialties:", df['medical_specialty'].nunique())
print("\nTop 10 specialties:")
print(df['medical_specialty'].value_counts().head(10))

# Plot top 15 specialties
plt.figure(figsize=(12, 6))
df['medical_specialty'].value_counts().head(15).plot(kind='bar', color='steelblue')
plt.title('Top 15 Medical Specialties')
plt.xlabel('Specialty')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Keep only top 10 specialties (others have too few samples)
top10 = df['medical_specialty'].value_counts().head(10).index
df_clean = df[df['medical_specialty'].isin(top10)].copy()

# Drop rows with missing transcription
df_clean = df_clean.dropna(subset=['transcription'])

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)           # remove numbers
    text = re.sub(r'[^\w\s]', '', text)       # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df_clean['clean_text'] = df_clean['transcription'].apply(clean_text)

print("Dataset size after cleaning:", df_clean.shape)
print("\nClass distribution:")
print(df_clean['medical_specialty'].value_counts())


X = df_clean['clean_text']
y = df_clean['medical_specialty']

# TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("\nTF-IDF matrix shape:", X_tfidf.shape)


# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

# Compare
print("=" * 40)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
print(f"Naive Bayes Accuracy:         {nb_acc:.4f}")
print("=" * 40)

# Bar chart comparison
plt.figure(figsize=(7, 4))
models = ['Logistic Regression', 'Naive Bayes']
scores = [lr_acc, nb_acc]
bars = plt.bar(models, scores, color=['steelblue', 'coral'])
plt.ylim(0.5, 1.0)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{score:.4f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()


print("LOGISTIC REGRESSION - Classification Report")
print("=" * 60)
print(classification_report(y_test, lr_pred))

print("\nNAIVE BAYES - Classification Report")
print("=" * 60)
print(classification_report(y_test, nb_pred))


labels = sorted(y_test.unique())

cm = confusion_matrix(y_test, nb_pred, labels=labels)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[l.split('/')[0].strip() for l in labels],
            yticklabels=[l.split('/')[0].strip() for l in labels])
plt.title('Naive Bayes - Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


feature_names = tfidf.get_feature_names_out()
nb_log_probs = nb.feature_log_prob_

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

categories = nb.classes_

for i, (cat, ax) in enumerate(zip(categories, axes)):
    top_indices = nb_log_probs[i].argsort()[-10:][::-1]
    top_words = [feature_names[j] for j in top_indices]
    top_scores = nb_log_probs[i][top_indices]
    
    ax.barh(top_words[::-1], top_scores[::-1], color='steelblue')
    ax.set_title(cat.split('/')[0].strip(), fontsize=9, fontweight='bold')
    ax.tick_params(axis='y', labelsize=7)

plt.suptitle('Top 10 Keywords per Medical Specialty (Naive Bayes)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


print("=" * 55)
print("   CLINICAL TEXT CLASSIFICATION - FINAL SUMMARY")
print("=" * 55)
print(f"\n Dataset: Medical Transcriptions (mtsamples)")
print(f" Total samples used:       {df_clean.shape[0]}")
print(f" Number of categories:     10")
print(f" Train / Test split:       80% / 20%")
print(f" TF-IDF features:          10,000")
print(f"\n{'Model':<25} {'Accuracy':>10}")
print("-" * 37)
print(f"{'Logistic Regression':<25} {lr_acc:>10.4f}")
print(f"{'Naive Bayes':<25} {nb_acc:>10.4f}")
print("-" * 37)
print(f"\n Best model: Naive Bayes ({nb_acc:.2%} accuracy)")
print(f"\n Key finding: Class imbalance (Surgery=1088)")
print(f" affects minority class predictions.")
print(f"\n Future work: Apply SMOTE or use weighted")
print(f" loss to handle class imbalance.")
print("=" * 55)