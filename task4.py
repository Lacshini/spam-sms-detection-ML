"""
============================================================
  TASK 4 — SPAM SMS DETECTION
  Dataset: Kaggle - uciml/sms-spam-collection-dataset
  Models: Naive Bayes, Logistic Regression, SVM
============================================================
"""

# ── 1. IMPORTS ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import re
import warnings
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)

warnings.filterwarnings('ignore')

print("=" * 60)
print("  SPAM SMS DETECTION — ML Pipeline")
print("=" * 60)


# ── 2. LOAD DATASET ─────────────────────────────────────────
df = pd.read_csv(
    r'C:\Users\panda\Downloads\archive (3)\spam.csv',
    encoding='latin-1'
)[['v1', 'v2']]

df.columns = ['label', 'message']
df['is_spam'] = (df['label'] == 'spam').astype(int)

print(f"\n✅  Dataset loaded: {len(df):,} messages")
print(f"    Spam     : {df['is_spam'].sum():,} ({df['is_spam'].mean()*100:.2f}%)")
print(f"    Ham      : {(df['is_spam']==0).sum():,}")


# ── 3. TEXT PREPROCESSING ───────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)   # URLs
    text = re.sub(r'\d+', ' num ', text)               # Numbers
    text = re.sub(r'[^a-z\s]', ' ', text)             # Special chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_msg'] = df['message'].apply(clean_text)
df['msg_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))

print("✅  Text preprocessing done.")


# ── 4. TRAIN / TEST SPLIT ───────────────────────────────────
X = df['clean_msg']
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅  Split: {len(X_train):,} train | {len(X_test):,} test")


# ── 5. THREE MODELS ─────────────────────────────────────────
tfidf_params = dict(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words='english'
)

models = {
    "Naive Bayes": Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   MultinomialNB(alpha=0.1))
    ]),
    "Logistic Regression": Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ]),
    "Linear SVM": Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   LinearSVC(C=1.0, max_iter=2000, random_state=42))
    ])
}

results = {}
print("\n── Training Models ──────────────────────────────────────")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else y_pred

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    results[name] = {
        'model': pipe, 'pred': y_pred,
        'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec
    }
    print(f"  {name:<22}  Acc={acc:.2%}  F1={f1:.2%}  Precision={prec:.2%}")


# ── 6. BEST MODEL ───────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['f1'])
best      = results[best_name]
print(f"\n🏆  Best Model: {best_name}")
print("\nClassification Report:")
print(classification_report(y_test, best['pred'],
                            target_names=['Ham (Legit)', 'Spam']))


# ── 7. VISUALISATION ────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0D1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
tkw = dict(color='white', fontsize=12, fontweight='bold', pad=10)

# A: Spam vs Ham Distribution
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor('#161B22')
counts = [(y==0).sum(), (y==1).sum()]
labels = ['Ham (Legit)', 'Spam']
colors = ['#2ECC71', '#E74C3C']
bars   = ax0.bar(labels, counts, color=colors, edgecolor='none', width=0.5)
for bar, val in zip(bars, counts):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f"{val:,}\n({val/len(y)*100:.1f}%)",
             ha='center', color='white', fontsize=11)
ax0.set_title("Spam vs Ham Distribution", **tkw)
ax0.set_ylabel("Count", color='#8B949E')
ax0.tick_params(colors='#8B949E')
ax0.spines[:].set_visible(False)

# B: Model Comparison
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_facecolor('#161B22')
mnames = list(results.keys())
accs  = [results[m]['acc']  for m in mnames]
f1s   = [results[m]['f1']   for m in mnames]
precs = [results[m]['prec'] for m in mnames]
x, w  = np.arange(len(mnames)), 0.25
b1 = ax1.bar(x - w, accs,  w, label='Accuracy',  color='#3498DB', edgecolor='none')
b2 = ax1.bar(x,     f1s,   w, label='F1 Score',  color='#2ECC71', edgecolor='none')
b3 = ax1.bar(x + w, precs, w, label='Precision', color='#E74C3C', edgecolor='none')
for bars in [b1, b2, b3]:
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{bar.get_height():.2f}", ha='center', color='white', fontsize=7)
ax1.set_title("Model Comparison", **tkw)
ax1.set_xticks(x)
ax1.set_xticklabels(['Naive\nBayes', 'Logistic\nRegr.', 'Linear\nSVM'],
                    color='#8B949E', fontsize=9)
ax1.set_ylim(0, 1.15)
ax1.tick_params(colors='#8B949E')
ax1.spines[:].set_visible(False)
ax1.legend(facecolor='#161B22', labelcolor='white', fontsize=8)

# C: Confusion Matrix
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#161B22')
cm = confusion_matrix(y_test, best['pred'])
ax2.imshow(cm, cmap='Reds')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Ham', 'Spam'], color='#8B949E', fontsize=11)
ax2.set_yticklabels(['Ham', 'Spam'], color='#8B949E', fontsize=11)
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f"{cm[i,j]:,}", ha='center', va='center',
                 color='white', fontsize=15, fontweight='bold')
ax2.set_title(f"Confusion Matrix\n({best_name})", **tkw)
ax2.set_xlabel("Predicted", color='#8B949E')
ax2.set_ylabel("Actual",    color='#8B949E')

# D: Message Length - Spam vs Ham
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#161B22')
ham_len  = df[df['is_spam']==0]['msg_length']
spam_len = df[df['is_spam']==1]['msg_length']
ax3.hist(ham_len,  bins=40, alpha=0.7, color='#2ECC71', label='Ham',  edgecolor='none')
ax3.hist(spam_len, bins=40, alpha=0.7, color='#E74C3C', label='Spam', edgecolor='none')
ax3.set_title("Message Length Distribution", **tkw)
ax3.set_xlabel("Character Count", color='#8B949E')
ax3.set_ylabel("Frequency", color='#8B949E')
ax3.tick_params(colors='#8B949E')
ax3.spines[:].set_visible(False)
ax3.legend(facecolor='#161B22', labelcolor='white', fontsize=9)

# E: Top Spam Keywords
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#161B22')
best_pipe  = best['model']
feat_names = best_pipe.named_steps['tfidf'].get_feature_names_out()
clf        = best_pipe.named_steps['clf']
if hasattr(clf, 'coef_'):
    spam_idx  = np.argsort(clf.coef_[0])[-12:][::-1]
    top_words = [feat_names[i] for i in spam_idx]
    top_vals  = [clf.coef_[0][i] for i in spam_idx]
else:
    spam_idx  = np.argsort(clf.feature_log_prob_[1])[-12:][::-1]
    top_words = [feat_names[i] for i in spam_idx]
    top_vals  = [clf.feature_log_prob_[1][i] for i in spam_idx]

top_words = top_words[::-1]
top_vals  = top_vals[::-1]
bars4 = ax4.barh(top_words, top_vals, color='#E74C3C', edgecolor='none')
ax4.set_title("Top Spam Keywords", **tkw)
ax4.set_xlabel("TF-IDF Weight", color='#8B949E')
ax4.tick_params(colors='#8B949E')
ax4.spines[:].set_visible(False)

# F: Live Predictions Table
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor('#161B22')
ax5.axis('off')

test_msgs = [
    "Congratulations! You won £1000 prize. Call now!",
    "Hey, are you coming to the party tonight?",
    "FREE entry to win fabulous cash prizes! Text WIN",
    "Ok I'll be home by 7pm for dinner",
    "URGENT! Your mobile number has won £5000 prize",
    "Can you pick up milk on your way home please?",
    "Claim your FREE gift now! Limited time offer!"
]
preds = best['model'].predict([clean_text(m) for m in test_msgs])
pred_labels = ['🔴 SPAM' if p == 1 else '✅ HAM' for p in preds]

cell_text   = [[m[:35] + '...', l] for m, l in zip(test_msgs, pred_labels)]
cell_colors = [['#1a0a0a' if p==1 else '#0a1a0a', '#2d0a0a' if p==1 else '#0a2d0a']
               for p in preds]

tbl = ax5.table(
    cellText=cell_text,
    colLabels=['Message', 'Prediction'],
    cellColours=cell_colors,
    loc='center', cellLoc='left'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.6)
for (r,c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#30363D')
    cell.set_text_props(color='white')
    if r == 0:
        cell.set_facecolor('#21262D')
        cell.set_text_props(color='white', fontweight='bold')
ax5.set_title("Live Predictions", **tkw)

fig.suptitle("📱  Spam SMS Detection — ML Pipeline",
             color='white', fontsize=16, fontweight='bold', y=0.98)

import os
desktop   = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop, "spam_detection_result.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
print(f"\n✅  Chart saved to Desktop: spam_detection_result.png")
plt.show()

# ── 8. LIVE TEST ────────────────────────────────────────────
print("\n── Live Predictions ─────────────────────────────────────")
for msg, pred in zip(test_msgs, preds):
    label = "🔴 SPAM" if pred == 1 else "✅ HAM"
    print(f"  {label}  →  {msg[:55]}")

print("\n" + "="*60)
print(f"  ✅  Best Model : {best_name}")
print(f"  ✅  Accuracy   : {best['acc']:.2%}")
print(f"  ✅  F1 Score   : {best['f1']:.2%}")
print(f"  ✅  Precision  : {best['prec']:.2%}")
print(f"  ✅  Recall     : {best['rec']:.2%}")
print("="*60)
