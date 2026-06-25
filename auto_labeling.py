"""
Auto Labeling dengan Keyword Extraction (TF-IDF)
Script untuk generate labels otomatis untuk setiap topic menggunakan TF-IDF
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
warnings.filterwarnings('ignore')

# Indonesian stopwords and noisy terms
STOPWORDS_ID = {
    'yang', 'dan', 'atau', 'di', 'ke', 'dari', 'untuk', 'pada', 'dengan', 'akan',
    'telah', 'dapat', 'adalah', 'ini', 'itu', 'jika', 'maka', 'oleh', 'serta',
    'lebih', 'dalam', 'tidak', 'seperti', 'hanya', 'karena', 'juga', 'namun',
    'sesuai', 'sistem', 'informasi', 'berbasis', 'menggunakan', 'metode',
    'analisis', 'implementasi', 'perancangan', 'aplikasi', 'studi', 'kasus',
    'palembang', 'raden', 'fatah', 'uin', 'sumatera', 'selatan', 'smk', 'kota',
    'negeri', 'universitas', 'stia', 'pelayanan', 'penerapan', 'penggunaan'
}

# Manual domain label mapping for professional topics
MANUAL_LABELS = {
    1: 'Sistem Informasi Hukum',
    2: 'Kualitas Layanan dan Website',
    3: 'Akademik & Keamanan Informasi',
    4: 'Evaluasi Kepuasan Pengguna',
    5: 'Usability & Keberhasilan Sistem',
    6: 'Web Service & Customer Satisfaction',
    7: 'Analisis Sentimen & Rekomendasi',
    8: 'Web Development & Rapid Application Development',
    9: 'Penerimaan Pengguna & Sistem Dukungan Keputusan',
    10: 'User Experience Evaluation',
    11: 'E-commerce & Administrasi Data',
    12: 'Extreme Programming & Pengembangan Perangkat Lunak',
    13: 'Perencanaan Strategis & Tracer Study',
    14: 'Geographic Information Systems'
}

NOISE_TERMS = {'palembang', 'raden', 'fatah', 'uin', 'stia', 'sumatera', 'selatan', 'kota', 'negeri', 'smk', 'universitas'}

def extract_keywords(text, top_n=10):
    """Extract keywords using TF-IDF"""
    try:
        # Lowercase and clean
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split into words
        words = text.split()
        # Filter stopwords, noise terms, and short words
        words = [w for w in words if len(w) > 2 and w not in STOPWORDS_ID and w not in NOISE_TERMS]
        
        if not words:
            return []
        
        # Get word frequencies
        from collections import Counter
        freq = Counter(words)
        
        # Return top keywords
        return [word for word, _ in freq.most_common(top_n)]
    
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def generate_label(topic_id, keywords):
    """Generate professional label from keywords or manual mapping"""
    if topic_id in MANUAL_LABELS:
        return MANUAL_LABELS[topic_id]
    
    if not keywords:
        return f"Topic {topic_id}"
    
    selected = [kw for kw in keywords if kw not in NOISE_TERMS]
    
    if len(selected) >= 2:
        label = " & ".join(selected[:2]).title()
    elif selected:
        label = selected[0].title()
    else:
        label = f"Topic {topic_id}"
    
    return label

print("=" * 80)
print("🤖 AUTO LABELING DENGAN KEYWORD EXTRACTION (TF-IDF)")
print("=" * 80)

# Load data
print("\n📂 Loading data...")
base_path = Path(__file__).parent

# Load datasets
topic_dist_path = base_path / "model" / "topic_distribution.csv"

if not topic_dist_path.exists():
    print(f"❌ Error: {topic_dist_path} not found")
    exit(1)

topic_dist_df = pd.read_csv(topic_dist_path)
print(f"✅ Loaded topic distribution for {len(topic_dist_df)} documents")

print("\n" + "=" * 80)
print("🔍 EXTRACTING KEYWORDS FOR EACH TOPIC")
print("=" * 80)

# Dictionary to store results
topic_labels = {}

# Get unique topics
topics = sorted(topic_dist_df['topik_dominan'].unique().astype(int))

for topic_id in topics:
    print(f"\n📌 Topic {topic_id}:")
    
    # Get all documents for this topic
    topic_docs = topic_dist_df[topic_dist_df['topik_dominan'] == topic_id]
    
    # Get text from titles
    topic_titles = topic_docs['Judul'].tolist()
    
    # Combine all titles into one text
    combined_text = " ".join(topic_titles)
    
    print(f"   Documents: {len(topic_docs)}")
    print(f"   Text length: {len(combined_text)}")
    
    if len(combined_text) < 50:
        print(f"   ⚠️  Very short text")
        label = f"Topic {topic_id}"
        keywords = []
    else:
        # Extract keywords
        keywords = extract_keywords(combined_text, top_n=10)
        print(f"   Keywords: {', '.join(keywords[:5]) if keywords else 'None'}")
        
        # Generate label
        label = generate_label(topic_id, keywords)
        print(f"   Generated Label: {label}")
    
    # Store in dictionary
    topic_labels[topic_id] = {
        'topic_id': topic_id,
        'label': label,
        'description': f"Topic with {len(topic_docs)} documents",
        'keywords': ";".join(keywords) if keywords else ""
    }

# Convert to DataFrame
labels_df = pd.DataFrame(list(topic_labels.values()))

# Save to CSV
output_path = base_path / "model" / "topic_labels.csv"
labels_df.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("✅ LABELING COMPLETE")
print("=" * 80)
print(f"\n📊 Generated labels for {len(labels_df)} topics")
print(f"💾 Saved to: {output_path}")

# Display results
print("\n📋 GENERATED LABELS:")
print(labels_df.to_string(index=False))

print("\n✨ Dashboard akan otomatis menampilkan label baru saat di-refresh!")

