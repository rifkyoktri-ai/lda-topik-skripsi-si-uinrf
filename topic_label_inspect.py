import pandas as pd
import re
from collections import Counter

ignore = set(['dan','di','ke','dari','yang','pada','dengan','untuk','berbasis','sistem','informasi','pada','akan','dalam','sebuah','sebagai','atau','itu','ada','menggunakan','terhadap','oleh','maka','dengan','melalui','serta','pada','sistem','informasi','sistem','dan','untuk'])

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())


topic = pd.read_csv('model/topic_distribution.csv')
for tid in sorted(topic['topik_dominan'].unique().astype(int)):
    titles = topic[topic['topik_dominan'] == tid]['Judul'].astype(str).tolist()
    words = []
    for t in titles:
        t = normalize(t)
        for w in t.split():
            if len(w) > 2 and w not in ignore:
                words.append(w)
    freq = Counter(words)
    most = freq.most_common(20)
    print(f'=== Topic {tid} ({len(titles)} docs) ===')
    print(', '.join([f'{w}:{c}' for w,c in most[:15]]))
    print()
