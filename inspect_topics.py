from pathlib import Path
from gensim.models import LdaModel
from gensim.corpora import Dictionary

base = Path('.')
model_path = base / 'model' / 'lda_model.gensim'
id2word_path = base / 'model' / 'lda_model.gensim.id2word'
if not model_path.exists() or not id2word_path.exists():
    raise FileNotFoundError(f'Model not found: {model_path} or {id2word_path}')

lda = LdaModel.load(str(model_path))
id2word = Dictionary.load(str(id2word_path))
print('num topics', lda.num_topics)
for i in range(lda.num_topics):
    terms = lda.show_topic(i, topn=15)
    print(f'Topic {i+1}: ' + ', '.join([t[0] for t in terms]))
