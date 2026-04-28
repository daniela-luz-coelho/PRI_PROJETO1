# Guia Completo de Testes — RepositóriUM Search Engine

## Pré-requisitos

```bash
# Instalar dependências
pip install -r requirements.txt

# Confirmar que o NLTK tem os dados necessários
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
```

---

## 1. Testes Automatizados (pytest)

### Correr TUDO de uma vez

```bash
# Na raiz do projeto:
pytest -v
```

### Correr com cobertura de código

```bash
pytest --cov=src --cov-report=term-missing -v
```

### Correr por módulo

```bash
# Preprocessor (tokenização, stemming, lematização, stop words)
pytest tests/preprocessor/test_preprocessor.py -v

# Índice invertido + pesquisa booleana
pytest tests/test_index_search.py -v

# TF-IDF (custom + sklearn)
pytest tests/test_tfidf.py -v

# API REST (FastAPI TestClient, sem servidor)
pytest tests/test_api.py -v
```

### Correr um teste específico

```bash
# Pelo nome da classe
pytest tests/test_tfidf.py::TestSearchCustom -v

# Por nome de função
pytest tests/test_tfidf.py::TestSearchCustom::test_relevant_doc_ranks_high -v

# Por palavra-chave
pytest -k "cosine" -v
```

### Output esperado (todos passados)

```
tests/preprocessor/test_preprocessor.py  ✓ ~25 testes
tests/test_index_search.py               ✓ ~30 testes
tests/test_tfidf.py                      ✓ ~35 testes
tests/test_api.py                        ✓ ~20 testes
```

---

## 2. Testar o Preprocessor manualmente

```bash
python -c "
from src.search.preprocessor import make_stemming_preprocessor, make_lemmatisation_preprocessor

# Stemming
pp = make_stemming_preprocessor('english')
print('Stemming:', pp.process('Information retrieval systems are fascinating'))
# → ['inform', 'retriev', 'system', 'fascin']

# Lematização
pp2 = make_lemmatisation_preprocessor('english')
print('Lemma:   ', pp2.process('Running dogs barked loudly'))
# → ['run', 'dog', 'bark', 'loudly']

# Português
pp3 = make_stemming_preprocessor('portuguese')
print('PT:      ', pp3.process('Os sistemas de recuperação de informação são úteis'))
"
```

---

## 3. Testar o Índice Invertido manualmente

```bash
python -c "
import json
from src.search.preprocessor import make_stemming_preprocessor
from src.search.inverted_index import InvertedIndex

with open('src/scraper/scraper_results.json') as f:
    docs = json.load(f)

pp = make_stemming_preprocessor('english')
idx = InvertedIndex(pp)
idx.build_from_documents(docs)

print('Estatísticas:', idx.stats())
print()
print('Postings para retrieval:', idx.get_postings('retrieval'))
print('DF de learning:', idx.get_document_frequency('learning'))
"
```

---

## 4. Testar a Pesquisa Booleana manualmente

```bash
python -c "
import json
from src.search.preprocessor import make_stemming_preprocessor
from src.search.inverted_index import InvertedIndex
from src.search.boolean_search import BooleanSearchEngine

with open('src/scraper/scraper_results.json') as f:
    docs = json.load(f)

pp = make_stemming_preprocessor('english')
idx = InvertedIndex(pp)
idx.build_from_documents(docs)
engine = BooleanSearchEngine(idx)

queries = [
    'graphics',
    'graphics AND rendering',
    'neural OR deep',
    'learning NOT deep',
    '(graphics OR rendering) AND 2022',
]

for q in queries:
    results = engine.search(q)
    print(f'Query: {q!r:40} → {len(results)} resultado(s)')
    for r in results[:2]:
        print(f'  [{r.score:.1f}] {r.document[\"title\"][:60]}')
"
```

---

## 5. Testar o TF-IDF manualmente

```bash
python -c "
import json
from src.search.preprocessor import make_stemming_preprocessor
from src.search.tfidf import TFIDFEngine

with open('src/scraper/scraper_results.json') as f:
    docs = json.load(f)

pp = make_stemming_preprocessor('english')

print('=== TF-IDF Personalizado ===')
engine = TFIDFEngine(pp, use_sklearn=False, tf_scheme='log')
engine.build_from_documents(docs)
results = engine.search('machine learning', top_k=5)
for r in results:
    print(f'  [{r.score:.4f}] {r.document[\"title\"][:60]}')

print()
print('=== TF-IDF Sklearn ===')
engine2 = TFIDFEngine(pp, use_sklearn=True)
engine2.build_from_documents(docs)
results2 = engine2.search('machine learning', top_k=5)
for r in results2:
    print(f'  [{r.score:.4f}] {r.document[\"title\"][:60]}')

print()
print('IDF de \"learning\":', engine.get_term_idf('learning'))
print('Estatísticas:', engine.stats())
"
```

---

## 6. Testar a API REST

### Arrancar o servidor

```bash
uvicorn src.api.fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

### Testes com curl

```bash
# Health check
curl http://localhost:8000/

# Estatísticas do índice
curl http://localhost:8000/stats

# Pesquisa livre (TF-IDF personalizado)
curl "http://localhost:8000/search?q=machine+learning&top_k=5"

# Pesquisa com sklearn
curl "http://localhost:8000/search?q=retrieval&algorithm=sklearn"

# Pesquisa com filtro de ano
curl "http://localhost:8000/search?q=graphics&year_from=2020&year_to=2023"

# Pesquisa booleana
curl "http://localhost:8000/search/boolean?q=graphics+AND+rendering"
curl "http://localhost:8000/search/boolean?q=neural+OR+deep"

# Pesquisa por autor
curl "http://localhost:8000/search/author?name=Silva"

# Documento por ID
curl http://localhost:8000/documents/0

# IDF de um termo
curl "http://localhost:8000/tfidf/idf?term=learning"
```

### Documentação interativa (Swagger)

Abrir no browser: **http://localhost:8000/docs**

---

## 7. Testar o Frontend

```bash
# Servidor estático simples
python -m http.server 3000 -d src/frontend

# Abrir no browser:
# http://localhost:3000
```

**Checklist manual no browser:**
- [ ] Pesquisa livre retorna resultados ordenados por score
- [ ] Mudar algoritmo para "sklearn" e comparar resultados
- [ ] Tab "Booleano": testar `graphics AND rendering`
- [ ] Tab "Booleano": testar `neural OR deep`
- [ ] Tab "Autor": pesquisar por apelido parcial
- [ ] Filtros de ano funcionam
- [ ] Painel educativo aparece após pesquisa
- [ ] Links dos documentos abrem numa nova janela
- [ ] Paginação funciona com muitos resultados

---

## 8. Docker

```bash
cd docker
docker-compose build
docker-compose up

# API disponível em: http://localhost:8000
# Frontend em:       http://localhost:3000
```

---

## 9. Troubleshooting comum

| Erro | Solução |
|------|---------|
| `LookupError: Resource stopwords not found` | `python -c "import nltk; nltk.download('stopwords')"` |
| `ModuleNotFoundError: No module named 'src'` | Correr os comandos sempre da **raiz do projeto** |
| `Connection refused localhost:8000` | Confirmar que `uvicorn` está a correr |
| `404` no frontend | API não está a correr ou CORS bloqueado |
| Testes do TF-IDF lentos | Normal — NLTK descarrega modelos na 1ª execução |