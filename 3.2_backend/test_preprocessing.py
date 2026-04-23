# este ficheiro só serviu para testar o preprocessing

############# NÃO ENTREGAR

from preprocessing import Preprocessor

p = Preprocessor(language="pt", remove_stopwords=True, use_stemming=True)

texto = "Os computadores estão a aprender cada vez mais rápido."

tokens = p.preprocess(texto)
print(tokens)



