from inverted_index import InvertedIndexBuilder
from boolean_model import BooleanModel

def main():
    # 1) Carregar índice invertido
    index, total_docs, doc_ids = InvertedIndexBuilder.load_index("../data/index/inverted_index.json")

    # 2) Criar modelo booleano
    bm = BooleanModel(index, total_docs)

    # 3) Testar algumas queries
    print("Query: cancer AND breast")
    print(bm.search("cancer AND breast"))

    print("Query: cancer breast")
    print(bm.search("cancer breast"))

    print("Query: NOT cancer")
    print(bm.search("NOT cancer"))

if __name__ == "__main__":
    main()

