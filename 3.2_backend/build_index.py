from inverted_index import InvertedIndexBuilder

def main():
    builder = InvertedIndexBuilder(language="pt")

    # 1) Ler dados crus do scraper
    raw_data = builder.load_raw_data("../data/raw/scraper_results.json")

    # 2) Construir índice invertido
    index = builder.build_index(raw_data)

    # 3) Guardar índice
    builder.save_index("../data/index/inverted_index.json")

    print("Índice invertido criado com sucesso!")

if __name__ == "__main__":
    main()



