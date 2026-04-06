import spacy

nlp = spacy.load("en_core_web_sm")


def normalize_text(text: str):

    doc = nlp(text.lower().strip())

    lemmas = []

    for token in doc:

        if token.is_stop or token.is_punct:
            continue

        lemmas.append(token.lemma_)

    return " ".join(lemmas)