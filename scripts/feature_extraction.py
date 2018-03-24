"""
Perform feature extraction on given texts

"""


import spacy
import textacy

text = "Donald Trump is running for president. Hillary Clinton is, too."

doc = textacy.TextDoc(text)
nes1 = list(doc.named_entities())
nes2 = list(textacy.extract.named_entities(doc.spacy_doc))

nlp = spacy.load('en')
spacy_doc = nlp(text)
nes3 = list(textacy.extract.named_entities(spacy_doc))

assert nes1 == nes2 == nes3
