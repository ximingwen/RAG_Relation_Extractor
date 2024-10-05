#!/usr/bin/env python3

import stanza
import csv
import os

os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'

pkg="craft"

stanza.download('en', package=pkg)

config = dict(
    lang = 'en',
    processors = 'tokenize,pos', #'tokenize,pos,lemma,depparse,ner', 
    package = pkg,
    pos_batch_size = 10, 
    memory = '24G',
    batch_size = 500,
    tokenize_batch_size = 500,
    ner_batch_size = 500
    )

nlp = stanza.Pipeline(**config)

line = ""
with open("./data/ChemProt.csv") as csvData:
    r = csv.reader(csvData)
    r.__next__()
    line = r.__next__()[1]
    orig = r[0]

doc = nlp(line)
doc2 = nlp(orig)

print(f"Sentence: {line}")

patterns = []
idxA = 0
idxB = len(words)
for word in doc.sentences[0].words:
    if "ENTITY_A" in word.text:
        idxA = word.id
    if "ENTITY_B" in word.text:
        idxB = word.id
    if word.tupos == "VERB" and word.id > idxA and word.id < idxB:
        patterns.append(doc2.sentences[0].words[idxA], word, doc2.sentences[0].words[idxB])


print(patterns)
#print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')


