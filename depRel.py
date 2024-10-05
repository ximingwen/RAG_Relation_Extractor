#!/usr/bin/env python3

from dataclasses import dataclass
import stanza
import csv
import os
import string
import requests

os.environ['CUDA_VISIBLE_DEVICES']='1,2'

@dataclass
class Token:
    index: str
    pos: str
    lemma: str
    dep: int
    entity: str
    head: int
    start_idx: int
    end_idx: int
    def __init__(self, tok):
        self.index = tok.id
        self.pos = tok.upos
        self.lemma = tok.lemma
        self.start_idx = tok.start_char
        self.end_idx = tok.end_char
        self.head = tok.head
        self.dep = 0
    

def getMain(sentence, ents, verbs):
    for e in ents:
        for v in verbs:
            for word in sentence.words:
                if word.text == e.text:
                    if word.head == v.id:
                        tok = Token(word)
                        tok.entity = e.type
                        return tok
    return None

def getReachEnts(sentence: str):
    url = r"http://agathon.sista.arizona.edu:8080/odinweb/api/text"
    resp = requests.post(url, params={'text': sentence})
    ents = resp.json()["entities"]
    res = [(e["text"], e["type"]) for e in ents["frames"]]
    
    return res
    
def cleanLine(line):
    temp = line.replace('%', ' percent ')
    temp = temp.replace('<<','')
    temp = temp.replace('>>','')
    temp = temp.replace('[[','')
    temp = temp.replace(']]','')
    return temp

pkg="craft"

stanza.download('en', package=pkg)

config = dict(
    lang = 'en',
    processors = {'ner': 'bionlp13cg'}, #'tokenize,pos,lemma,depparse,ner', 
    package = pkg,
    pos_batch_size = 10, 
    memory = '24G',
    batch_size = 500,
    tokenize_batch_size = 500,
    ner_batch_size = 500
    )

nlp = stanza.Pipeline(**config)

line = ""
def getRels():
    with open("./data/ChemProt.csv") as csvData:
        r = csv.reader(csvData)
        #r.__next__()
        #line = r.__next__()[0]
        next(r)
        for row in r:
            doc = nlp(row[0])
            for sentence in doc.sentences:
                ents = [ ent for sent in doc.sentences for ent in sent.ents]
                verbs = filter(lambda word: word.upos == "VERB" and "mod" not in word.deprel, sentence.words)
                #print(ents)
                main = getMain(sentence, ents, verbs)
                if main == None:
                    continue
                root = sentence.words[main.head-1].text
                antagonists = filter(lambda x: x.type != main.entity, ents)
                protagonists = filter(lambda x: x.type == main.entity, ents)
                for p in protagonists:
                    for a in antagonists:
                        print((p.text,root,a.text))
#line = line.translate(str.maketrans('', '', string.punctuation))

#line = line.replace(']', '')
#line = line.replace('[', '')

with open("./data/ChemProt.csv") as csvData:
    r = csv.reader(csvData)
    r.__next__()
    line = r.__next__()[0]
    '''for row in r:
        doc = nlp(cleanLine(row[0]))
        for sentence in doc.sentences:
            ents = [ (ent.text, ent.type) for sent in doc.sentences for ent in sent.ents]
            print(ents)'''


doc = nlp(cleanLine(line))

for sentence in doc.sentences:
    ents = [ ent for sent in doc.sentences for ent in sent.ents]
    verbs = filter(lambda word: word.upos == "VERB" and "mod" not in word.deprel, sentence.words)
    main = getMain(sentence, ents, verbs)
    root = sentence.words[main.head].text
    antagonists = filter(lambda x: x.type != main.entity, ents)
    protagonists = filter(lambda x: x.type == main.entity, ents)
    for p in protagonists:
        for a in antagonists:
            print((p.text,root,a.text))
exit(0)

print(f"Sentence: {line}")
print("Entities:")

verbs = []
deps = []
for sentence in doc.sentences:
    '''for word in sentence.words:
        if word.upos == "VERB":
            verbs.append(word.id)
    '''
#    verbs = filter(lambda word: word.upos == "VERB" and "mod" not in word.deprel, sentence.words)

    for word in sentence.words:
        print(word)
        #if word.head in verbs and word.upos == "NOUN":
            #deps.append(word)
            
    

#print(deps)

#for ent in doc.entities:
#    print(f'{ent.text}\t{ent.type}')
#print(*[f'token: {token.text}\tner: {token.ner}' for sent in doc.sentences for token in sent.tokens], sep='\n')

#print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')

print(*[f'{ent}' for sent in doc.sentences for ent in sent.ents], sep='\n')
doc.sentences[0].print_dependencies()


