import spacy

########################## 用model来断句
nlp = spacy.load("en_core_web_sm")

text = "Mr. Wang is a teacher. He teaches A.I. (?). Does he love his work? Of course!"
doc = nlp(text)

for sent in doc.sents:
    print(sent.text)
