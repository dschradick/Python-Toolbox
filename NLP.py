########## NATURAL LANGUAGE PROCESSING

#### Reguläre Ausdrücke
# split, findall, search, match:
# => siehe Regex.py
import re

my_string = "Dies ist der erste Satz!  Ist das lustig?  Denke schon.  Sind das 4 Sätze?  Oder wieviele Worte?"
# Satzenden finden
sentence_endings = "[.?!]"
print(re.split(sentence_endings, my_string))
# Grossgeschriebene Wörter
capitalized_words = "[A-Z]\w+"
print(re.findall(capitalized_words, my_string))
# Durch spaces trennen
spaces = "\s+"
print(re.split(spaces, my_string))
# Alle Zahlen finden
digits = "\d+"
print(re.findall(digits, my_string))




##### Tokenization
# String in mehrere Tokens konvertieren
# => bereitet Text für NLP vor
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import regexp_tokenize
from nltk.book import text6
from nltk.corpus import webtext

scene_one = webtext.raw("grail.txt")
sentences = sent_tokenize(holy_grail)

## Standard Regex + Tokenization
# Erstes auftreten von coconuts
match = re.search("coconuts", scene_one)
print(match.start(), match.end())
# Erster text in eckigen Klammern
pattern1 = r"\[.*\]"
print(re.search(pattern1, scene_one))
# Vierte Skript Notation => ARTHUR:
pattern2 = r"[\w\s]+:"
print(re.match(pattern2, sentences[3]))

### Word Tokenization
word_tokenize("Guten Tag!")

### Sentence Tokenization
holy_grail = webtext.raw("grail.txt")
sentences = sent_tokenize(holy_grail)
# Biespiel: Den vierten Satz word-tokenizen
tokenized_sent = word_tokenize(sentences[3])
unique_tokens = set(word_tokenize(holy_grail))
print(unique_tokens)

## Regex Tokenizer
from nltk.tokenize import regexp_tokenize
text = "Wann gehen wir Pizza essen? 🍕 Holst du mich mit dem Auto? 🚕 "
capital_words = r"[A-ZÜ]\w+"
print(regexp_tokenize(text, capital_words))
# Beispiel: Emojis filtern
text = "Wann gehen wir Pizza essen? 🍕 Holst du mich mit dem Auto? 🚕 "
emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
print(regexp_tokenize(text, emoji))

## Tweet Tokenizer
tweets =  ['Dies ist ein #np Parser in #python', '#nlp toll! <3 #lernen', 'Danke @daniel :) #nlp #python']
from nltk.tokenize import TweetTokenizer
# Hashtags finden
pattern1 = r"#\w+"
regexp_tokenize(tweets[0], pattern1)
# Hashtags und Mentions
pattern2 = r"([#|@]\w+)"
regexp_tokenize(tweets[-1], pattern2)
# Alle tweets in eine Liste
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)




#### Charts für NLP
import matplotlib.pyplot as plt

## Wort Frequenz
# Beispiel: Wörter pro Zeile
lines = holy_grail.split('\n')
# Sprecher entfernen -z.b. ARTHUR:
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', l) for l in lines]
# Zeile tokenizen
tokenized_lines = [regexp_tokenize(s, "\w+") for s in lines]
line_num_words = [len(t_line) for t_line in tokenized_lines]
plt.hist(line_num_words)
plt.show()




#### Bag-of-words
# Grundlegende Methode um Topics im Text zu Finden
# 1. Tokens erzeugen,
# 2. Tokens zählen
# => je häufiger ein Wort desto wichtiger könnte es sein
from collections import Counter

tokens = word_tokenize(holy_grail)
lower_tokens = [t.lower() for t in tokens]
bow_simple = Counter(lower_tokens)
print(bow_simple.most_common(10)) # => die 10 häufigsten Wörter
# => beinhaltet z.B. noch Satzzeichen und Stopwörter => Preprocessing




#### Preprocessing
# Tokenization, Lowercasing,
# Lemmatization/Stemming = Wörter kürzen auf ihren Wortstamm
# Entfernen von Satzzeichen und weitere ungewollte Tokens wie
# Stop-Wörter = Wörter die keine große Bedeutung tragen wie "the", "a"
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Nur Wörter mit Buchstaben => keine Satzzeichen oder Zahlen
alpha_only = [t for t in lower_tokens if t.isalpha()]
# Stopwörter entfernen
no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
# Lemmatisieren
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
#  Bag-of-words
bow = Counter(lemmatized)
lemmatized
print(bow.most_common(10))
# => Resultat: Aufgräumtes Bag-of-words




#### Gensim
# NLP library für komplexere Aufgaben
# - Dokument oder word vectoren erstellen
# - Topic identifikation und Dokumenten Vergleich
# Word Embedding / Vector: mehrdimensionale Reprästentation eines Wortes -
#      Großes array mit sparse features (viele nullen, wenige einsen)
#      => erlaubt es basierend auf Nähe Verwandschaft zwischen Wörtern und Dokumenten zu sehen
#      => oder auch vergleiche: Vektoroperation King-Queen ungefähr gleich zu Man-Woman
#          bzw. Spanien ist zu Madrid wie Italien zu Rom
#      => trainiert durch ein größeren Corpus
from gensim.corpora.dictionary import Dictionary

### Korpus erstellen
# Korpus = Menge von Texten
# Gensium erlaubt einfaches anlegen von einem Korpus
# Verwendet Bag-Of-Words model
# Erzeugt mapping: Id für jeden token
# => Dokumente dann repräsentiert durch Tokenids und wie häufig sie vorkommen
my_documents = ['The movie was about a spaceship and aliens.', 'I really liked the movie!',
'Awesome action scenes, but boring characters.', 'The movie was awful! I hate alien films.', 'Space is cool! I liked the movie.',
'More space films, please!',]
# Dokumente erzeugen
tokenized_docs = [word_tokenize(doc.lower()) for doc in my_documents]
# Dictonary aus Dokumenten erzugen
dictionary = Dictionary(tokenized_docs)
# id für "awesome"
awesome_id = dictionary.token2id.get("awesome")
# Id benutzen um das Wort auszugeben
print(dictionary.get(awesome_id))
dictionary.token2id
# Korpus erzeugen
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
corpus
# 10 häufigste Wörter im 5. Dokument
print(corpus[4][:10])
# LDA: Latent Dirichlet allocation
# kann mit gensim auf texte für topic analysis und modeling angewendet werden


### Häufigste Wöter in den Dokumenten
doc = corpus[4]
## Sortieren des dokuments nach frequenz
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

## Top 5 Wörter mit Frequenz
for word_id, word_count in bow_doc[:5]:
    print(dictionary.get(word_id), word_count)

## Top 5 Wörter in allen Dokumenten mit count
from collections import defaultdict
import itertools
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)
for word_id, word_count in sorted_word_count[:5]:
    print(dictionary.get(word_id), word_count)





#### tf-idf
# Term frequency - inverse document frequency
# Erblaubt es die wichtigsten Wörter und topics in
# einem Dokument eines Korpus (mit geiltem Vokabular) zu bestimmmen
#
# Idee: Korpus hat gemeinsame Wörter neben den Stopwörtern, welche aber nicht wichtig sind
# => diese sollen von der Wichtigkeit herabgewichtet werden
#    Bsp: Computer in Informatik-Artikeln => soll weniger gewicht bekommen
# => stellt sicher, dass die (häufigen) Wörter, welche in allen Dokumenten vorbkommen
#    nicht als Key-Wörter bestimmt werden
#    sondern die Dokument-spezfischen Wörter mit hoher frequenz
# Formel:  w_{i,j} = tf_{i,j} * log( N / df_i)
# w_{i,j} = tf-idf gewichtung des tokens i in dokument j
#  => Wert von 0-1: abhängig vom ersten (tf) oder zweiten Faktor (log)
# tf_{if} = anzahl der vorkommen von token i in dokument j (tf = term frequency)
# N = Anzal der Dokumente
# df_i = anzahl der dokumente die token i enthalten (df = document frequency)
from gensim.models.tfidfmodel import TfidfModel

tfidf = TfidfModel(corpus)
tfidf_weights = tfidf[corpus[1]]
print(tfidf_weights)
text6.generate("In the beginning of his brother is a hairy man , whose top may reach")
# Sortieren der Gewichtungen
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Top 5 gewichtete Wörter
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)
#!!! => je höher die Gewichtung eines Wortes desto eindeutiger bestimmt es das Topic


import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

displacy.serve(doc, style='dep')

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)





import spacy
import codecs

nlp = spacy.load('de', tagger=False, parser=False, matcher=False)

with codecs.open('nachricht.txt', 'r', 'utf-8') as myfile:
  data = myfile.read()

# Create a new document: doc
doc = nlp(data)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)
