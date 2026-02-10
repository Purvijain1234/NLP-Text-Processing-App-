import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Functions
def tokenize(text):
    return word_tokenize(text)

def stem_text(text):
    return [stemmer.stem(w) for w in word_tokenize(text)]

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]

def pos_tagging(text):
    return nltk.pos_tag(word_tokenize(text))

def ner(text):
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(pos_tags)

    entities = []
    for chunk in tree:
        if hasattr(chunk, 'label'):
            entity = " ".join(c[0] for c in chunk)
            entities.append((entity, chunk.label()))

    return entities if entities else "No named entities found"

# Streamlit UI
st.title("🧠 NLP Text Processing App")

text = st.text_area("Enter your text")

option = st.selectbox(
    "Choose NLP Operation",
    ["Tokenization", "Stemming", "Lemmatization", "POS Tagging", "NER"]
)

if st.button("Process"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        if option == "Tokenization":
            st.write(tokenize(text))

        elif option == "Stemming":
            st.write(stem_text(text))

        elif option == "Lemmatization":
            st.write(lemmatize_text(text))

        elif option == "POS Tagging":
            st.write(pos_tagging(text))

        elif option == "NER":
            st.write(ner(text))
