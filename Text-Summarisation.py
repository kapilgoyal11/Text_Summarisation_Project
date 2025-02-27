import sys
import math
import bs4 as bs
import urllib.request
import re
import PyPDF2
import nltk
import spacy
from nltk.stem import WordNetLemmatizer 

# Download necessary NLTK data files
nltk.download('wordnet')
nltk.download('stopwords')

# Load Spacy model and initialize the lemmatizer
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

# Function to read text file
def file_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().replace("\n", " ")

# Function to read PDF file
def pdf_reader(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
    return text

# Function to extract text from Wikipedia page
def wiki_text(url):
    scrap_data = urllib.request.urlopen(url)
    article = scrap_data.read()
    parsed_article = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed_article.find_all('p')
    article_text = " ".join([p.text for p in paragraphs])
    return re.sub(r'\[[0-9]*\]', '', article_text)

# Select input source
input_text_type = int(input("Select input type:\n1. Manual Text Entry\n2. Load from .txt file\n3. Load from .pdf file\n4. Wikipedia URL\n\n"))

if input_text_type == 1:
    text = input("Enter your text:\n")
elif input_text_type == 2:
    text = file_text(input("Enter file path: "))
elif input_text_type == 3:
    text = pdf_reader(input("Enter file path: "))
elif input_text_type == 4:
    text = wiki_text(input("Enter Wikipedia URL: "))
else:
    print("Invalid input type!")
    sys.exit()

# Preprocess text
doc = nlp(text)
sentences = list(doc.sents)
stop_words = set(nltk.corpus.stopwords.words('english'))

# Function to create frequency matrix
def frequency_matrix(sentences):
    freq_matrix = {}
    for sent in sentences:
        freq_table = {}
        words = [lemmatizer.lemmatize(word.text.lower()) for word in sent if word.text.isalnum()]
        for word in words:
            if word not in stop_words:
                freq_table[word] = freq_table.get(word, 0) + 1
        freq_matrix[sent.text] = freq_table
    return freq_matrix

# Function to calculate TF
def tf_matrix(freq_matrix):
    return {sent: {word: count / len(freq_table) for word, count in freq_table.items()} for sent, freq_table in freq_matrix.items()}

# Function to count sentences per word
def sentences_per_words(freq_matrix):
    word_count = {}
    for freq_table in freq_matrix.values():
        for word in freq_table.keys():
            word_count[word] = word_count.get(word, 0) + 1
    return word_count

# Function to calculate IDF
def idf_matrix(freq_matrix, sent_per_words, total_sentences):
    return {sent: {word: math.log10(total_sentences / sent_per_words[word]) for word in freq_table.keys()} for sent, freq_table in freq_matrix.items()}

# Function to calculate TF-IDF
def tf_idf_matrix(tf_matrix, idf_matrix):
    return {sent: {word: tf * idf_matrix[sent].get(word, 0) for word, tf in tf_table.items()} for sent, tf_table in tf_matrix.items()}

# Function to score sentences
def score_sentences(tf_idf_matrix):
    return {sent: sum(f_table.values()) / len(f_table) for sent, f_table in tf_idf_matrix.items() if f_table}

# Function to compute summary
def create_summary(sentences, sentence_scores, threshold):
    return " ".join([sent.text for sent in sentences if sentence_scores.get(sent.text, 0) >= threshold])

# Generate frequency and TF-IDF matrices
freq_matrix = frequency_matrix(sentences)
tf_matrix = tf_matrix(freq_matrix)
sent_per_words = sentences_per_words(freq_matrix)
idf_matrix = idf_matrix(freq_matrix, sent_per_words, len(sentences))
tf_idf_matrix = tf_idf_matrix(tf_matrix, idf_matrix)
sentence_scores = score_sentences(tf_idf_matrix)
threshold = sum(sentence_scores.values()) / len(sentence_scores)
summary = create_summary(sentences, sentence_scores, 1.3 * threshold)

# Print summary
print("\n\n", "*" * 20, "Summary", "*" * 20, "\n")
print(summary)
print("\nTotal words in original article:", len(text.split()))
print("Total words in summarized article:", len(summary.split()))
