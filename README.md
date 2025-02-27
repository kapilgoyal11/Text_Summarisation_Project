# Text_Summarisation_Project

Using Spacy and NLTK module with TF-IDF algorithm for text-summarisation. This code will give you the summary of inputted article. You can input the text by typing (or copy-paste) or from Txt file, PDF file or from Wikipedia Page Url.

## Purpose :- 

To save time while reading by summarizing a large article or text into fewer lines. 


## Description :-

It usage Term Frequency-Inverse Document Frequency (TF-IDF) algorithm for summarising the article.

## Features :-

You can read the text of your long article in 4 ways :-

  - By typing text on your own (or copy-paste).
  - Reading the text from **.txt file**.
  - Reading the text from **.pdf file**.(You can choose either to get summary of entire pdf or select any page interval).
  - Reading the text from **wikipedia page** (All you have to do is to provide the url of that page. Program will automatically scrap the text and summarise it for you).
  
## Requirements :-

- Python3 
- Spacy Module (short, medium, or long any type is sufficient)
- NLTK Module
- PyPdf2
- Beautiful Soup (bs4)
- urllib (already available with python itself, no need for external installation)


## How to install Requirements :-

1. Python3 can be installed from their official site https://www.python.org/ . Or you can use anaconda environment.
2. Spacy can be installed by
For Anaconda Environment > 
```
conda install -c conda-forge spacy

python3 -m spacy download en
```
For other environments > 
```
pip3 install spacy

python3 -m spacy download en
```
3. NLTK can be installed by
For Anaconda Environment > 
```
conda install -c anaconda nltk
```
For other environments > 
```
pip3 install nltk
```

4. PyPdf2 can be installed by
For Anaconda Environment > 
```
conda install -c conda-forge pypdf2
```
For other environments > 
```
pip3 install PyPDF2
```

5. Beautiful Soup (bs4)
For Anaconda Environment > 
```
conda install -c anaconda beautifulsoup4
```
For other environments > 
```
pip3 install beautifulsoup4`
```
## Getting Started :-

- Download or clone repository.

- Open cmd or terminal in same directory where **Text-Summarizer.py** file is stored and then run it by followng command :- 
```
python3 Text-Summarizer.py
```
- Now just follow along with the program.
