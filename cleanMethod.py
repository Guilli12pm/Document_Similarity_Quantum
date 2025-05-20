import re 
import unicodedata
import string
from io import open
import glob
import os
import torch
import random
from nltk.corpus import words
import nltk

nltk.download('words', quiet=True)

english_words = set(words.words())

def is_english_word(word):
    # Check if the word is in the set of English words
    return word.lower() in english_words

all_letters = string.ascii_letters + " .,;'"
MAX_LENGTH_WORD = 15

def findFiles(path): return glob.glob(path)

def clean_and_tokenize(text):
    """
    Cleaning a document with:
        - Lowercase        
        - Removing numbers with regular expressions
        - Removing punctuation with regular expressions
        - Removing other artifacts
    And separate the document into words by simply splitting at spaces
    Params:
        text (string): a sentence or a document
    Returns:
        tokens (list of strings): the list of tokens (word units) forming the document
    """        
    # Lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r"[0-9]+", "", text)
    # Remove punctuation
    REMOVE_PUNCT = re.compile("[.;:!\'?,\"()\[\]]<>")
    text = REMOVE_PUNCT.sub("", text)
    # Remove small words (1 and 2 characters)
    text = re.sub(r"\b\w{1,2}\b", "", text)
    # Remove HTML artifacts specific to the corpus we're going to work with
    REPLACE_HTML = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    text = REPLACE_HTML.sub(" ", text)
         
    return text

def remove_adjacent_duplicates(s):
    words = s.split()
    if not words:  # Handle the case where the string is empty
        return ""

    # Start with the first word in the result
    result = [words[0]]

    # Go through the remaining words and add only if different from the last added
    for current_word in words[1:]:
        if current_word != result[-1]:
            result.append(current_word)
    
    return ' '.join(result)

def remove_duplicate_spaces(s):
    # Replace occurrences of one or more spaces with a single space
    return re.sub(r'\s+', ' ', s).strip()

def remove_urls(text):
    # Regular expression to match URLs
    url_pattern = r"http\S+"
    # Replace URLs with an empty string
    text_without_urls = re.sub(url_pattern, '', text)
    return text_without_urls

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename,remove_non_english):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    allLines = []
    for line in lines:
        line = remove_urls(line)
        line = unicodeToAscii(line)
        line = clean_and_tokenize(line)
        line = remove_adjacent_duplicates(line)
        line = remove_duplicate_spaces(line)
        if remove_non_english:
            line = " ".join([ele for ele in line.split() if is_english_word(ele)])
        allLines.append(line)
    return allLines

def giveData(path, remove_non_english):
    category_lines = {}
    all_categories = []
    
    n_words = 1
    dictionary = {"UNK":0}
    
    for filename in findFiles(path + "*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename,remove_non_english)
        category_lines[category] = lines
        
        for line in lines:
            tokens = line.split()
            for token in tokens:
                if token not in dictionary:
                    dictionary[token] = n_words
                    n_words += 1
        
    return all_categories, category_lines, dictionary

"""

def giveData(path, remove_non_english):
    category_lines = {}
    all_categories = []
    for filename in findFiles(path + "*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename,remove_non_english)
        category_lines[category] = lines
    return all_categories, category_lines
    
    
def buildDictionary(all_categories, category_lines):
    n_words = 1
    dictionary = {"UNK":0}
    all_lines = []
    #all_lines = [ele for ele in [category_lines[cate] for cate in all_categories]]
    for cate in all_categories:
        all_lines += category_lines[cate]
        
    for line in all_lines:
        tokens = line.split()
        for token in tokens:
           
                if token not in dictionary:
                    dictionary[token] = n_words
                    n_words += 1
    return dictionary
    
    
    def buildDictionary222222(all_categories, category_lines):
    n_words = 1
    dictionary = {"UNK":0}
    #all_lines = [ele for ele in [category_lines[cate] for cate in all_categories]]
    for cate in all_categories:
        for line in category_lines[cate]:
            tokens = line.split()
            for token in tokens:
                if token not in dictionary:
                    dictionary[token] = n_words
                    n_words += 1
    return dictionary
"""


def cleanText(text,remove_non_english):
    text = remove_urls(text)
    text = unicodeToAscii(text)
    text = clean_and_tokenize(text)
    text = remove_adjacent_duplicates(text)
    text = remove_duplicate_spaces(text)
    if remove_non_english:
        text = " ".join([ele for ele in text.split() if is_english_word(ele)])
    return text