# AUTOGENERATED! DO NOT EDIT! File to edit: ../core/01_topic_modeling.ipynb.

# %% auto 0
__all__ = ['load_data_from_json', 'remove_stop_words', 'deep_clean_text', 'get_body_from_issues', 'generate_topic_name']

# %% ../core/01_topic_modeling.ipynb 2
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import json

import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd

# %% ../core/01_topic_modeling.ipynb 3
from openai import OpenAI
from bertopic.backend import OpenAIBackend

import tiktoken
from bertopic import BERTopic

# %% ../core/01_topic_modeling.ipynb 4
def load_data_from_json(filename):
    """Load data from a JSON file."""
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# %% ../core/01_topic_modeling.ipynb 5
def remove_stop_words(issue_body):
    """
    Clean the text of a GitHub issue for topic modeling.
    """
    # Remove URLs
    issue_body = re.sub(r'http\S+', ' ', issue_body)

    # Remove user mentions
    issue_body = re.sub(r'@\S+', ' ', issue_body)

    # Remove special characters and numbers
    issue_body = re.sub(r'[^A-Za-z\s]', ' ', issue_body)

    # Tokenize text
    tokens = word_tokenize(issue_body)

    # Convert to lower case
    tokens = [word.lower() for word in tokens]

    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Optional: Lemmatization/Stemming (can be added based on requirement)

    # Rejoin tokens into a string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

# %% ../core/01_topic_modeling.ipynb 6
def deep_clean_text(text):
    """
    Clean and curate the text from a GitHub issue for better processing with BERTopic.

    Parameters:
    issue_body (str): The text content of a GitHub issue.

    Returns:
    str: Cleaned and curated text.
    """

    # Remove HTML tags using BeautifulSoup
    issue_body = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    issue_body = re.sub(r'http\S+', ' ', issue_body)

    # Remove GitHub flavored markdown
    issue_body = re.sub(r'```[a-z]*\n[\s\S]*?\n```', ' ', issue_body)  # Code blocks
    issue_body = re.sub(r'`[^`]*`', ' ', issue_body)  # Inline code
    issue_body = re.sub(r'\*\*[^*]*\*\*', ' ', issue_body)  # Bold text
    issue_body = re.sub(r'\*[^*]*\*', ' ', issue_body)  # Italic text
    issue_body = re.sub(r'~~[^~]*~~', ' ', issue_body)  # Strikethrough
    issue_body = re.sub(r'\[[^\]]*\]\([^\)]*\)', ' ', issue_body)  # Hyperlinks

    return remove_stop_words(issue_body)

# %% ../core/01_topic_modeling.ipynb 7
def get_body_from_issues(issues: list):
  return [remove_stop_words(issue['body']) for issue in issues if issue['body'] is not None]

# %% ../core/01_topic_modeling.ipynb 8
def generate_topic_name(keywords, OPEN_AI_KEY):
    #openai.api_key = OPEN_AI_KEY
    client = OpenAI(api_key=OPEN_AI_KEY)
    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "user", "content": f"Generate a short descriptive name for the category of a topic, based on these keywords: {', '.join(keywords)}"}])
    print(response.choices[0].message.content)
    return response.choices[0].message.content