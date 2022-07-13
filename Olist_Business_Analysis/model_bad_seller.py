
# import libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import string
import unidecode
import numpy as np
import pandas as pd

from Olist_Business_Analysis.data_preparation import Olist

# import olist data
df = pd.read_csv("https://wagon-public-datasets.s3.amazonaws.com/Machine%20Learning%20Datasets/reviews.csv")
df.head()

# filter out data
df = df[(df['review_creation_date'] <= df['order_estimated_delivery_date'])]
df = df[['order_id','product_category_name','review_comment_title','review_comment_message','review_score']]

# combine review title and review message
df = df.dropna(subset=['review_comment_title','review_comment_message'])
df['title_comment'] = df["review_comment_title"].fillna('') + " " + df['review_comment_message'].fillna('')

# clean text
def clean (text):

    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation

    lowercased = text.lower() # Lower Case
    unaccented_string = unidecode.unidecode(lowercased) # remove accents
    tokenized = word_tokenize(unaccented_string) # Tokenize
    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
    stop_words = set(stopwords.words('portuguese')) # Make stopword list
    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words

    return " ".join(without_stopwords)

df['clean_text'] = df['title_comment'].apply(clean)

# Groupby product category and aggregate mean, min, max review scores
product_performance = df.groupby('product_category_name').agg({'review_score': ['count','mean', 'min', 'max']}).sort_values([('review_score','mean')],ascending=False)

# Keep categories that have more than 100 reviews
product_performance = product_performance[product_performance[('review_score','count')] >=100]

# Filter out category
relogios_presentes = df[df['product_category_name'].isin(['relogios_presentes'])]

# Filter out bad reviews
relogios_presentes_bad_reviews = relogios_presentes[relogios_presentes['review_score'].isin([1])]

# Tuned TFidfvectorizer
vec = TfidfVectorizer(ngram_range = (2,2), min_df=0.01, max_df = 0.05).fit(relogios_presentes_bad_reviews.clean_text)
vectors = vec.transform(relogios_presentes_bad_reviews.clean_text) # Transform text to vectors
sum_tfidf = vectors.sum(axis=0) # Sum of tfidf weighting by word
tfidf_list = [(word, sum_tfidf[0, idx]) for word, idx in     vec.vocabulary_.items()]  # Get the word and associated weight
sorted_tfidf_list =sorted(tfidf_list, key = lambda x: x[1], reverse=True)  # Sort

# Get seller ID
data = Olist().get_data()
sellers = data['orders'].merge(relogios_presentes_bad_reviews)

# Filter out reviews with words associated with conterfeit watches
bad_sellers = sellers[sellers['clean_text'].str.contains("nao original|falso|outro modelo|produto falsificado|produto diferente")]

# Groupby seller id
bad_sellers.groupby('seller_id').agg({'seller_id': ['count']})

# Filter out the one seller with 11 counterfeit related reviews
bad_sellers[bad_sellers["seller_id"]== "2eb70248d66e0e3ef83659f71b244378"].sort_values(by='shipping_limit_date')
