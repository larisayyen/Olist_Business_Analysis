
# Data Manipulation
import numpy as np
import pandas as pd
pd.set_option("max_columns",None)

# Olist packages
from Olist_Business_Analysis.data_preparation import Olist
from Olist_Business_Analysis.review import Review
from Olist_Business_Analysis.orders_analysis import Order
from Olist_Business_Analysis.product import Product
from Olist_Business_Analysis.seller_analysis import Seller

# Machine Learning
from sklearn.pipeline import make_pipeline

# Language Processing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import unidecode

# Vectorizers and NLP Models
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# import data
reviews = Review().get_training_data()
data = Olist().get_data()

orders = data['orders']
reviews_data = data['order_reviews']

# merge orders and reviews
df = reviews.merge(reviews_data).merge(orders)

# filter out data
rated_after_received = df["order_delivered_customer_date"] < df["review_creation_date"]
df = df[rated_after_received]

# remove reviews for undelivered orders
delivered = df['order_status'] == "delivered"
df = df[delivered]

# get full reviews
df = df.dropna(subset=['review_comment_title','review_comment_message'])
df['full_review'] = df["review_comment_title"].fillna('') + " " \
            + df['review_comment_message'].fillna('')

# Focus on certain columns
columns_of_interest = ['review_id',
                       'length_review',
                       'review_score',
                       'order_id',
                       'product_category_name',
                       'full_review']
df = df[columns_of_interest]

# NLP PROCESS BEGIN

## clean text
def cleaning(sentence):

    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercasing
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## removing numbers
    sentence = unidecode.unidecode(sentence) # remove accents
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## removing punctuation
    # Advanced cleaning
    tokenized_sentence = word_tokenize(sentence) ## tokenizing
    stop_words = set(stopwords.words('portuguese')) ## defining stopwords
    tokenized_sentence = [w for w in tokenized_sentence
                                  if not w in stop_words] ## remove stopwords
    lemmatized_sentence = [WordNetLemmatizer().lemmatize(word, pos = "v")  # v --> verbs
              for word in tokenized_sentence]
    cleaned_sentence = ' '.join(word for word in lemmatized_sentence)
    return cleaned_sentence

df["full_review_cleaned"] = df["full_review"].apply(cleaning)

## get proportion of reviews
round(df["review_score"].value_counts(normalize = True),2)

threshold = 3
bad = df["review_score"] <= threshold
df = df[bad]

## vectorizing
vectorizer = TfidfVectorizer(max_df = 0.75, max_features = 5000, ngram_range=(1,2))
vectorized_reviews = pd.DataFrame(vectorizer.fit_transform(df["full_review_cleaned"]).toarray(),
                                 columns = vectorizer.get_feature_names())

#print(f" vectorized_reviews.shape = {vectorized_reviews.shape}")

## LDA
n_components = 3
lda = LatentDirichletAllocation(n_components = n_components)
lda.fit(vectorized_reviews)

document_mixture = lda.transform(vectorized_reviews)

round(pd.DataFrame(document_mixture,
                   columns = [f"Topic {i+1}" for i in range(n_components)]),2)

# get the most important topic for each review
df["most_important_topic"] = np.argmax(document_mixture, axis = 1)
topic_mixture = pd.DataFrame(lda.components_,
                             columns = vectorizer.get_feature_names())

# top words with weights for one topic
def topic_word(vectorizer, model, topic, topwords, with_weights = True):
    topwords_indexes = topic.argsort()[:-topwords - 1:-1]
    if with_weights == True:
        topwords = [(vectorizer.get_feature_names()[i], round(topic[i],2)) for i in topwords_indexes]
    if with_weights == False:
        topwords = [vectorizer.get_feature_names()[i] for i in topwords_indexes]
    return topwords

# different topics found by LDA with their topwords
def print_topics(vectorizer, model, topwords):
    for idx, topic in enumerate(model.components_):
        print("-"*20)
        print("Topic %d:" % (idx))
        print(topic_word(vectorizer, model, topic, topwords))

# top words associated to a topic
topic_word_mixture = [topic_word(vectorizer, lda, topic, topwords = 5, with_weights = False)
                      for topic in lda.components_]

df["most_important_words"] = df["most_important_topic"].apply(lambda i: topic_word_mixture[i])
df[["review_id",
        "review_score",
        "product_category_name",
        "full_review_cleaned",
        "most_important_topic",
        "most_important_words"]
      ].head(3)

# PIPELINE
# Parameters
max_df = 0.75
max_features = 5000
ngram_range = (1,2)

# Pipeline Vectorizer + LDA
pipeline = make_pipeline(
    TfidfVectorizer(max_df = max_df,
                    max_features = max_features,
                    ngram_range = ngram_range),
    LatentDirichletAllocation(n_components = n_components)
)

# Fit the pipeline on the cleaned texts
pipeline.fit(df["full_review_cleaned"])

pipeline._final_estimator

pipeline._final_estimator.components_

# Transform the original cleaned texts with the pipeline
# no need to get the vectorized texts first since it's done through the Pipeline
document_mixture = pipeline.transform(df["full_review_cleaned"])

topic_mixture = pd.DataFrame(pipeline._final_estimator.components_)

# groupby product category

# Product categories by performance - look at the count, mean, median and std
product_categories = df.groupby(by = 'product_category_name').agg({
        'review_score': ["count", "mean", "median", "std"]
    })

# Removing products which were sold less than a certain times for the analysis
cutoff = 50
product_categories = product_categories[product_categories[("review_score", "count")] > cutoff]

# Sorting the product categories by performance
product_categories = product_categories.sort_values(by = [('review_score', 'mean'),
                                                          ('review_score', 'std')],
                                                    ascending = [False, True])

# get worst products
worst_products = product_categories.tail(5).sort_values(by = [("review_score", "count")],
                                                       ascending = False)

worst_products_reviews = df[df.product_category_name.isin(worst_products.index)]
worst_products_reviews[["review_id",
                        "review_score",
                        "product_category_name",
                        "full_review_cleaned",
                        "most_important_topic",
                        "most_important_words"]
      ]

worst_products_reviews["most_important_topic"].value_counts()

bad_frequency = list(worst_products_reviews["most_important_topic"].value_counts().index)

[topic_word_mixture[i] for i in bad_frequency]

# get worst seller
sellers = Seller().get_training_data()

worst_sellers = sellers[["seller_id", "review_score", "profits"]].sort_values(
    by = "profits",
    ascending = True).head(10)

# worst products sold by worst seller
products = Product().get_training_data() [["product_id", "category"]]

sellers_product_category = data["order_items"].merge(products,
                                             on = "product_id", how = "left")[["seller_id", "category"]]

sellers_product_category.groupby("seller_id").count()

# categories and topics for the worst sellers
def focus_seller(seller_id):
    return sellers_product_category[sellers_product_category.seller_id == seller_id].value_counts()

bad_reviews_sellers = df.merge(data["order_items"])

def bad_reviews_seller(bad_reviews_sellers, seller_id):
    mask = (bad_reviews_sellers.seller_id == seller_id)
    temp = bad_reviews_sellers[mask]
    most_frequent_topic_seller = list(temp.most_important_topic.value_counts().head(1).index)[0]
    return topic_word_mixture[most_frequent_topic_seller]

for worst_seller in worst_sellers["seller_id"]:
    print("-"*50)
    print(f"Focusing on the seller #{worst_seller}...")
    print(focus_seller(worst_seller))
    print(bad_reviews_seller(bad_reviews_sellers, worst_seller))
