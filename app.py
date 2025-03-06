import streamlit as st
import pymongo
import re
import nltk
import spacy
import gensim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora
from wordcloud import WordCloud
from newspaper import Article
from GoogleNews import GoogleNews
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import spacy
import os

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ------------------------------- STEP 1: CONNECT TO MONGODB ATLAS -------------------------------
uri = "mongodb+srv://madstriker:Vijay1muru@cluster0.68cyr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))
db = client["women_cricket_news"]
collection = db["articles"]

# ------------------------------- STEP 2: SCRAPING GOOGLE NEWS -------------------------------
def scrape_google_news(query="Women Indian Cricket", num_articles=10):
    googlenews = GoogleNews(lang='en', region='IN')
    googlenews.search(query)
    
    articles = []
    news_results = googlenews.result()
    
    for news in news_results[:num_articles]:
        try:
            article = Article(news['link'])
            article.download()
            article.parse()
            articles.append({
                "platform": "GoogleNews",
                "title": news["title"],
                "date": news["date"],
                "content": article.text,
                "source": news["media"],
                "link": news["link"]
            })
        except:
            continue
    return articles

# ------------------------------- STEP 3: STORE DATA IN MONGODB ATLAS -------------------------------
def store_in_mongo(data):
    if isinstance(data, list) and data:
        collection.insert_many(data)
    elif isinstance(data, dict):
        collection.insert_one(data)

# ------------------------------- STEP 4: TEXT PREPROCESSING -------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
spacy_model = spacy.load('en_core_web_sm')

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)  
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# ------------------------------- STEP 5: SENTIMENT ANALYSIS -------------------------------
def analyze_sentiment(text):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(text)
    return "Positive" if score['compound'] > 0 else "Negative" if score['compound'] < 0 else "Neutral"

# ------------------------------- STEP 6: TOPIC MODELING -------------------------------
def topic_modeling(texts, num_topics=3):
    tokenized_texts = [word_tokenize(text) for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model.print_topics()

# ------------------------------- STEP 7: BAG OF WORDS VISUALIZATION -------------------------------
def generate_bag_of_words(texts):
    word_freq = {}
    for text in texts:
        words = word_tokenize(text)
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])  # Top 20 words
    
    plt.figure(figsize=(12,6))
    sns.barplot(x=list(word_freq.values()), y=list(word_freq.keys()))
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title("Top Words in the News Articles")
    plt.savefig("bag_of_words.png")  # Save figure
    plt.close()

# ------------------------------- STEP 8: STREAMLIT DASHBOARD -------------------------------
st.title("📊 Women Indian Cricket - Textual Data Analysis Report")

st.subheader("1️⃣ Data Collection & Processing")
st.write("Fetching latest news articles from **Google News** & storing them in MongoDB Atlas.")

if st.button("Scrape Latest News"):
    news_data = scrape_google_news()
    store_in_mongo(news_data)
    st.success(f"Successfully fetched & stored {len(news_data)} new articles!")

# Fetch stored articles
articles = list(collection.find({}))
texts = [article["content"] for article in articles if article.get("content")]

if not texts:
    st.warning("No data available yet. Click the button above to fetch news.")
    st.stop()

# Preprocessing
processed_texts = [preprocess_text(text) for text in texts]

st.subheader("2️⃣ Sentiment Analysis")
sentiments = [analyze_sentiment(text) for text in processed_texts]
sentiment_counts = pd.Series(sentiments).value_counts()

fig1, ax1 = plt.subplots()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax1)
ax1.set_title("Sentiment Distribution")
st.pyplot(fig1)

st.subheader("3️⃣ Topic Modeling (LDA)")
topics = topic_modeling(processed_texts)
st.write("🔹 **Identified Topics from News Articles:**")
for topic in topics:
    st.write(topic)

st.subheader("4️⃣ Word Cloud")
wordcloud = WordCloud(width=800, height=400, background_color="black").generate(" ".join(processed_texts))
fig2, ax2 = plt.subplots()
ax2.imshow(wordcloud, interpolation="bilinear")
ax2.axis("off")
st.pyplot(fig2)

st.subheader("5️⃣ Bag of Words Analysis")
generate_bag_of_words(processed_texts)
st.image("bag_of_words.png", caption="Word Frequency Analysis")

st.subheader("6️⃣ Key Insights & Conclusion")
st.write("""
- **Most topics discussed in the news articles revolve around recent cricket matches, player performances, and upcoming tournaments.**
- **The sentiment distribution suggests that media coverage is mostly [Positive/Negative/Neutral] towards Women's Cricket.**
- **Key players like 'Smriti Mandhana', 'Harmanpreet Kaur', and 'Shafali Verma' appear frequently in the data.**
- **The Bag of Words analysis highlights the most frequently mentioned words in news articles.**
""")

st.success("🎉 Report Completed! You can now analyze more data by scraping new articles.")


st.subheader("Developed by")
st.write("""
- Madhavan
""")
