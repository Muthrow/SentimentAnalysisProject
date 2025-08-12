# import torch
from transformers import pipeline, AutoTokenizer
import streamlit as st
import tweepy
import praw
import altair as alt
import numpy as np
import pandas as pd

DISCLAIMER = 'Due to Twitter API limits, only a small amount of data is available for each tweet, with long wait times between analyses'

T_BEARER_TOKEN = st.secrets["T_BEARER_TOKEN"]
T_ACCESS_TOKEN = st.secrets["T_ACCESS_TOKEN"]
T_ACCESS_TOKEN_SECRET = st.secrets["T_ACCESS_TOKEN_SECRET"]
T_CONSUMER_SECRET = st.secrets["T_CONSUMER_SECRET"]
T_CONSUMER_KEY = st.secrets["T_CONSUMER_KEY"]


auth = tweepy.OAuthHandler(T_CONSUMER_KEY, T_CONSUMER_SECRET)
auth.set_access_token(T_ACCESS_TOKEN, T_ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)
client = tweepy.Client(bearer_token=T_BEARER_TOKEN)

model = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_pipeline = pipeline("sentiment-analysis", model = model)
tokenizer = AutoTokenizer.from_pretrained(model)

tweet = ''
if tweet not in st.session_state:
    st.session_state['tweet'] = ''

# Streamlit app setup
st.title("Twitter Analysis")
st.write(DISCLAIMER)
url_input = st.text_input("Enter Tweet URL here")
def retrieveV1(tweet_id):
        # tweet = client.get_tweet(tweet_id) 
        # tweet = tweet.data['text'] 
        # Fetch the tweet 
        tweet = api.get_status(tweet_id) 
        # Fetch comments (replies) to the tweet 
        comments = api.get_replies(tweet) 
        for comment in comments: 
            st.write(f"Comment Text: {comment.text}") 
            st.write(f"Timestamp: {comment.created_at}") 
            st.write(f"Likes: {comment.favorite_count}") 
            st.write(f"Retweets: {comment.retweet_count}") 
            st.write(f"Reply ID: {comment.id}")
            break
        st.session_state['tweet'] = tweet.text
        
def retrieveV2(tweet_id): 
    comments = []
    client = tweepy.Client(bearer_token=T_BEARER_TOKEN) 
    query = f'conversation_id:{tweet_id}' 
    tweets = client.search_recent_tweets(query=query, tweet_fields=['created_at', 'public_metrics']) 
    if tweets.data: 
        for tweet in tweets.data: 
            comments.append({
                'Comment': tweet.text, 
                'Timestamp': tweet.created_at, 
                'Likes': tweet.public_metrics['like_count'], 
                'Retweets': tweet.public_metrics['retweet_count'], 
                'Reply ID': tweet.id
            })
    
        else: 
            st.write("No replies found.")
        
    df = pd.DataFrame(comments)

    return df
    # st.session_state['tweet'] = tweet.text
def analyze_sentiment(df):
    df['Sentiment'] = df['Truncated Text'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
    df['Confidence Score'] = df['Truncated Text'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
    return df

def filter_by_confidence(df, threshold=0.95):
    return df[df['Confidence Score'] >= threshold]

def apply_log_scale(df):
    df['Quality Score'] = df['Confidence Score'].apply(lambda x: np.log10(1 + x))
    return df

def calculate_influence(df):
    df['Influence Score'] = df.apply(lambda row: row['Quality Score'] * row['Score'] if row['Sentiment'] == 'POSITIVE' else -row['Quality Score'] * row['Score'], axis=1)
    return df    

def truncate_comment_lambda(comment): 
    tokens = tokenizer.tokenize(comment) 
    if len(tokens) > MAX_TENSOR_LEN - 2: 
        tokens = tokens[:MAX_TENSOR_LEN - 2]
    return tokenizer.convert_tokens_to_string(tokens)

def truncate_comments(df):
    df['Truncated Text'] = df['Comment'].apply(lambda x: truncate_comment_lambda(x))
    return df

def calculate_score(df):
    df['Score'] = df['Likes'] + (df['Retweets'] * 2)

def process_data(df):
    df = truncate_comments(df)
    df = analyze_sentiment(df)
    df = filter_by_confidence(df)
    df = apply_log_scale(df)
    df = calculate_score(df)
    df = calculate_influence(df)
    return df

if st.button("Retrieve"): 
    if url_input: 
        tweet_id = url_input.split('/')[-1] 
        df = process_data(retrieveV2(tweet_id))
        st.write("Tweet retrieved successfully!") 
        st.write(df)

        scatter_plot = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X('Timestamp:T', title='Date'),  # Ensure correct column name
            y=alt.Y('Influence Score:Q', title='Influence Score', scale=alt.Scale(zero=True)),
            color='Sentiment:N',
            tooltip=['Comment', 'Influence Score', 'Sentiment']
        ).properties(
            width=600,
            height=400
        ).configure_axis(
            gridColor='gray',
            domainWidth=0.5,
            labelFontSize=14,
            titleFontSize=16,
            labelAngle=0,
            titlePadding=10
        ).configure_view(
            strokeOpacity=0
        )
        # Display the scatterplot in Streamlit
        st.altair_chart(scatter_plot, use_container_width=True)
    else: 
        st.write("Invalid URL.") 
    
# retain text accross session states
# text_input = st.text_area("Enter Analysis Text Here", height=200, value=f'{st.session_state["tweet"]}') 

# if st.button("Analyze"): 
#     if text_input: 
#         results = sentiment_pipeline(text_input) 
#         st.table(results) 
#     else: 
#         st.write("Please enter text for analysis.")

# if st.button("Retrieve"):
#     tweet='Success!'


# text_input = st.text_area("Enter Analysis Text Here", height=200, value=f'{tweet}')

# if st.button("Analyze"):
#     results = sentiment_pipeline(text_input)
#     st.table(results)

