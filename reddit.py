import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import praw
from transformers import pipeline, AutoTokenizer


# Define your reddit_page function
st.title("Reddit Analysis")    
# Set up Reddit API credentials
R_CLIENT_ID = st.secrets["R_CLIENT_ID"] 
R_CLIENT_SECRET = st.secrets["R_CLIENT_SECRET"] 
R_USER_AGENT = st.secrets["R_USER_AGENT"] 

MAX_TENSOR_LEN = 512
MAX_COMMENTS = 100

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)    
tokenizer = AutoTokenizer.from_pretrained(model_name)

reddit = praw.Reddit(client_id=R_CLIENT_ID, client_secret=R_CLIENT_SECRET, user_agent=R_USER_AGENT)

def extract_post_id(url): 
    return url.split('/comments/')[1].split('/')[0] 

def fetch_comments_from_post(url): 
    post_id = extract_post_id(url) 
    submission = reddit.submission(id=post_id) 
    submission.comments.replace_more(limit=None) 
    comments = [] 
    for comment in submission.comments.list(): 
        comments.append({ 'Comment': comment.body, 'Author': comment.author.name if comment.author else 'Unknown', 'Score': comment.score, 'Created At': comment.created_utc }) 
    return comments

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

def process_data(df):
    df = truncate_comments(df)
    df = analyze_sentiment(df)
    # df = filter_by_confidence(df)
    df = apply_log_scale(df)
    df = calculate_influence(df)
    return df

# Fetch and analyze comments when button is clicked
post_url_input = st.text_input("Enter a Reddit post URL") 
MAX_COMMENTS = st.number_input("Enter the maximum number of comments to analyze, as sorted by score.", min_value=1, max_value=1000, value=100)

if st.button("Analyze Comments"):
    st.write("Fetching and analyzing comments...")
    comments_df = pd.DataFrame(fetch_comments_from_post(post_url_input))
    final_df = process_data(comments_df.nlargest(MAX_COMMENTS, 'Score'))        
    st.write("Analysis Complete")        
    if not final_df.empty:
        st.write(final_df)
        scatter_plot = alt.Chart(final_df).mark_circle(size=60).encode(
            x=alt.X('Created At:T', title='Date'),  # Ensure correct column name
            y=alt.Y('Influence Score:Q', title='Influence Score', scale=alt.Scale(zero=True)),
            color='Sentiment:N',
            tooltip=['Comment', 'Influence Score', 'Sentiment', 'Author']
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
        st.write("No comments met the confidence threshold.")