from transformers import pipeline, AutoTokenizer
import streamlit as st
import tweepy
import praw
import pandas as pd

st.set_page_config( 
    page_title="Sentiment Analysis",
    page_icon="✏️", 
    layout="centered", 
    initial_sidebar_state="expanded"
)
youtube_page = st.Page("youtube.py", title="YouTube Analysis", icon="▶️")
reddit_page = st.Page("reddit.py", title="Reddit Analysis", icon="🤖")
twitter_page = st.Page("twitter.py", title="Twitter Analysis", icon="🐦")
text_page = st.Page("text.py", title="Text Analysis", icon="📄")


pg = st.navigation(
    [
        youtube_page, 
        reddit_page, 
        twitter_page, 
        text_page
    ]
)
pg.run()