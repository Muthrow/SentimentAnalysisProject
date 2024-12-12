import torch
from transformers import pipeline, AutoTokenizer
import streamlit as st
import tweepy
import praw
import pandas as pd

model = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_pipeline = pipeline("sentiment-analysis", model = model)
tokenizer = AutoTokenizer.from_pretrained(model)
MAX_TENSOR_LEN = 512


def truncate_text(comment): 
    tokens = tokenizer.tokenize(comment) 
    if len(tokens) > MAX_TENSOR_LEN - 2: 
        tokens = tokens[:MAX_TENSOR_LEN - 2]
    return tokenizer.convert_tokens_to_string(tokens)

st.title("Custom Text Analysis")
# st.write('Text my be truncated')
text_input = st.text_area("Enter Text Here", height=200, value=f'Write whatever you like here...') 


if st.button("Analyze"): 
    if text_input: 
        results = sentiment_pipeline(truncate_text(text_input)) 
        st.table(results) 
    else: 
        st.write("Please enter text for analysis.")
