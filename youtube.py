from googleapiclient.discovery import build
import pandas as pd
import streamlit as st
import re
from transformers import pipeline, AutoTokenizer
import altair as alt
import numpy as np


API_KEY = st.secrets["G_API_KEY"]
MAX_TENSOR_LEN = 512
MAX_COMMENTS = 100

model = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model = model,
    device_map="auto",)
tokenizer = AutoTokenizer.from_pretrained(model)

youtube = build('youtube', 'v3', developerKey=API_KEY)

def extract_video_id(url):
    video_id_match = re.search(r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([\w-]{11})', url)
    return video_id_match.group(1) if video_id_match else None

def get_all_comments(video_id):
    comments = []
    try:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            textFormat='plainText'
        ).execute()

        while response:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt'], comment['likeCount']])
            
            if 'nextPageToken' in response:
                response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=100,
                    textFormat='plainText'
                ).execute()
            else:
                break
    except Exception as e:
        st.error(f"An error occurred: {e}")

    return comments
def get_video_info(video_id):
    try:
        response = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()
        
        if response['items']:
            video_info = response['items'][0]
            st.write(f"Title: {video_info['snippet']['title']}")
            st.write(f"Views: {video_info['statistics']['viewCount']}")
            st.write(f"Likes: {video_info['statistics']['likeCount']}")
            st.write(f"Comments: {video_info['statistics'].get('commentCount', 'N/A')}")
        else:
            st.error("No video information found. Please check the video ID.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def truncate_comment(comment): 
    tokens = tokenizer.tokenize(comment) 
    if len(tokens) > MAX_TENSOR_LEN - 2: 
        tokens = tokens[:MAX_TENSOR_LEN - 2]
    return tokenizer.convert_tokens_to_string(tokens)

def filter_recent(df):
    df = df.sort_values(by='Published At', ascending=False) 
    top_200 = df.head(200) 
    return top_200

def process_data(df):
    df['truncated_text'] = df['Comment'].apply(lambda x: truncate_comment(x) )
    df = filter_recent(df)
    df['sentiment'] = df['truncated_text'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
    df['sentiment_score'] = df['truncated_text'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
    processed_df = df.query('sentiment_score > 0.96')
    processed_df['quality_score'] = processed_df['sentiment_score'].apply(lambda x: np.log10(1 + x)) 
    processed_df['influence_score'] = processed_df.apply(lambda row: row['quality_score'] * row['Likes'] if row['sentiment'] == 'POSITIVE' else -row['quality_score'] * row['Likes'], axis=1)
    return processed_df


st.title("YouTube Analysis")
url_input = st.text_input("Enter YouTube Video URL here")
MAX_COMMENTS = st.number_input("Enter the maximum number of comments to analyze, as sorted by time and likes.", min_value=1, max_value=1000, value=100)
if st.button("Retrieve Video"):
    if url_input:
        video_id = extract_video_id(url_input)
        if video_id:
            comments = get_all_comments(video_id)
            if comments:
                df = pd.DataFrame(comments, columns=['Comment', 'Author', 'Published At', 'Likes'])
                df = process_data(df.nlargest(MAX_COMMENTS, 'Likes'))
                top_10_comments = df.nlargest(10, 'Likes')
                st.write("Top 10 Comments by Likes")
                st.write(top_10_comments)
                # Embed YouTube video
                st.markdown(f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', unsafe_allow_html=True)
                get_video_info(video_id)
                st.write(f'Overall Sentiment: {df["influence_score"].sum()}')
                st.write("Video information retrieved successfully!")
                scatter_plot = alt.Chart(df).mark_circle(size=60).encode(
                    x=alt.X('Published At:T', title='Date'),  # Ensure correct column name
                    y=alt.Y('influence_score:Q', title='Influence Score', scale=alt.Scale(zero=True)),
                    color='sentiment:N',
                    tooltip=['Comment', 'influence_score', 'sentiment']
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
                st.altair_chart(scatter_plot, use_container_width=True)
            else:
                st.write("No comments found.")
        else:
            st.error("Invalid YouTube URL. Please enter a valid URL.")
    else:
        st.error("Invalid URL.")
