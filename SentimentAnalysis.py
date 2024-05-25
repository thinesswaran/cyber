import streamlit as st
import pandas as pd
import altair as alt
from textblob import TextBlob 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set page config to wide layout with a white background
st.set_page_config(page_title='Cyberbullying Detection', page_icon="😊🙂😡",
                               layout="wide", initial_sidebar_state='collapsed',
                               theme={
                                   'base': 'light',
                                   'backgroundColor': '#FFFFFF',
                                   'secondaryBackgroundColor': '#FFFFFF',
                                   'textColor': '#000000'
                               })

# Function to convert sentiment to dataframe
def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

# Function to analyze token sentiment
def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)

        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)

    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result 

# Main function to run the app
def main():
    st.title("Cyberbullying Detection Using Machine Learning")
    st.subheader("Sentiment Analysis using NLP")

    st.markdown("Entering the text to analyze will provide a result.")
    with st.form(key='nlpForm'):
        raw_text = st.text_area("Enter Text Here")
        submit_button = st.form_submit_button(label='Analyze')

    # Layout
    col1, col2 = st.columns(2)
    if submit_button:
        with col1:
            st.info("Results")
            sentiment = TextBlob(raw_text).sentiment
            st.write(sentiment)

            # Emoji
            if sentiment.polarity > 0:
                st.markdown("Sentiment:: Positive :smiley: ")
            elif sentiment.polarity < 0:
                st.markdown("Sentiment:: Negative :angry: ")
            else:
                st.markdown("Sentiment:: Neutral 😐 ")

            # Dataframe
            result_df = convert_to_df(sentiment)
            st.dataframe(result_df)

            # Visualization
            c = alt.Chart(result_df).mark_bar().encode(
                x='metric',
                y='value',
                color='metric')
            st.altair_chart(c, use_container_width=True)

        with col2:
            st.info("Token Sentiment")
            token_sentiments = analyze_token_sentiment(raw_text)
            st.write(token_sentiments)

# Run the main function
if __name__ == '__main__':
    main()
