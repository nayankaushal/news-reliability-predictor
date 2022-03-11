from matplotlib.pyplot import text, title, xticks
import streamlit as st

header = st.container()
user_input = st.container()
importing_pickle = st.container()
reponse = st.container()
viz = st.container()
n_grams = st.container()
model_response = st.container()

with header:
    st.markdown("# News Reliability Predictor 📰")
    st.caption("The News Reliability Predictor aims to fight the spread of misinformation and disinformation by predicting, using NLP and machine learning techniques, if a news article is reliable or unreliable.")

with user_input:
    st.subheader("Please enter the news text:")
    txt = st.text_input("News Text")
    #col1, col2, col3 = st.columns([1,1,1])
    #with col1:
    #    n_grams_button = st.button("Top nGrams")
    #with col2:
    #    sentiment_button = st.button("Sentiment Analysis")
    #with col3:
    #    predict_button = st.button("Predict Reliability")

with importing_pickle:
    import matplotlib.pyplot as plt
    from textblob import TextBlob
    import time
    import pandas as pd
    import _pickle as cPickle
    import numpy as np
    import seaborn as sns
    import nltk
    from nltk.util import ngrams
    from nltk import word_tokenize
    from collections import Counter
    import gzip
    
    def split_into_tokens(message):
        return TextBlob(message).words

    def split_into_lemmas(message):
        message = message.lower()
        words = TextBlob(message).words
        # for each word, take its "base form" = lemma 
        return [word.lemma for word in words]

    def load_zipped_pickle(filename):
        with gzip.open(filename, 'rb') as f:
            loaded_object = cPickle.load(f)
            return loaded_object

    svm_detector_reloaded = load_zipped_pickle('news_reliability_detector.pkl')
    #svm_detector_reloaded = cPickle.load(open('/Users/nayankaushal/Desktop/Projects/Misinformation/sms_news_reliability_detector.pkl', 'rb'))

    if txt:
        test = pd.DataFrame({'text': [txt]})
        #predicting labels for test data
        test_predictions = svm_detector_reloaded.predict(test['text'])

with viz:
    st.subheader("Top nGrams: ")
    col1, col2 = st.columns([1,3])
    with col1:            
        n  = st.slider("nGrams Value: ", 1, 5, value = 1)        
        top_n = st.slider("For top n nGrams, n: ", 1, 10, value = 10)
        change_flag = 1

    with col2:            
        #downloading stopwords
        import nltk
        from nltk.corpus import stopwords
        nltk.download('stopwords')


        #creating a function which returns a list containing the top ngrams of each row
        def top_ngrams(df, n, top):
            top_ngrams = []
            for text in df['text']:
                #removing the stopwords
                words = " ".join([word for word in text.split(" ") if (word.lower() not in set(stopwords.words('english'))) and (word.isalpha())])
                #tokenizing the words
                token = word_tokenize(words)
                #taking the ngrams by specifying n value given by the user
                n_grams = ngrams(token, n)
                #getting the most common 10 words
                for x in Counter(n_grams).most_common(top):
                    top_ngrams.append(x)
            return top_ngrams

        try:
            #top nGrams
            top_unigrams = pd.DataFrame(top_ngrams(test, n, top_n))
            #displaying the top n nGrams
            fig1 = plt.figure(figsize = (10, 5))
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            sns.barplot(data = top_unigrams, x = 0, y = 1)
            plt.title("Top "+ str(top_n) + " nGrams")
            plt.xlabel("nGrams")
            plt.ylabel("Number of occurences")
            plt.xticks(rotation = 45)
            st.pyplot(fig1)
        except:
            st.write("Please Enter the text to view nGrams")

    try:
        st.subheader("Sentiment Analysis: ")
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        nltk.download('vader_lexicon')

        sid = SentimentIntensityAnalyzer()
        def get_vader_score(sent):
        # Polarity score returns dictionary
            ss = sid.polarity_scores(sent)
            #excluding the compound
            dict_sentiment = {k:v for k,v in ss.items() if k != "compound"}
            return dict_sentiment
        senti = get_vader_score(test["text"][0])
        hist_df = pd.DataFrame([{"Sentiment": sentiment, "Probability": probability} for (sentiment, probability) in senti.items()])

        fig3 = plt.figure(figsize = (10, 5))
        plot = sns.barplot(data = hist_df, y = "Sentiment", x = "Probability")
        plot.set_yticklabels(["Negative", "Neutral", "Positive"])
        plt.title("Sentiment Analysis")
        plt.xlabel("Probability")
        plt.ylabel("Sentiment")
        st.pyplot(fig3)
    except NameError:
        st.write("Please Enter the text to view the sentiment analysis")

    try:
        #with st.spinner(text='In Progress'):
        #    time.sleep(1)
        #    st.success('Done')
        if test_predictions[0] == 1:
            st.markdown("## This news seems to be: NOT RELIABLE 🥲")
        elif test_predictions[0] == 0:
            st.markdown("## This news seems to be: RELIABLE 😍")
    except:
        st.write("Please Enter the text to view reliability")
