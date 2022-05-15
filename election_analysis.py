#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple NLP analysis of USA presidential elect speeches

"""


import pandas as pd
import nltk
import seaborn as sns
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from PIL import Image
import numpy as np

gsfont = {'fontname':'Gill Sans'}

#data vis plot type = population pyramid

def make_pd_data(speech1, speech2, index=['df1', 'df2']): 
    """takes 2 speech class objects and converts some values into a pandas DataFrame"""
    
    df1 = extract_values_to_pd(speech1)
    df2 = extract_values_to_pd(speech2)
    
    df = pd.concat([df1, df2], axis=0)
    df['Names'] = index
    df = df.set_index('Names')
    
    return(df)


def extract_values_to_pd(speech): 
    """extracts values from a speech object into a single dataframe.  Workings are hardcoded"""
    
    mean_wordlen_excl = np.mean(list(map(len, speech.token_word_clean)))
    
    data = {"Absolute Token Count":speech.word_count, 
            "Word Count":len(speech.token_word_clean),
            "Absolute Sentence Count":speech.sent_count, 
            "Average Overall Word Length":speech.mean_word_len,
            "Average Word Length":mean_wordlen_excl,
            "Average Overall Sentence Length":speech.mean_sent_len,
            "Overall Sentiment Value":speech.global_sentiment, 
            "Mean Sentence Sentiment Value":speech.mean_sent_sentiment}
    
    df = pd.DataFrame(data=data, index=[1])
    return(df)



class speech(): 
    
    def __init__(self, fpath): 
        self.word_count = None
        self.sent_count = None
        self.mean_word_len = None
        self.mean_sent_len = None
        self.mean_sent_sentiment = None
        self.mean_word_sentiment = None
        self.global_sentiment = None
        self.sentence_sentiment = None
        self.filepath = fpath
        self.speech_block = None
        self.token_sent = None
        self.token_word = None
        self.token_word_clean = None
        self.stopwords = list(set(stopwords.words('english')))
        self.fdist = None
        self.valency = None
        self.wc = None
        
        self.load_speech()
        self.tokenize_sents()
        self.tokenize_words()
        self.calculate_descriptors()
        self.calc_mean_sentiment_sent()
        self.calc_global_sentiment()
        self.calc_freq()
        self.calc_valency()
        
        
    def load_speech(self): 
        """loads and returns a clean block of text"""
        try: 
            with open(self.filepath) as f:
                txt = f.read()
        
        except UnicodeDecodeError: 
            with open(self.filepath, encoding="utf8") as f:
                txt = f.read()
                    
        self.speech_block = txt.replace("\n", " ")
        
        
    def tokenize_sents(self): 
        """tokenizes block speech by sentences"""
        self.token_sent = nltk.tokenize.sent_tokenize(self.speech_block)
        
        
    def tokenize_words(self): 
        """tokenize block of text into word chunks"""
        self.token_word = nltk.tokenize.word_tokenize(self.speech_block)
        
    
    def calc_mean_sentiment_sent(self): 
        """analyze sentiments of class objects, both words and sentences"""
        analyzer = SentimentIntensityAnalyzer()
        compound_sent_values = [analyzer.polarity_scores(x)['compound'] for x in self.token_sent]
        self.mean_sent_sentiment = np.mean(compound_sent_values)
        
    
    def calc_global_sentiment(self): 
        """calculate the overall compound sentiment using VADER"""
        analyzer = SentimentIntensityAnalyzer()
        self.global_sentiment = analyzer.polarity_scores(self.speech_block)['compound']
    
    
    def calc_freq(self): 
        """calculate Frequency distribution of text without stop words"""
        self.fdist = FreqDist(self.token_word_clean)
        
        
    def calc_valency(self): 
        """outputs a pandas data frame of word, count, and compound valence per VADER"""
        self.calc_freq()
        wordlist=[]
        countlist=[]
        valencelist=[]
        
        analyzer = SentimentIntensityAnalyzer()
        
        for item in self.fdist.most_common(): 
            wordlist.append(item[0])
            countlist.append(item[1])
            valencelist.append(analyzer.polarity_scores(item[0])['compound'])
        
        self.valency = pd.DataFrame(data={'word':wordlist, 'count':countlist, 'valence':valencelist})
    
    
    def calculate_descriptors(self): 
        """calculates mean word length and mean sentence length"""
        self.mean_sent_len = np.mean(list(map(len, self.token_sent)))
        self.mean_word_len = np.mean(list(map(len, self.token_word)))
        self.word_count = len(self.token_word)
        self.sent_count = len(self.token_sent)
    
        self.token_word_clean = [x.lower() for x in self.token_word if x.isalnum() and x.lower() not in self.stopwords]
    
    
    def print_values(self): 
        """Prints out the basic descriptors of the speech"""
        print ("Token count = %d" % self.word_count)
        print ("Count of words excluding stopwords = %d" % len(self.token_word_clean))
        print ("Sentence count = %d" % self.sent_count)
        print ("Mean word length = %.2f" % self.mean_word_len)
        print ("Mean length of words without stopwords = %.2f" % np.mean(list(map(len, self.token_word_clean))))
        print ("Mean sentence length = %.2f" % self.mean_sent_len)
        print ("Mean Sentence Sentiment = %.2f" % self.mean_sent_sentiment)
        print ("Global sentiment value = %.2f" % self.global_sentiment)
        
    def create_wc(self, title="", save=None): 
        usa_mask = np.array(Image.open("usa_mask.jpeg"))
        self.wc = WordCloud(background_color='white',
                            scale=5,
                            mask=usa_mask).fit_words(self.fdist)
        plt.figure(figsize=(8, 6), dpi=200)
        plt.imshow(self.wc, interpolation='bilinear')
        plt.title(title, fontsize=13, **gsfont)
        plt.axis("off")
        
        if save!=None: 
            if type(save)==str: 
                plt.savefig(save)
            else: 
                print("Error in saving plot.  Could not resolve save arg")
        


###main###
biden_fpath = "Biden.txt"
harris_fpath = "Harris.txt"
trump_fpath = "Trump.txt"
obama_fpath = "obama2012.txt"






