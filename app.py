from typing import List
import pandas as pd
import preprocessor as p
import nltk
from nltk.tokenize import TweetTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from afinn import Afinn
import re
from nltk import sent_tokenize
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from itertools import groupby
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import string
nltk.download('averaged_perceptron_tagger')
from collections import Counter
import time


def prediction(ds,feature):
    '''
    This function uses tfidf vector to predict the ticket-type.
    And returns the prediction as well asprediction probabilty too.
    '''

    # cyberbullying classification
    if ds == 1:
        # svm_tfidf
        svm_tfidf_1= pickle.load(open('ds1_svm_comb.pkl', 'rb'))
        prediction  = svm_tfidf_1.predict(feature)
    elif ds==2:
        svm_tfidf_2 = pickle.load(open('ds2_svm_comb.pkl', 'rb'))
        prediction = svm_tfidf_2.predict(feature)

    with st.spinner(text='Predicting...'):
        time.sleep(3)

    # returning prediction with its probability
    return prediction


def preprocess_feature_text(ds,text):
    '''
    This function allows to convert the text data into tf-idf vector and then returns it.
    '''
    with st.spinner(text='Analyzing tweet...'):
        time.sleep(3)

    # Replacing url,emoji,hashtags,mentions,smileys,numbers by the correspondng terms
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.MENTION, p.OPT.SMILEY, p.OPT.NUMBER)
    pp_data = p.tokenize(text)

    # Lowercase
    lower_text = pp_data.lower()

    #Lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(t):
            return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((t))]

    #Tokenization
    w_tokenizer = TweetTokenizer()

    #Removing punctuations
    def remove_punctuation(words):
        new_words= []
        for word in words:
                new_word = re.sub(r'[^\w\s]', '', (word))
                if new_word != '':
                    new_words.append(new_word)
        return new_words


    words = lemmatize_text(lower_text)
    words = remove_punctuation(words)

    #Stopword removal
    stop_words = set(stopwords.words('english'))
    clean_text = ' '.join(word.strip() for word in words if word not in stop_words)

    # checker
    if clean_text == '':
        st.error(
            'No content after preprocessing')
        st.stop()

    # List of profane words
    profane_words = ['4r5e', '5h1t', '5hit', 'a55', 'anal', 'anus', 'ar5e', 'arrse', 'arse', 'ass', 'ass-fucker',
                     'asses', 'assfucker',
                     'assfukka', 'asshole', 'assholes', 'asswhole', 'a_s_s', 'b!tch', 'b00bs', 'b17ch', 'b1tch',
                     'ballbag', 'balls',
                     'ballsack', 'bastard', 'beastial', 'beastiality', 'bellend', 'bestial', 'bestiality', 'bi+ch',
                     'biatch', 'bitch',
                     'bitcher', 'bitchers', 'bitches', 'bitchin', 'bitching', 'bloody', 'blow job', 'blowjob',
                     'blowjobs', 'boiolas',
                     'bollock', 'bollok', 'boner', 'boob', 'boobs', 'booobs', 'boooobs', 'booooobs', 'booooooobs',
                     'breasts', 'buceta',
                     'bugger', 'bum', 'bunny fucker', 'butt', 'butthole', 'buttmunch', 'buttplug', 'c0ck', 'c0cksucker',
                     'carpet muncher',
                     'cawk', 'chink', 'cipa', 'cl1t', 'clit', 'clitoris', 'clits', 'cnut', 'cock', 'cock-sucker',
                     'cockface', 'cockhead',
                     'cockmunch', 'cockmuncher', 'cocks', 'cocksuck ', 'cocksucked ', 'cocksucker', 'cocksucking',
                     'cocksucks ', 'cocksuka',
                     'cocksukka', 'cok', 'cokmuncher', 'coksucka', 'coon', 'cox', 'crap', 'cum', 'cummer', 'cumming',
                     'cums', 'cumshot',
                     'cunilingus', 'cunillingus', 'cunnilingus', 'cunt', 'cuntlick ', 'cuntlicker ', 'cuntlicking ',
                     'cunts', 'cyalis',
                     'cyberfuc', 'cyberfuck ', 'cyberfucked ', 'cyberfucker', 'cyberfuckers', 'cyberfucking ', 'd1ck',
                     'damn', 'dick',
                     'dickhead', 'dildo', 'dildos', 'dink', 'dinks', 'dirsa', 'dlck', 'dog-fucker', 'doggin', 'dogging',
                     'donkeyribber',
                     'doosh', 'duche', 'dyke', 'ejaculate', 'ejaculated', 'ejaculates ', 'ejaculating ', 'ejaculatings',
                     'ejaculation',
                     'ejakulate', 'f u c k', 'f u c k e r', 'f4nny', 'fag', 'fagging', 'faggitt', 'faggot', 'faggs',
                     'fagot', 'fagots',
                     'fags', 'fanny', 'fannyflaps', 'fannyfucker', 'fanyy', 'fatass', 'fcuk', 'fcuker', 'fcuking',
                     'feck', 'fecker',
                     'felching', 'fellate', 'fellatio', 'fingerfuck ', 'fingerfucked ', 'fingerfucker ',
                     'fingerfuckers', 'fingerfucking '
        , 'fingerfucks ', 'fistfuck', 'fistfucked ', 'fistfucker ', 'fistfuckers ', 'fistfucking ', 'fistfuckings ',
                     'fistfucks ', 'flange', 'fook', 'fooker', 'fuck', 'fucka', 'fucked', 'fucker', 'fuckers',
                     'fuckhead', 'fuckheads',
                     'fuckin', 'fucking', 'fuckings', 'fuckingshitmotherfucker', 'fuckme ', 'fucks', 'fuckwhit',
                     'fuckwit',
                     'fudge packer', 'fudgepacker', 'fuk', 'fuker', 'fukker', 'fukkin', 'fuks', 'fukwhit', 'fukwit',
                     'fux', 'fux0r',
                     'f_u_c_k', 'gangbang', 'gangbanged ', 'gangbangs ', 'gaylord', 'gaysex', 'goatse', 'God',
                     'god-dam', 'god-damned',
                     'goddamn', 'goddamned', 'hardcoresex ', 'hell', 'heshe', 'hoar', 'hoare', 'hoer', 'homo', 'hore',
                     'horniest', 'horny',
                     'hotsex', 'jack-off ', 'jackoff', 'jap', 'jerk-off ', 'jism', 'jiz ', 'jizm ', 'jizz', 'kawk',
                     'knob', 'knobead',
                     'knobed', 'knobend', 'knobhead', 'knobjocky', 'knobjokey', 'kock', 'kondum', 'kondums', 'kum',
                     'kummer', 'kumming',
                     'kums', 'kunilingus', 'l3i+ch', 'l3itch', 'labia', 'lmfao', 'lust', 'lusting', 'm0f0', 'm0fo',
                     'm45terbate',
                     'ma5terb8', 'ma5terbate', 'masochist', 'master-bate', 'masterb8', 'masterbat*', 'masterbat3',
                     'masterbate',
                     'masterbation', 'masterbations', 'masturbate', 'mo-fo', 'mof0', 'mofo', 'mothafuck', 'mothafucka',
                     'mothafuckas',
                     'mothafuckaz', 'mothafucked ', 'mothafucker', 'mothafuckers', 'mothafuckin', 'mothafucking ',
                     'mothafuckings',
                     'mothafucks', 'mother fucker', 'motherfuck', 'motherfucked', 'motherfucker', 'motherfuckers',
                     'motherfuckin',
                     'motherfucking', 'motherfuckings', 'motherfuckka', 'motherfucks', 'muff', 'mutha', 'muthafecker',
                     'muthafuckker',
                     'muther', 'mutherfucker', 'n1gga', 'n1gger', 'nazi', 'nigg3r', 'nigg4h', 'nigga', 'niggah',
                     'niggas', 'niggaz',
                     'nigger', 'niggers ', 'nob', 'nob jokey', 'nobhead', 'nobjocky', 'nobjokey', 'numbnuts', 'nutsack',
                     'orgasim ',
                     'orgasims ', 'orgasm', 'orgasms ', 'p0rn', 'pawn', 'pecker', 'penis', 'penisfucker', 'phonesex',
                     'phuck', 'phuk',
                     'phuked', 'phuking', 'phukked', 'phukking', 'phuks', 'phuq', 'pigfucker', 'pimpis', 'piss',
                     'pissed', 'pisser',
                     'pissers', 'pisses ', 'pissflaps', 'pissin ', 'pissing', 'pissoff ', 'poop', 'porn', 'porno',
                     'pornography',
                     'pornos', 'prick', 'pricks ', 'pron', 'pube', 'pusse', 'pussi', 'pussies', 'pussy', 'pussys ',
                     'rectum', 'retard',
                     'rimjaw', 'rimming', 's hit', 's.o.b.', 'sadist', 'schlong', 'screwing', 'scroat', 'scrote',
                     'scrotum', 'semen',
                     'sex', 'sh!+', 'sh!t', 'sh1t', 'shag', 'shagger', 'shaggin', 'shagging', 'shemale', 'shi+', 'shit',
                     'shitdick', 'shite',
                     'shited', 'shitey', 'shitfuck', 'shitfull', 'shithead', 'shiting', 'shitings', 'shits', 'shitted',
                     'shitter',
                     'shitters ', 'shitting', 'shittings', 'shitty ', 'skank', 'slut', 'sluts', 'smegma', 'smut',
                     'snatch', 'son-of-a-bitch',
                     'spac', 'spunk', 's_h_i_t', 't1tt1e5', 't1tties', 'teets', 'teez', 'testical', 'testicle', 'tit',
                     'titfuck', 'tits',
                     'titt', 'tittie5', 'tittiefucker', 'titties', 'tittyfuck', 'tittywank', 'titwank', 'tosser',
                     'turd', 'tw4t', 'twat',
                     'twathead', 'twatty', 'twunt', 'twunter', 'v14gra', 'v1gra', 'vagina', 'viagra', 'vulva', 'w00se',
                     'wang', 'wank',
                     'wanker', 'wanky', 'whoar', 'whore', 'willies', 'willy', 'xrated', 'xxx'
                     ]

    # list of pronouns
    pronouns = ["she", "shes", "she's", "her", "hers", "her's", "he", "hes", "he's", "him", "his", "i", "you", "u",
                "it", "ur", "we",
                "they", "mine", "yours", "your", "your's", "theirs", "their's", "their", "its", "it's", "ours", "our",
                "ones", "one's"]

    # Function to get word occurence
    def word_count_pronoun(text):

        # split the sentence in words
        li = text.split()

        # for each word check if present in list
        sent_counter = 0
        dw = []
        for i in li:
            # if present in list count occurence
            if i in pronouns and i not in dw:
                count_word = li.count(i)
                sent_counter = sent_counter + count_word
                dw.append(i)

        return sent_counter

    # Function to get word occurence
    def word_count_profane(text):

        # split the sentence in words
        li = text.split()

        # for each word check if present in list
        sent_counter = 0
        dw = []
        for i in li:
            # if present in list count occurence
            if i in profane_words and i not in dw:
                count_word = li.count(i)
                sent_counter = sent_counter + count_word
                dw.append(i)

        return sent_counter

    # Functions to find Lexical richness by Yule I method
    # https://swizec.com/blog/measuring-vocabulary-richness-with-python/
    # Larger the value - richer is the vocabulary
    def words(entry):
        return filter(lambda w: len(w) > 0,
                      [w.strip("0123456789!:,.?(){}[]") for w in entry.split()])

    def yule(entry):
        # yule's I measure (the inverse of yule's K measure)
        # higher number is higher diversity - richer vocabulary
        d = {}
        stemmer = PorterStemmer()
        for w in words(entry):
            w = stemmer.stem(w).lower()
            try:
                d[w] += 1
            except KeyError:
                d[w] = 1

        M1 = float(len(d))
        M2 = sum([len(list(g)) * (freq ** 2) for freq, g in groupby(sorted(d.values()))])

        try:
            return (M1 * M1) / (M2 - M1)
        except ZeroDivisionError:
            return 0

    #Punctuations count
    def get_punct(x):
        count = 0
        count_occ = 0
        for i in string.punctuation:
            if i in x:
                count_occ = x.count(i)
                count = count + count_occ
        return count

    #text features
    tweet_length = len(text)
    i = sent_tokenize(text)
    no_of_sent_ = len(i)
    lex_r = yule(text)
    t = text.lower()
    pw = word_count_profane(t)
    pro = word_count_pronoun(t)
    af = Afinn()
    sentiment_scores = af.score(text)
    if sentiment_scores > 0:
        sentiment_category = '1'
    elif sentiment_scores < 0:
        sentiment_category = '-1'
    else:
        sentiment_category = '0'
    if ds == 1:
        txt_features = pd.DataFrame([[tweet_length, no_of_sent_, lex_r, pw, pro, sentiment_category]],columns=['tweet_length', 'no_of_sent_', 'lex_r', 'pw', 'pro', 'sentiment_category'])
    elif ds==2:

        tx = clean_text
        hashtag = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('hashtag'), tx))
        url = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('url'), tx))
        mention = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('mention'), tx))
        number = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('number'), tx))
        smilem = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('emoji'), tx)) + sum( 1 for _ in re.finditer(r'\b%s\b' % re.escape('smiley'), tx))
        punct = get_punct(text)
        tokens = nltk.word_tokenize(text.lower())
        tx = nltk.Text(tokens)
        tags = nltk.pos_tag(tx)
        counts = Counter(tag for word, tag in tags)
        noun = counts['NN'] + counts['NNS'] + counts['NNP'] + counts['NNPS']
        verb = counts['VBG'] + counts['VB'] + counts['VBN'] + counts['VBP'] + counts['VBZ']
        adjective = counts['JJ'] + counts['JJR'] + counts['JJS']
        adverb = counts['RB'] + counts['RBR'] + counts['RBS'] + counts['WRB']

        txt_features = pd.DataFrame([[tweet_length, no_of_sent_, lex_r, pw, pro, sentiment_category, hashtag, url, smilem, mention, number, punct, noun, verb, adjective, adverb]],
                                    columns=['tweet_length', 'no_of_sent_', 'lex_r', 'pw', 'pro', 'sentiment_category',
                                             'hashtag', 'url', 'smiley', 'mention', 'number', 'punct', 'noun', 'verb',
                                             'adjective', 'adverb'])

    #tfidf transforming text
    transformer = TfidfTransformer()
    if ds == 1:
        #loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("ds1_tfidf_vocab_comb.pkl", "rb")))
        with open('ds1_tfidf_vocab_comb.pkl', 'rb') as ifp1:
            loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(ifp1))
    elif ds == 2:
        #loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("ds2_tfidf_vocab_comb.pkl", "rb")))
        with open('ds2_tfidf_vocab_comb.pkl', 'rb') as ifp2:
            loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(ifp2))
    tfidf_feature = transformer.fit_transform(loaded_vec.fit_transform(np.array([clean_text])))
    tf_f = pd.DataFrame(tfidf_feature.toarray())

    comb_f = pd.concat([tf_f, txt_features], axis=1)

    return comb_f


def main():
    # app title
    #st.header('Twitter Cyberbullying Classification')

    html_temp = '''
    <div style="background-color:tomato; padding:20px; border-radius: 25px;">
    <h2 style="color:white; text-align:center; font-size: 30px;"><b>Twitter Cyberbullying Classification</b></h2>
    </div><br><br>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)
    # input
    text = st.text_area('Please enter tweet:')

    # predicting text type
    if st.button('Predict'):

        # necessary requirements

        # no empty text
        if text.strip() == '':
            st.warning('No tweet is written! Kindly enter a tweet.')
            st.stop()


        #preprocessing of text
        features_1 = preprocess_feature_text(1,text)

        # predicting ticket-type
        pred1 = prediction(1,features_1)
        print(pred1)

        # result display
        if pred1 == 1:
            value = 'Cyberbullying'
            result = 'The classification of the tweet is: ' + value
        else:
            features_2 = preprocess_feature_text(2, text)
            pred2 = prediction(2, features_2)
            if pred2 == 2:
                value = 'None'
            elif pred2==3:
                value = 'Spam'
            elif pred2 == 4:
                value ='Sarcasm'
            elif pred2 == 5:
                value ='Profanity'
            result = 'The Non-cyberbullying text is classified as: ' + value
        print(value)
        st.success(result + '\n')


if __name__ == '__main__':
    main()
