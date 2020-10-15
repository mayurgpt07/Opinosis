import networkx as nx
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

## Creating function to get the average starting position of word
def average(avgList):
    return len(avgList), sum(avgList)/len(avgList)

def getAveragePosition(uniqueWords, allSentence):
    startingWordList = uniqueWords
    eachWordAvg = []
    wordAveragePosition = {}
    wordNumber = {}
    for eachWord in startingWordList:
        for eachSentence in allSentence:
            index = eachSentence.lower().strip().replace('\n', ' ').replace('\r', '').find(eachWord)
            if index != -1:
                eachWordAvg.append(index + 1)
        wordNumber[eachWord], wordAveragePosition[eachWord] = average(eachWordAvg)
        eachWordAvg.clear()
    return wordNumber, wordAveragePosition


fileName = "./opinosis-summarization/sample_data/toyota_camry.txt"

## Get the main points using nouns
## Filter starting position list using pos
## Try to derive paths that must go through a word (and then find the most weighted amongst it)
## Use opionosis method as well
## Linking Verbs (is, are etc) are hubs - Collapsible node
## Cordinating Conjunction, punctuations () are ends


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

'''
Read the reviews to create two variables that can be used to create the graph and most occuring word dictionary
Graph contains all the words, digits and characters
Most frequently occuring words requires only words
'''

reviews = open(fileName,"r")
allText = reviews.read()

# Remove next line character and breaklines fro reviews
combinedSentences = allText.lower().strip().replace('\n', ' ').replace('\r', '')

# Remove all uncessary characters and digits
combinedSentences = re.sub("[^a-zA-Z]", " ", combinedSentences)
combinedSentences = combinedSentences.strip()
# print(combinedSentences)
# Create another copy of initial review set for the graph
allText = allText.lower().strip().replace('\n', ' ').replace('\r', '')
allText = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in allText.split(" ")]) 
# print(allText)

uniqueWords = list(set(allText.split(' ')))
reviews.close()

# Create new dictionary to see the maximum occuring word 
dict_word = {}
stop_words = set(stopwords.words('english'))

for word in combinedSentences.split(' '):
    if word not in stop_words:
        if word in list(dict_word.keys()):
            dict_word[word] = dict_word[word] + 1
        else:
            dict_word[word] = 1

# Dictionary containing most occuring word sorted in ascending order
sorted_word_dictionary = {k: v for k, v in sorted(dict_word.items(), key=lambda item: item[1])}

'''
Create a directed graph for the vocabulary in reviews
Directed graphs allow traversal that supports the language used to write the review

'''
reviews = open(fileName,"r")
allText = reviews.read()
allSentence = allText.lower().split('\n')

resolveShortForm = lambda x: ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in x.split(" ")])
allSentence = list(map(resolveShortForm, allSentence))
wordNumber, wordAveragePosition = getAveragePosition(uniqueWords, allSentence)


wordAveragePosition = {k: v for k, v in sorted(wordAveragePosition.items(), key=lambda item: item[1])}
print(wordAveragePosition)

# Create directed graph
G = nx.DiGraph()

## Add pos tagged nodes as the root nodes
G.add_nodes_from(uniqueWords)

# Add weights to each edge, based on the occurance of relationship between words (i -> i+1)
pos_tag_sentence = []
weight_dict = {}
for sentence in allSentence:
    pos_tag_sentence.append(pos_tag(word_tokenize(sentence.lower().strip().replace('\n', ' ').replace('\r', ''))))
    allWordsinSentence = sentence.lower().strip().replace('\n', ' ').replace('\r', '').split(' ')
    for i in range(len(allWordsinSentence)-1):
        tup = (allWordsinSentence[i], allWordsinSentence[i+1])
        G.add_edge(allWordsinSentence[i], allWordsinSentence[i+1])
        if tup in list(weight_dict.keys()):
            weight_dict[tup] = weight_dict[tup] + 1
        else:
            weight_dict[tup] = 1

reviews.close()

# print(pos_tag_sentence[0])
keys = list(weight_dict.keys())

# Set the weights to edges between words
for key in keys:
    G[key[0]][key[1]]['weight'] = int(weight_dict[key])


'''
Use TF-IDF to find the most important set of words in the reviews

'''
reviews = open(fileName,"r")
eachReview = reviews.readlines()

# Clean reviews before running Tf-Idf
removeNewLine = lambda x: x.lower().strip().replace('\n', ' ').replace('\r', '').replace('.','').replace(' .', '').replace('!', '').replace(',', '').replace('?', '').strip()
eachReview = list(map(removeNewLine, eachReview))

# Do not remove any accent from the words, otherwise the words won't match the graph
vectorizer = TfidfVectorizer(stop_words = 'english', sublinear_tf = False)
tfidfVector = vectorizer.fit_transform(eachReview)

tfidf = tfidfVector.todense()
tfidf[tfidf == 0] = np.nan
means = np.nanmean(tfidf, axis=0)
means = dict(zip(vectorizer.get_feature_names(), means.tolist()[0]))


sorted_means = {k: v for k, v in sorted(means.items(), key=lambda item: item[1])}
# print(sorted_means)
# tfidf = tfidfVector.todense()
# ordered = np.argsort(tfidf*-1)
# words = vectorizer.get_feature_names()


# global_average_score = {}
# top_k = 5
# for i, doc in enumerate(eachReview):
#     result = {}
#     for t in range(top_k):
#         result[words[ordered[i,t]]] = means[words[ordered[i,t]]]


'''
Create summary reviews using graph and TF-IDF
Select the starting word (must be in vocabulary) and the length of reviews
The review either is of selected length or until it encounters a full stop (.)

'''

initialNode = 'appears'#top_n[9]
# print(wordNumber[initialNode])
maxWeight = -1
node = ''
finalString = initialNode
lengthOfReviews = 10

# Create summary review using max weight graph traversal (most common words coming after each other) 
while lengthOfReviews > 0 and (node != '.' and node != '!'):
    for neighbor in G.neighbors(initialNode):
        weight = G[initialNode][neighbor]['weight']
        if weight > maxWeight:
            maxWeight = weight
            node = neighbor
    finalString = finalString + ' ' + node
    maxWeight = -1
    initialNode = node
    lengthOfReviews = lengthOfReviews - 1

print('Final Summary: ', finalString)