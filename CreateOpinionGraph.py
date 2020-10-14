import networkx as nx
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

fileName = "./opinosis-summarization/sample_data/kindle.txt" 

## Get the main points using nouns
## Filter starting position list using pos
## Try to derive paths that must go through a word (and then find the most weighted amongst it)
## Use opionosis method as well

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

## Creating function to get the average starting position of word
def average(avgList):
    return sum(avgList)/len(avgList)

def getAveragePosition(sorted_word_dictionary, allSentence):
    startingWordList = list(sorted_word_dictionary.keys())
    eachWordAvg = []
    wordAveragePosition = {}
    for eachWord in startingWordList:
        for eachSentence in allSentence:
            index = eachSentence.lower().strip().replace('\n', ' ').replace('\r', '').find(eachWord)
            if index != -1:
                eachWordAvg.append(index + 1)
        wordAveragePosition[eachWord] = average(eachWordAvg)
        eachWordAvg.clear()
    return wordAveragePosition
      
wordAveragePosition = getAveragePosition(sorted_word_dictionary, allSentence)
wordAveragePosition = {k: v for k, v in sorted(wordAveragePosition.items(), key=lambda item: item[1])}
# print(wordAveragePosition)
# Create directed graph
G = nx.DiGraph()
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
vectorizer = TfidfVectorizer(stop_words = 'english', sublinear_tf = True)
tfidfVector = vectorizer.fit_transform(eachReview)

tfidf = tfidfVector.todense()
tfidf[tfidf == 0] = np.nan
means = np.nanmean(tfidf, axis=0)
means = dict(zip(vectorizer.get_feature_names(), means.tolist()[0]))

tfidf = tfidfVector.todense()
ordered = np.argsort(tfidf*-1)
words = vectorizer.get_feature_names()

global_average_score = {}
top_k = 200
for i, doc in enumerate(eachReview):
    result = { }
    for t in range(top_k):
        result[words[ordered[i,t]]] = means[words[ordered[i,t]]]
        

'''
Create summary reviews using graph and TF-IDF
Select the starting word (must be in vocabulary) and the length of reviews
The review either is of selected length or until it encounters a full stop (.)

'''

initialNode = 'orientation'#top_n[9]
maxWeight = -1
node = ''
finalString = initialNode
lengthOfReviews = 10

# Create summary review using max weight graph traversal (most common words coming after each other) 
while lengthOfReviews > 0 and node != '.':
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