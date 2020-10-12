import networkx as nx
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

'''
Read the reviews to create two variables that can be used to create the graph and most occuring word dictionary
Graph contains all the words, digits and characters
Most frequently occuring words requires only words

'''
reviews = open("./opinosis-summarization/sample_data/toyota_camry.txt","r")
allText = reviews.read()

# Remove next line character and breaklines fro reviews
combinedSentences = allText.lower().strip().replace('\n', ' ').replace('\r', '')
# Remove all uncessary characters and digits
combinedSentences = re.sub("[^a-zA-Z]", " ", combinedSentences)
combinedSentences = combinedSentences.strip()

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
reviews = open("./opinosis-summarization/sample_data/toyota_camry.txt","r")
allText = reviews.read()
allSentence = allText.lower().split('\n')

# Create directed graph
G = nx.DiGraph()
G.add_nodes_from(uniqueWords)

# Add weights to each edge, based on the occurance of relationship between words (i -> i+1)
weight_dict = {}
for sentence in allSentence:
    allWordsinSentence = sentence.lower().strip().replace('\n', ' ').replace('\r', '').split(' ')
    for i in range(len(allWordsinSentence)-1):
        tup = (allWordsinSentence[i], allWordsinSentence[i+1])
        G.add_edge(allWordsinSentence[i], allWordsinSentence[i+1])
        if tup in list(weight_dict.keys()):
            weight_dict[tup] = weight_dict[tup] + 1
        else:
            weight_dict[tup] = 1

reviews.close()
keys = list(weight_dict.keys())

# Set the weights to edges between words
for key in keys:
    G[key[0]][key[1]]['weight'] = int(weight_dict[key])


'''

Use TF-IDF to find the most important set of words in the reviews

'''
reviews = open("./opinosis-summarization/sample_data/toyota_camry.txt","r")
eachReview = reviews.readlines()

# Clean reviews before running Tf-Idf
removeNewLine = lambda x: x.lower().strip().replace('\n', ' ').replace('\r', '').replace('.','').replace(' .', '').replace('!', '').replace(',', '').replace('?', '').strip()
eachReview = list(map(removeNewLine, eachReview))

# Do not remove any accent from the words, otherwise the words won't match the graph
vectorizer = TfidfVectorizer(stop_words = 'english', sublinear_tf = True)
tfidfVector = vectorizer.fit_transform(eachReview)

tfidf_score = list(tfidfVector[0].toarray()[0])

words = vectorizer.get_feature_names()

feature_array = np.array(words)
tfidf_sorting = np.argsort(tfidfVector.toarray()).flatten()[::-1]

# Get the top n words based on descending order of tfidf score
n = 10
top_n = feature_array[tfidf_sorting][:n]

print(top_n)

'''

Create summary reviews using graph and TF-IDF
Select the starting word (must be in vocabulary) and the length of reviews
The review either is of selected length or until it encounters a full stop (.)

'''
initialNode = top_n[9]
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

