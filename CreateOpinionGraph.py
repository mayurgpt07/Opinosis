import networkx as nx
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords

reviews = open("./opinosis-summarization/sample_data/kindle.txt","r")
allText = reviews.read()
combinedSentences = allText.lower().strip().replace('\n', ' ').replace('\r', '')
combinedSentences = re.sub("[^a-zA-Z]", " ", combinedSentences)
combinedSentences = combinedSentences.strip()
allText = allText.lower().strip().replace('\n', ' ').replace('\r', '')
uniqueWords = list(set(allText.split(' ')))
reviews.close()

dict_word = {}
stop_words = set(stopwords.words('english'))
# print(stop_words)
for word in combinedSentences.split(' '):
    if word not in stop_words:
        if word in list(dict_word.keys()):
            dict_word[word] = dict_word[word] + 1
        else:
            dict_word[word] = 1

sorted_word_dictionary = {k: v for k, v in sorted(dict_word.items(), key=lambda item: item[1])}

reviews = open("./opinosis-summarization/sample_data/kindle.txt","r")
allText = reviews.read()
allSentence = allText.lower().split('\n')

G = nx.DiGraph()
G.add_nodes_from(uniqueWords)
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

print(nx.info(G))
# nx.draw(G, node_size = 8, font_size = 10, labels = dict_word)
# plt.show()
reviews.close()
keys = list(weight_dict.keys())

for key in keys:
    G[key[0]][key[1]]['weight'] = int(weight_dict[key])

# print(G['being']['able']['weight'])
# print(sorted_word_dictionary)

def aggregate_weights(G, path):
    return sum([G[i][i + 1]['weight'] for i in range(len(path) - 2)])

print(nx.dijkstra_path(G, 'being', 'font'))

initialNode = 'different'
maxWeight = -1
node = ''
finalString = initialNode
lengthOfReviews = 10

while lengthOfReviews > 0 and node != '.':
    print(node)
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
