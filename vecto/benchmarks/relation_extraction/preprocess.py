
from __future__ import print_function
import numpy as np
import gzip
import os
import sys
import pickle as pkl





#Mapping of the labels to integers
labelsMapping = {'Other':0,
                 'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2,
                 'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4,
                 'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6,
                 'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
                 'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
                 'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,
                 'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
                 'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
                 'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}




words = {}
maxSentenceLen = [0,0]


distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
minDistance = -30
maxDistance = 30
for dis in range(minDistance,maxDistance+1):
    distanceMapping[dis] = len(distanceMapping)
print(distanceMapping)


def getWordIdx(token, word2Idx):
    """Returns from the word2Idex table the word index for a given token"""
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    return 0

def createTensor(file, word2Idx, maxSentenceLen=100):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
    positionMatrix1 = []
    positionMatrix2 = []
    tokenMatrix = []

    for line in open(file):
        splits = line.strip().split('\t')

        label = splits[0]
        pos1 = splits[1]
        pos2 = splits[2]
        sentence = splits[3]
        tokens = sentence.split(" ")

        #print(label, pos1, pos2, sentence, tokens)


        tokenIds = np.zeros(maxSentenceLen)
        positionValues1 = np.zeros(maxSentenceLen)
        positionValues2 = np.zeros(maxSentenceLen)

        for idx in range(0, min(maxSentenceLen, len(tokens))):
            tokenIds[idx] = getWordIdx(tokens[idx], word2Idx)

            distance1 = idx - int(pos1)
            distance2 = idx - int(pos2)
            #print(distance1, distance2)
            if distance1 in distanceMapping:
                #print('helo')
                positionValues1[idx] = distanceMapping[distance1]
            elif distance1 <= minDistance:
                positionValues1[idx] = distanceMapping['LowerMin']
            else:
                positionValues1[idx] = distanceMapping['GreaterMax']

            if distance2 in distanceMapping:
                positionValues2[idx] = distanceMapping[distance2]
            elif distance2 <= minDistance:
                positionValues2[idx] = distanceMapping['LowerMin']
            else:
                positionValues2[idx] = distanceMapping['GreaterMax']

        tokenMatrix.append(tokenIds)
        positionMatrix1.append(positionValues1)
        positionMatrix2.append(positionValues2)

        labels.append(labelsMapping[label])



    return np.array(labels, dtype='int32'), np.array(tokenMatrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'),







def load_data(embeddings, path_dataset):
    files = [os.path.join(path_dataset, 'train.txt'), os.path.join(path_dataset, 'test.txt')]
    for fileIdx in range(len(files)):
        file = files[fileIdx]
        for line in open(file):
            splits = line.strip().split('\t')

            label = splits[0]


            sentence = splits[3]
            tokens = sentence.split(" ")
            maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))
            for token in tokens:
                words[token.lower()] = True


    print("Max Sentence Lengths: ", maxSentenceLen)

    # :: Read in word embeddings ::
    # :: Read in word embeddings ::
    word2Idx = embeddings.vocabulary.dic_words_ids
    wordEmbeddings = embeddings.matrix


    print("Embeddings shape: ", wordEmbeddings.shape)
    print("Len words: ", len(words))



    # :: Create token matrix ::
    train_set = createTensor(files[0], word2Idx, max(maxSentenceLen))
    test_set = createTensor(files[1], word2Idx, max(maxSentenceLen))


    data = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx,
            'train_set': train_set, 'test_set': test_set}

    return data



    print("Data preprocessing done!")