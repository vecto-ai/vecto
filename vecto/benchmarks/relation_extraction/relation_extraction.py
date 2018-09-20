from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility
import gzip
import sys
import pickle as pkl
from .preprocess import load_data
from ..base import Benchmark

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.regularizers import Regularizer
from keras.preprocessing import sequence


# Function to calculate the precision
def getPrecision(pred_test, yTest, targetLabel):
    # Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0

    for idx in range(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1

            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1

    if correctTargetLabelCount == 0:
        return 0

    return float(correctTargetLabelCount) / targetLabelCount

class Relation_extraction(Benchmark):

    def __init__(self, batch_size=16, nb_filter=100, filter_length=3, hidden_dims=100, nb_epoch=100, position_dims=50):
        self.batch_size = batch_size
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.hidden_dims = hidden_dims
        self.nb_epoch = nb_epoch
        self.position_dims = position_dims


    def run(self, embeddings, path_dataset):

        print("Load dataset")
        data = load_data()

        yTrain, sentenceTrain, positionTrain1, positionTrain2 = data['train_set']
        yTest, sentenceTest, positionTest1, positionTest2 = data['test_set']

        max_position = max(np.max(positionTrain1), np.max(positionTrain2)) + 1

        n_out = max(yTrain) + 1
        # train_y_cat = np_utils.to_categorical(yTrain, n_out)
        max_sentence_len = sentenceTrain.shape[1]

        print(sentenceTrain[10])

        print("sentenceTrain: ", sentenceTrain.shape)
        print("positionTrain1: ", positionTrain1.shape)
        print("yTrain: ", yTrain.shape)

        print("sentenceTest: ", sentenceTest.shape)
        print("positionTest1: ", positionTest1.shape)
        print("yTest: ", yTest.shape)

        print("Embeddings: ", embeddings.shape)

        words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
        words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input)

        distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
        distance1 = Embedding(max_position, position_dims)(distance1_input)

        distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
        distance2 = Embedding(max_position, position_dims)(distance2_input)

        output = concatenate([words, distance1, distance2], -1)

        output = Convolution1D(filters=nb_filter,
                               kernel_size=filter_length,
                               padding='same',
                               activation='tanh',
                               strides=1)(output)

        # we use standard max over time pooling
        output = GlobalMaxPooling1D()(output)

        output = Dropout(0.25)(output)
        output = Dense(n_out, activation='softmax')(output)

        # create the model
        model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=[output])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        model.summary()

        print("Start training")

        max_prec, max_rec, max_acc, max_f1 = 0, 0, 0, 0
        accs = []

        def predict_classes(prediction):
            return prediction.argmax(axis=-1)

        # for epoch in range(nb_epoch):
        model.fit([sentenceTrain, positionTrain1, positionTrain2], yTrain, batch_size=batch_size, verbose=True,
                  epochs=nb_epoch)
        pred_test = predict_classes(model.predict([sentenceTest, positionTest1, positionTest2], verbose=False))

        dctLabels = np.sum(pred_test)
        totalDCTLabels = np.sum(yTest)

        acc = np.sum(pred_test == yTest) / float(len(yTest))
        max_acc = max(max_acc, acc)
        print("Accuracy: %.4f (max: %.4f)" % (acc, max_acc))

        f1Sum = 0
        f1Count = 0
        for targetLabel in range(1, max(yTest)):
            prec = getPrecision(pred_test, yTest, targetLabel)
            recall = getPrecision(yTest, pred_test, targetLabel)
            f1 = 0 if (prec + recall) == 0 else 2 * prec * recall / (prec + recall)
            f1Sum += f1
            f1Count += 1
        accs.append(max_acc)

        macroF1 = f1Sum / float(f1Count)
        max_f1 = max(max_f1, macroF1)
        print("Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1))

    def get_result(self, embeddings, path_dataset):
        results = self.run(embeddings, path_dataset)
        return results


