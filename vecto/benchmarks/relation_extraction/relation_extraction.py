from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility
import gzip
import sys
import pickle as pkl
from .preprocess import load_data
from ..base import Benchmark
import os

from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D


# Function to calculate the precision
def getPrecision(pred_test, yTest, targetLabel):
    # Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0

    for idx, prediction in enumerate(pred_test):
        if prediction == targetLabel:
            targetLabelCount += 1

            if prediction == yTest[idx]:
                correctTargetLabelCount += 1

    if correctTargetLabelCount == 0:
        return 0

    return float(correctTargetLabelCount) / targetLabelCount

class Relation_extraction(Benchmark):

    def __init__(self, batchsize=16, nb_filter=100, filter_length=3, hidden_dims=100, epoch=1, position_dims=50):
        self.batchsize = batchsize
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.hidden_dims = hidden_dims
        self.epoch = epoch
        self.position_dims = position_dims

    def run(self, embeddings, dataset):
        print("Load dataset")
        path_dataset = dataset.path
        data = load_data(embeddings, path_dataset)

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

        print("Embeddings: ", embeddings.matrix.shape)

        words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
        words = Embedding(embeddings.matrix.shape[0], embeddings.matrix.shape[1], weights=[embeddings.matrix],
                          trainable=False)(words_input)
        distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
        distance1 = Embedding(max_position, self.position_dims)(distance1_input)
        distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
        distance2 = Embedding(max_position, self.position_dims)(distance2_input)
        output = concatenate([words, distance1, distance2], -1)
        output = Convolution1D(filters=self.nb_filter,
                               kernel_size=self.filter_length,
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
        model.fit([sentenceTrain, positionTrain1, positionTrain2], yTrain, batch_size=self.batchsize, verbose=True,
                  epochs=self.epoch)
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

        experiment_setup = self.__dict__
        experiment_setup["embeddings"] = embeddings.metadata
        experiment_setup["category"] = "default"
        experiment_setup["dataset"] = os.path.basename(path_dataset)
        experiment_setup["method"] = 'cnn'
        experiment_setup['task'] = 'relation_extraction'
        result = {}
        result['experiment_setup'] = experiment_setup
        result['experiment_setup']['default_measurement'] = 'macroF1'
        result['result'] = []
        result['result'] = {"macroF1": macroF1, "max_f1": max_f1, "accuracy": acc, "max_accuracy": max_acc}
        return result

    def get_result(self, embeddings, path_dataset):
        results = self.run(embeddings, path_dataset)
        return [results]
