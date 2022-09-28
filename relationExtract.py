from collections import Counter
import pickle
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras_contrib.layers.crf import CRF


def getWordAndTagId(filePath):
    with open(filePath, encoding='utf-8') as file:
        wordsAndtags = [line.split(' ') for line in file]

        words = [i[0].strip() for i in wordsAndtags if len(i)==2 ]

        tags = [i[1].strip()  for i in wordsAndtags if len(i)==2 ]
        words = Counter(words)
        tags = Counter(tags)
        words = sorted(words.items(), key=lambda x: -x[1])
        words = words[:4000]
        tags = sorted(tags.items(), key=lambda x: -x[1])
        word_size = len(words)
        word2id = {count[0]: index for index, count in enumerate(words,start=1)}
        id2word = {index: count[0] for index, count in enumerate(words,start=1)}
        tag2id = {count[0]: index for index, count in enumerate(tags)}
        id2tag = {index: count[0] for index, count in enumerate(tags)}
        word2id['<PAD>'] = 0
        word2id['<UNK>'] = word_size + 1
        return word2id, tag2id, id2word, id2tag

def saveWordAndTagId(word2id,tag2id):
    word2idFile = open('data/word2id', 'wb')
    tag2idFile = open('data/tag2id', 'wb')
    pickle.dump(word2id, word2idFile)
    pickle.dump(tag2id, tag2idFile)
    word2idFile.close()
    tag2idFile.close()

def loadWordAndTagId():
    word2idFile = open('data/word2id', 'wb')
    tag2idFile = open('data/tag2id', 'wb')
    word2id = pickle.load(word2idFile)
    tag2id = pickle.load(tag2idFile)
    return word2id, tag2id

def getSentencesAndTags(filePath):
    '''
    从文件里面获取句子和标注
    :param filePath:
    :return:
    '''
    with open(filePath,encoding='utf-8') as file:
        wordsAndtags=[line.split() for line in file]
        sentences=[]
        tags=[]
        sentence=[]
        tag=[]
        for wordAndTag in wordsAndtags:
            if len(wordAndTag)==2:
                sentence.append(wordAndTag[0])
                tag.append(wordAndTag[1])
            else:
                sentences.append(sentence)
                tags.append(tag)
                sentence=[]
                tag = []
    return sentences,tags

def sentencesAndTags2id(sentences,tags,word2id, tag2id):
    '''
    将句子和标注转换为id
    :param sentences:
    :param tags:
    :param word2id:
    :param tag2id:
    :return:
    '''
    sentencesIds = [[word2id.get(char,len(word2id)) for char in sentence] for sentence in sentences]
    tagsIds = [[tag2id[char] for char in tag] for tag in tags]
    return sentencesIds,tagsIds

def model(vocabSize,embeddingDim,inputLength,tagSize):
    model = Sequential()
    model.add(Embedding(vocabSize + 1,embeddingDim,input_length=inputLength,trainable=False,mask_zero=True))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(TimeDistributed(Dense(tagSize)))
    crf_layer = CRF(tagSize, sparse_target=True)
    model.add(crf_layer)
    model.compile('adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
    model.summary()
    return model


def get_triple(y_labels,input_data):
    subjects, predicates, objects = '', '', ''
    for s, t in zip(input_data, y_labels):
        if t in ('B-SUBJECT', 'I-SUBJECT'):
            subjects += ' ' + s if (t == 'B-SUBJECT') else s
        if t in ('B-PREDICATE', 'I-PREDICATE'):
            predicates += ' ' + s if (t == 'B-PREDICATE') else s
        if t in ('B-OBJECT', 'I-OBJECT'):
            objects += ' ' + s if (t == 'B-OBJECT') else s
    y_counter = Counter(y_labels)
    # 主 谓 宾、宾..
    if y_counter['B-SUBJECT'] == 1 and y_counter['B-PREDICATE'] == 1 and y_counter['B-OBJECT'] >= 1:
        for object in objects.strip().split(' '):
            print('抽取结果：',subjects+'-'+predicates+'->'+object)
    # 主 谓 宾、谓 宾..
    elif y_counter['B-SUBJECT'] == 1 and y_counter['B-PREDICATE'] > 1 and y_counter['B-OBJECT'] >= 1:
        for i in range(len(predicates.strip().split(' '))):
            print('抽取结果：',subjects+'-'+predicates.strip().split(' ')[i]+'->'+objects.strip().split(' ')[i])
    elif y_counter['B-SUBJECT'] > 1:
        get_triple(y_labels[y_labels.index('I-OBJECT'):],input_data[y_labels.index('I-OBJECT'):])


def predict(model,input_data,length,word2id,id2tag):
    '''
    预测
    :param model:
    :param inputData:
    :param length:
    :param word2id:
    :param id2tag:
    :return:
    '''
    input = [word2id.get(char,len(word2id)) for char in input_data]
    input = np.reshape(input,[1,-1])
    # 填充
    input = pad_sequences(input,maxlen=length)
    y = model.predict(input)
    # 输出为三维  转为二维
    y = y.reshape([-1,7])
    y = np.argmax(y,axis=1)
    # 去除填充的部分
    y = y[len(y)-len(input_data):]
    y = [id2tag[i] for i in y]
    get_triple(y,input_data)
    # print(y)

if __name__=='__main__':
    # 获取词典映射
    word2id, tag2id, id2word, id2tag = getWordAndTagId('train.txt')
    # saveWordAndTagId(word2id, tag2id)
    # 获取句子和标注
    sentences, tags=getSentencesAndTags('train.txt')
    # 将句子和标注转换为id
    sentencesIds, tagsIds = sentencesAndTags2id(sentences, tags,word2id, tag2id)
    # 将句子和标注进行填充，确保输入维度一致
    sentencesIds = pad_sequences(sentencesIds, padding='post')
    tagsIds = pad_sequences(tagsIds, padding='post')
    print(sentencesIds.shape)
    print(tagsIds.shape)
    # 载入模型
    model=model(len(word2id),100,sentencesIds.shape[1],len(tag2id))
    history = model.fit(sentencesIds, tagsIds.reshape([len(tagsIds),-1,1]), epochs=700)
    model.save('kg-mode.model')
    from keras_contrib.layers.crf import CRF, crf_loss, crf_viterbi_accuracy

    model = load_model('kg-mode.model', custom_objects={"CRF": CRF, 'crf_loss': crf_loss,
                                                   'crf_viterbi_accuracy': crf_viterbi_accuracy})
    while True:
        str = input("原始句子>")
        predict(model,str,sentencesIds.shape[1],word2id,id2tag)
