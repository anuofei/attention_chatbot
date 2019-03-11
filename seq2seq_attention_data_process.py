from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  time
import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np
import random
class Batch:
    #batch类，里面包含了encoder输入，decoder输入，decoder标签，decoder样本长度mask
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []
PAD, UNK,GO, EOS = 0, 1, 2, 3

def make_ask_response_dual():
    dbfile = open('all_ask_vec_shuffle_pad_obj', 'rb')
    all_ask_vec_shuffle2 = pickle.load(dbfile)
    dbfile2 = open('all_response_vec_shuffle_pad_obj', 'rb')
    all_response_vec_shuffle2 = pickle.load(dbfile2)
    all_ask_vec_shuffle = []
    all_response_vec_shuffle=[]
    for one_ask_vec_shuffle2 in all_ask_vec_shuffle2:
        one_ask_vec_shuffle = list(reversed(one_ask_vec_shuffle2))
        all_ask_vec_shuffle.append(one_ask_vec_shuffle)
    for one_response_vec_shuffle2 in all_response_vec_shuffle2:

        one_response_vec_shuffle = list(one_response_vec_shuffle2)

        all_response_vec_shuffle.append(one_response_vec_shuffle)

    ask_response = zip(all_ask_vec_shuffle, all_response_vec_shuffle)
    ask_response_list2=[]
    for ask_response in ask_response:
        ask_response_list2.append(ask_response)
    ask_response_list=np.array(ask_response_list2)
    dbfilereverse = open('ask_response_text_to_index_reverse_obj2', 'wb')
    pickle.dump(ask_response_list, dbfilereverse)

def getBatches2(batch_size):

    dbfile = open('all_ask_vec_shuffle_pad_obj', 'rb')
    all_ask_vec_shuffle_pad = pickle.load(dbfile)
    dbfile = open('ask_response_text_to_index_reverse_obj', 'rb')
    ask_response_text_to_index_reverse2 = pickle.load(dbfile)
    ask_response_text_to_index_reverse=list(ask_response_text_to_index_reverse2)[:int(len(all_ask_vec_shuffle_pad)*0.7)]
    """
    获取batch
    """
    print('int(len(all_ask_vec_shuffle_pad)*0.7)==', int(len(all_ask_vec_shuffle_pad)*0.7))
    for batch_i in range(0, len(ask_response_text_to_index_reverse) // batch_size):

        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        ask_res_batch2 = ask_response_text_to_index_reverse[start_i:start_i + batch_size]
        ask_res_batch=np.array(ask_res_batch2)

        yield ask_res_batch

def ask_response_dual():
    dbfile = open('all_ask_vec_shuffle_obj', 'rb')
    all_ask_vec_shuffle = pickle.load(dbfile)
    dbfile = open('all_response_vec_shuffle_obj', 'rb')
    all_response_vec_shuffle = pickle.load(dbfile)
    ask_response=zip(all_ask_vec_shuffle,all_response_vec_shuffle)
    ask_response_list=[]
    for x in ask_response:

        ask_response_list.append(list(x))

    dbfile = open('ask_response_all_vec_shuffle_obj', 'wb')  # 必须以2进制打开文件，否则pickle无法将对象序列化只文件
    pickle.dump(ask_response_list, dbfile)

def createBatch(samples):
    '转化成能够输入的数据'

    '''
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer]的形式，id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    batch = Batch()
    batch.encoder_inputs_length =  [len(sample[0]) for sample in samples]

    batch.decoder_targets_length = [len(sample[1]) for sample in samples]

    max_source_length = max(batch.encoder_inputs_length)
    min_source_length = min(batch.encoder_inputs_length)

    max_target_length = max(batch.decoder_targets_length)


    for sample in samples:

        source = list(reversed(sample[0]))

        pad = [PAD] * (max_source_length - len(source))

        batch.encoder_inputs.append(pad + source)


        #将target进行PAD，并添加END符号
        target = sample[1]

        pad = [PAD] * (max_target_length - len(target))

        batch.decoder_targets.append(target + pad)


    return batch
def getBatches(batch_size):
    '将数据shuffle，分批次'
    '''
    根据读取出来的所有数据和batch_size将原始数据分成不同的小batch。对每个batch索引的样本调用createBatch函数进行处理
    :param data: loadDataset函数读取之后的trainingSamples，就是QA对的列表
    :param batch_size: batch大小
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''
    dbfile = open('ask_response_all_vec_shuffle_del0_obj', 'rb')
    data = pickle.load(dbfile)

    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():

        batch = createBatch(samples)

        batches.append(batch)

    return batches
def createBatch2():
    '转化成能够输入的数据'

    '''
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer]的形式，id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    dbfile = open('ask_response_all_vec_shuffle_del0_obj', 'rb')
    samples = pickle.load(dbfile)
    batch = Batch()
    batch.encoder_inputs_length =  [len(sample[0]) for sample in samples]

    batch.decoder_targets_length = [len(sample[1]) for sample in samples]

    max_source_length = max(batch.encoder_inputs_length)
    min_source_length = min(batch.encoder_inputs_length)
    print('min_source_length===', min_source_length)
   
    max_target_length = max(batch.decoder_targets_length)
    min_target_length = min(batch.decoder_targets_length)
    print('min_target_length===', min_target_length)



    return batch
if __name__=='__main__':
    1==1
    createBatch2()
