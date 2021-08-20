from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute
from tensorflow.keras.layers import Dropout, add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from functools import reduce
import pickle
import tarfile
import numpy as np
import re
import os
import time

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data \
            if not max_length or len(flatten(story)) < max_length]
    return data



def vectorize_stories(data):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers))

try:
    path = get_file('babi_tasks_1-20_v1-2.tar.gz',
        origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print("""
Error downloading dataset, please download it manually:\n'
$ wget http://www.thespermwhale.com/jaseweston/babi/
    tasks_1-20_v1-2.tar.gz\n'
$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/
    babi-tasks-v1-2.tar.gz""")
    raise
tar = tarfile.open(path)

train_stories = get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa4_two-arg-relations_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa5_three-arg-relations_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa7_counting_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa8_lists-sets_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa9_simple-negation_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa10_indefinite-knowledge_train.txt'))+ get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa11_basic-coreference_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa12_conjunction_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa13_compound-coreference_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa14_time-reasoning_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa15_basic-deduction_train.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa16_basic-induction_train.txt'))+ get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa17_positional-reasoning_train.txt'))+ get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa18_size-reasoning_train.txt'))+ get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa19_path-finding_train.txt'))+ get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa20_agents-motivations_train.txt'))
test_stories = get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa4_two-arg-relations_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa5_three-arg-relations_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa7_counting_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa8_lists-sets_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa9_simple-negation_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa10_indefinite-knowledge_test.txt'))+ get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa11_basic-coreference_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa12_conjunction_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa13_compound-coreference_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa14_time-reasoning_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa15_basic-deduction_test.txt')) + get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa16_basic-induction_test.txt'))+ get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa17_positional-reasoning_test.txt'))+ get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa18_size-reasoning_test.txt'))+ get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa19_path-finding_test.txt'))+ get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa20_agents-motivations_test.txt'))

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')


for s in list(enumerate(vocab)):
    print(s)

for i in range(5):
    print("Story: {}".format(' '.join(train_stories[i][0])))
    print("Query: {}".format(' '.join(train_stories[i][1])))
    print("Answer: {}".format(train_stories[i][2]))
    print("---")

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

print('Vectorizing the word sequences...')
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train \
    = vectorize_stories(train_stories)
inputs_test, queries_test, answers_test \
    = vectorize_stories(test_stories)

print('Compiling...')
#
# input_sequence = Input((story_maxlen,))
# question = Input((query_maxlen,))
#
# input_encoder_m = Sequential()
# input_encoder_m.add(Embedding(input_dim=vocab_size,
#                               output_dim=64))
# input_encoder_m.add(Dropout(0.3))
#
# input_encoder_c = Sequential()
# input_encoder_c.add(Embedding(input_dim=vocab_size,
#                               output_dim=query_maxlen))
# input_encoder_c.add(Dropout(0.3))
#
#
#
# question_encoder = Sequential()
# question_encoder.add(Embedding(input_dim=vocab_size,
#                                output_dim=64,
#                                input_length=query_maxlen))
# question_encoder.add(Dropout(0.3))
#
#
# input_encoded_m = input_encoder_m(input_sequence)
# input_encoded_c = input_encoder_c(input_sequence)
# question_encoded = question_encoder(question)
#
#
# match = dot([input_encoded_m, question_encoded], axes=(2, 2))
# match = Activation('softmax')(match)
#
# response = add([match, input_encoded_c])
# response = Permute((2, 1))(response)
#
# answer = concatenate([response, question_encoded])
#
# answer = LSTM(32)(answer)
#
# answer = Dropout(0.3)(answer)
# answer = Dense(vocab_size)(answer)
#
# answer = Activation('softmax')(answer)
#
# # build the final model
# model = Model([input_sequence, question], answer)
# model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# print("Done.")
#
# start_time = time.time()
# # train
# model.fit([inputs_train, queries_train], answers_train,
#           batch_size=32,
#           epochs=120,
#           validation_data=([inputs_test, queries_test], answers_test))
#
# # save
# save_path = "./data/"
# # save entire network to HDF5 (save everything, suggested)
# model.save(os.path.join(save_path,"chatbot.h5"))
# # save the vocab too, indexes must be the same
# pickle.dump( vocab, open( os.path.join(save_path,"vocab.pkl"), "wb" ) )
#
# elapsed_time = time.time() - start_time
# print("Elapsed time: {}".format(hms_string(elapsed_time)))

save_path = "/home/dani/Desktop/jarvalous/data"
model = load_model(os.path.join(save_path,"chatbot_818.h5"))
vocab = pickle.load( open( os.path.join(save_path,"vocab_818.pkl"), "rb" ) )

pred = model.predict([inputs_test, queries_test])
print(pred)
pred = np.argmax(pred,axis=1)
print(pred)

score = metrics.accuracy_score(answers_test, pred)
print("Final accuracy: {}".format(score))

print("Remember, I only know these words: {}".format(vocab))
print()
story = "Daniel got the chocolate. Afterwards he gave the chocolate to Fred."\
     "Fred went to the bedroom."

story = input("Tell me a story: ")

while True:
    query = input ("ask smth: ") #"Who did Daniel give the chocolate to?"
    # query = "Where is Fred?"

    adhoc_stories = (tokenize(story), tokenize(query), '?')

    adhoc_train, adhoc_query, adhoc_answer = vectorize_stories([adhoc_stories])

    pred = model.predict([adhoc_train, adhoc_query])
    print(pred[0])
    pred = np.argmax(pred,axis=1)
    print("Answer: {}({})".format(vocab[pred[0]-1],pred))
