import nltk
import spacy
import numpy as np

nlp = spacy.load('en_core_web_lg')

datasets = ['barbot','aws','hsbc','nhs','powershift']

data_dir = '/Users/raj.shah/projects/chatbot/data/'

def clear_text(s):
    cases = ['-',"'",'"','\n',',','?','.','{','}']
    for case in cases:
        s = s.replace(case," ")
    s = s.strip()
    s = s.rstrip('\r\n')
    s = s.split()
    s = ' '.join([word for word in s if (not nlp.vocab[word.lower()].is_stop)])
    return s

def load_words():
    data_dict = {}
    for data in datasets:
        with open(data_dir+data+'_utterances.txt', "r") as myfile:
            lines=myfile.read()
        # print(lines)
        lines = clear_text(lines)
        words = lines.split()
        words = [ word  for word in words ]
        data_dict[data] = words

    return data_dict

def load_lines():
    data_dict = {}
    for data in datasets:
        with open(data_dir+data+'_utterances.txt', "r") as myfile:
            lines=myfile.readlines()
        # print(lines)
        lines = [clear_text(line) for line in lines]
        print(data)
        print(lines)
        data_dict[data] = lines

    return data_dict

def gaussian(x, mu=0, sig=300):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def sentance_similarity(s1,s2):
    s1 = clear_text(s1)
    s2 = clear_text(s2)
    comb = 0
    counter = 0
    s1 = s1.split()
    s2 = s2.split()
    for i in range(len(s1)):
        for j in range(i,len(s2)):
            comb += nlp.vocab[s1[i].lower()].similarity(nlp.vocab[s2[j].lower()])
            counter += 1
    if counter == 0:
        return 0
    else:
        return comb/counter

data_dict = load_lines()
sentences = data_dict['barbot']
for s1 in sentences:
    for s2 in sentences:
        print("s1: ",s1)
        print("s2: ",s2)
        print('Similarity: ',sentance_similarity(s1,s2))





