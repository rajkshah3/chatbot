import pandas as pd
from data.read_squad import load_data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from infer import infer_beam_search
from keras.models import load_model

checkpoint = ModelCheckpoint('./checkpoints/checkpoint_{epoch:02d}-{val_loss:.2f}.pkl', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


def clean_text(this_row):

    this_row = this_row.str.replace(r'http\S+', '')
    this_row = this_row.str.replace("\n",' ')
    this_row = this_row.str.replace(u"!",'')
    # this_row = this_row.str.replace('-', ' ')
    # this_row = this_row.str.replace('"','')
    # this_row = this_row.str.replace("'",'')
    # this_row = this_row.str.replace("`",'')
    # this_row = this_row.str.replace("#",'')
    # this_row = this_row.str.replace("@",'')
    # this_row = this_row.str.replace("&",'')
    this_row = this_row.str.replace("?",' ?')
    this_row = this_row.str.replace(".",' .')
    this_row = this_row.str.replace(",",' ,')
    this_row = this_row.str.replace("(",' (')
    this_row = this_row.str.replace(")",' )')
    this_row = this_row.str.replace(":",' :')
    # this_row = textacy.preprocess.preprocess_text(this_row,
    #                                               fix_unicode=True, no_urls=True, no_emails=True,
    #                                               lowercase=True, no_contractions=True,
    #                                               no_numbers=True, no_currency_symbols=True, no_punct=False)

    return this_row


def count_words(li):
    li = ' '.join(li.tolist())
    return len(set(li.split(' ')))

def count_lengths(li):
    lengths = li.str.lower().str.split().apply(len)
    mean = lengths.mean()
    std = lengths.std()
    maximum = lengths.max()
    median = lengths.median()
    return mean, std, maximum


def main():
    df = load_data()
    df = df.head(20000)
    df['context'] = clean_text(df['context'].str.lower())
    df['question'] = clean_text(df['question'].str.lower())
    df['text'] = clean_text(df['text'].str.lower())

    start_text = '<start>'
    end_text = '<end>'
    question_text = ' questionstart '

    df['questionstart'] = 'questionstart'

    df['start'] = start_text+' '
    df['end'] =  ' ' + end_text

    df['augtext'] = df["start"].map(str) + df['start'] + df["text"]
    df['text'] = df["start"].map(str) + df["text"] + df["end"]
    df['context'] = df['context'].map(str) + df['questionstart'] + df['question']


    words = count_words(df['context'] + df['text'])
    mean, std, maximum_con  = count_lengths(df['context'])
    mean, std, maximum_text  = count_lengths(df['text'])
    #Tokenize

    tok = Tokenizer(num_words=words,filters='!"#$%&*+-/;=?@[]^_`{|}~',oov_token=1) 
    tok.fit_on_texts(list(df['context'].unique()) + df['text'].tolist())

    tokenized_paragraphs = tok.texts_to_sequences(df['context'])
    tokenized_answers = tok.texts_to_sequences(df['text'])
    tokenized_answers2 = tok.texts_to_sequences(df['augtext'])


    padded_paragraphs = pad_sequences(tokenized_paragraphs, maxlen=maximum_con, dtype='int32', padding='pre', truncating='post', value=0)
    padded_answers = pad_sequences(tokenized_answers, maxlen=maximum_text, dtype='int32', padding='post', truncating='post', value=0)
    padded_answers2 = pad_sequences(tokenized_answers2, maxlen=maximum_text, dtype='int32', padding='post', truncating='post', value=0)

    print('words: ' + str(words))
    print('tokens: ' + str(len(tok.word_counts)))
    print(padded_answers)
    words = len(tok.word_counts)+2
    from architecture import lstm_enc_dec_vector

    # model, model_encoder, mdoel_decoder = lstm_enc_dec_vector(n_units=100,word_vspace=50,n_timesteps_in=maximum_con,n_timesteps_out=maximum_text,input_vspace=words)

    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    # model.summary()
    
    model = load_model('./checkpoints/checkpoint_45-0.67.pkl')
    # model.fit(x=[padded_paragraphs,padded_answers2],y=np.expand_dims(padded_answers,-1),shuffle=True,batch_size=200,validation_split=0.1,epochs=10, verbose=1,callbacks=[tbCallBack,checkpoint])
    # model.save_weights("model.h5")
    # model.save('model.pkl')
    outputs = infer_beam_search(model,[padded_paragraphs[1]],tok.word_index[start_text],maximum_text=maximum_text,max_keep=10,search_bredth=5,return_top=True,tokenizer=tok)

    print(outputs)

main()
