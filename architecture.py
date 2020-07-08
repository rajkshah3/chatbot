

from keras.models import Sequential, Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector, Embedding, TimeDistributed, Input, Bidirectional, Concatenate, BatchNormalization, Embedding



def lstm_enc_dec_vector(n_units=512,word_vspace = 100,n_timesteps_in=30,n_timesteps_out=30,input_vspace=5000):
    print(n_timesteps_in)
    encoder_orig_inputs = Input(shape=(n_timesteps_in,))
    embedder = Embedding(input_dim =input_vspace,output_dim=word_vspace,name='Body-Word-Embedding', mask_zero=False,input_length=n_timesteps_in)
    embedder2 = Embedding(input_dim =input_vspace,output_dim=word_vspace,name='Body-Word-Embedding-output', mask_zero=False,input_length=n_timesteps_out)

    encoder_inputs = embedder(encoder_orig_inputs)
    encoder_inputs = Bidirectional(LSTM(n_units, return_state=False, return_sequences=True))(encoder_inputs)
    # encoder_inputs = Bidirectional(LSTM(n_units, return_state=False, return_sequences=True))(encoder_inputs)
    encoder = Bidirectional(LSTM(n_units, return_state=True, return_sequences=False))
    encoder_sequence = Bidirectional(LSTM(n_units, return_state=True, return_sequences=True))

    encoder_outputs, state_hf, state_cf, state_hb, state_cb = encoder(encoder_inputs)
    encoder_outputs_seq, state_hf_seq, state_cf_seq, state_hb_seq, state_cb_seq = encoder_sequence(encoder_inputs)

#attention goes here to get state_h and state_c from all over the sequence instead of end!
    encoder_states_seq = [Concatenate()([state_hf_seq,state_hb_seq]), Concatenate()([state_cf_seq,state_cb_seq])]    


    encoder_states = [Concatenate()([state_hf,state_hb]), Concatenate()([state_cf,state_cb])]
    # define training decoder
    print(n_timesteps_out)
    decoder_orig_inputs = Input(shape=(n_timesteps_out,))
    decoder_inputs = embedder2(decoder_orig_inputs)
    decoder_lstm = LSTM(n_units*2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decode_lstm_1 = LSTM(n_units*2, return_sequences=True, return_state=False)
    decode_lstm_2 = LSTM(n_units*2, return_sequences=True, return_state=False)
    decode_tdd_output = TimeDistributed(Dense(input_vspace,activation='softmax'))

    decoder_outputs = decode_lstm_1(decoder_outputs)
    decoder_outputs = decode_lstm_2(decoder_outputs)
    
    decoder_outputs = decode_tdd_output(decoder_outputs)
    

    model = Model([encoder_orig_inputs, decoder_orig_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_orig_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units*2,))
    decoder_state_input_c = Input(shape=(n_units*2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = decode_lstm_1(decoder_outputs)
    
    decoder_outputs = decode_lstm_2(decoder_outputs)
    
    decoder_outputs = decode_tdd_output(decoder_outputs)
    

    decoder_model = Model([decoder_orig_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model



def lstm_enc_dec_vector_final_layer_attention(n_units=512,word_vspace = 100,n_timesteps_in=30,n_timesteps_out=30,input_vspace=5000):
#   https://arxiv.org/pdf/1508.04025.pdf

    print(n_timesteps_in)
    encoder_orig_inputs = Input(shape=(n_timesteps_in,))
    embedder = Embedding(input_dim =input_vspace,output_dim=word_vspace,name='Body-Word-Embedding', mask_zero=False,input_length=n_timesteps_in)
    embedder2 = Embedding(input_dim =input_vspace,output_dim=word_vspace,name='Body-Word-Embedding-output', mask_zero=False,input_length=n_timesteps_out)

    encoder_inputs = embedder(encoder_orig_inputs)
    encoder_l1_out, state_l1_hf, state_l1_cf, state_l1_hb, state_l1_cb  = Bidirectional(LSTM(n_units, return_state=True, return_sequences=True))(encoder_inputs)
    encoder_l1_states = [Concatenate()([state_hf_seq,state_hb_seq]), Concatenate()([state_cf_seq,state_cb_seq])]    
    # encoder_inputs = Bidirectional(LSTM(n_units, return_state=False, return_sequences=True))(encoder_inputs)
    encoder = Bidirectional(LSTM(n_units, return_state=True, return_sequences=False))
    encoder_sequence = Bidirectional(LSTM(n_units, return_state=True, return_sequences=True))

    encoder_l2, state_hf, state_cf, state_hb, state_cb = encoder(encoder_l1)
    encoder_l2_seq, state_hf_seq, state_cf_seq, state_hb_seq, state_cb_seq = encoder_sequence(encoder_l1)

#attention goes here to get state_h and state_c from all over the sequence instead of end!
    encoder_states_seq = [Concatenate()([state_hf_seq,state_hb_seq]), Concatenate()([state_cf_seq,state_cb_seq])]    


    encoder_states = [Concatenate()([state_hf,state_hb]), Concatenate()([state_cf,state_cb])]
    # define training decoder
    print(n_timesteps_out)
    decoder_orig_inputs = Input(shape=(n_timesteps_out,))
    decoder_inputs = embedder2(decoder_orig_inputs)
    decoder_lstm = LSTM(n_units*2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decode_lstm_1 = LSTM(n_units*2, return_sequences=True, return_state=False)
    decode_lstm_2 = LSTM(n_units*2, return_sequences=True, return_state=False)
    decode_tdd_output = TimeDistributed(Dense(input_vspace,activation='softmax'))

    decoder_outputs = decode_lstm_1(decoder_outputs)
    decoder_outputs = decode_lstm_2(decoder_outputs)
    
    decoder_outputs = decode_tdd_output(decoder_outputs)
    

    model = Model([encoder_orig_inputs, decoder_orig_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_orig_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units*2,))
    decoder_state_input_c = Input(shape=(n_units*2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = decode_lstm_1(decoder_outputs)
    
    decoder_outputs = decode_lstm_2(decoder_outputs)
    
    decoder_outputs = decode_tdd_output(decoder_outputs)
    

    decoder_model = Model([decoder_orig_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model
def lstm_enc_dec_vector_residual_lstm(n_units=512,word_vspace = 100,n_timesteps_in=30,n_timesteps_out=30,input_vspace=5000):
    print(n_timesteps_in)
    encoder_orig_inputs = Input(shape=(n_timesteps_in,))
    embedder = Embedding(input_dim =input_vspace,output_dim=word_vspace,name='Body-Word-Embedding', mask_zero=False,input_length=n_timesteps_in)
    embedder2 = Embedding(input_dim =input_vspace,output_dim=word_vspace,name='Body-Word-Embedding-output', mask_zero=False,input_length=n_timesteps_out)
    lstm1 = Bidirectional(LSTM(n_units, return_state=False, return_sequences=True))
    lstm2 = Bidirectional(LSTM(n_units, return_state=False, return_sequences=True))
    encoder_inputs = embedder(encoder_orig_inputs)
    lstm_output = lstm1(encoder_inputs)
    lstm_output = Concatenate([lstm_output,encoder_inputs])
    lstm_output = lstm2(lstm_output)

    # encoder_inputs = Bidirectional(LSTM(n_units, return_state=False, return_sequences=True))(encoder_inputs)
    encoder_output = Bidirectional(LSTM(n_units, return_state=True, return_sequences=False))

    encoder_outputs, state_hf, state_cf, state_hb, state_cb = encoder(encoder_output)

#attention goes here to get state_h and state_c from all over the sequence instead of end!



    encoder_states = [Concatenate()([state_hf,state_hb]), Concatenate()([state_cf,state_cb])]
    # define training decoder
    print(n_timesteps_out)
    decoder_orig_inputs = Input(shape=(n_timesteps_out,))
    decoder_inputs = embedder2(decoder_orig_inputs)
    decoder_lstm = LSTM(n_units*2, return_sequences=True, return_state=True)
    decode_lstm_1 = LSTM(n_units*2, return_sequences=True, return_state=False)
    decode_lstm_2 = LSTM(n_units*2, return_sequences=True, return_state=False)
    decode_tdd_output = TimeDistributed(Dense(input_vspace,activation='softmax'))

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = Concatenate([decoder_outputs,decoder_inputs])
    decoder_outputs = decode_lstm_1(decoder_outputs)
    decoder_outputs = Concatenate([decoder_outputs,decoder_inputs])
    decoder_outputs = decode_lstm_2(decoder_outputs)
    decoder_outputs = Concatenate([decoder_outputs,decoder_inputs])
    decoder_outputs = decode_tdd_output(decoder_outputs)
    

    model = Model([encoder_orig_inputs, decoder_orig_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_orig_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units*2,))
    decoder_state_input_c = Input(shape=(n_units*2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = Concatenate([decoder_outputs,decoder_inputs])

    decoder_outputs = decode_lstm_1(decoder_outputs)
    decoder_outputs = Concatenate([decoder_outputs,decoder_inputs])

    decoder_outputs = decode_lstm_2(decoder_outputs)
    decoder_outputs = Concatenate([decoder_outputs,decoder_inputs])

    decoder_outputs = decode_tdd_output(decoder_outputs)
    
    decoder_model = Model([decoder_orig_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model
