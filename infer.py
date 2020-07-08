import numpy as np
import copy

def infer_beam_search(model,model_input,start_answer,maximum_text,max_keep=10,search_bredth=5,return_top=True,tokenizer=None):

    text_objs = [word_prediction(start_answer,maximum_text) for _ in range(1)]
    
    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return(words)

    for k in range(maximum_text):
        next_text_objs = []
        for i in text_objs:
                output = np.squeeze(model.predict([model_input,[i.get_input()]]))[k]

                for j in range(search_bredth):
                    max_index = np.argmax(output)
                    text_obj_new = copy.deepcopy(i)
                    text_obj_new.append_to_answer(max_index,output[max_index])
                    # print('Max index')
                    # print(max_index)
                    # print(text_obj_new.get_prediction())
                    # print(text_obj_new.get_probability())
                    output[max_index] = 0
                    next_text_objs.append(text_obj_new)

        text_objs = sorted(next_text_objs,key=lambda x: x.get_probability(),reverse=True)[:max_keep]

    if(return_top):
        import pdb; pdb.set_trace()  # breakpoint 08fcdcfa //
        # return tokenizer.sequences_to_texts([text_obj.get_prediction() for text_obj in text_objs])
        return [sequence_to_text(text_obj.get_prediction()) for text_obj in text_objs]

    import pdb; pdb.set_trace()  # breakpoint ff6e2c57 //
    return sequence_to_text(text_objs[0].get_prediction())




class word_prediction(object):
    probability = 1

    def __init__(self, start_answer,maximum_text):
        self.output_text_input = np.zeros(maximum_text)
        self.output_text_input[0] = start_answer
        self.output_text = []

    def append_to_answer(self, value,probability):
        self.output_text.append(value)
        if(len(self.output_text) < len(self.output_text_input)):
            self.output_text_input[len(self.output_text)] = self.output_text[-1]
        self.probability *= probability

    def get_input(self):
        return self.output_text_input

    def get_prediction(self):
        return self.output_text

    def get_probability(self):
        return self.probability
