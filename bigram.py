import torch

file_path = "names.txt" 
train_ratio = 0.9

class BigramModel():
    
    def __init__(self,file_path,train_ratio):
        self.file_path = file_path
        self.train_ratio = train_ratio
        self.chars = []
        self.count_dict = {}
        self.count_dict_test = {}
        self.read_file()
    
    def read_file(self):
        with open(self.file_path,'r') as f:
            self.text = f.read().split()
        self.start_end_char_append(self.text)

    def start_end_char_append(self,text):    
        self.chars = sorted(list(set("".join(text))))
        self.chars.append("<S>")
        self.chars.append("<E>")
        self.vocab_size = len(self.chars)
        self.strTOint()
        self.intTOstr()
        self.train_test_split()
        
    # Creating a mapping from string to integers and vice versa
    def strTOint(self):
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}

    def intTOstr(self):
        self.itos = {i:ch for i,ch in enumerate(self.chars)}

    def encoder(self,seq):
        enc_list = []
        for ch in seq:
            enc_list.append(self.stoi[ch])
        return enc_list
        
    def decoder(self,seq):
        dec_string = ""
        for ind in seq:
            dec_string += self.itos[ind]
        return dec_string

    def train_test_split(self):

        idx_train_data = []
        idx_test_data = []
        
        n = int(self.train_ratio*len(self.text))
        self.train = self.text[:n]
        self.test = self.text[n:]
        print(f'Number of Words (Training) = {len(self.train)}')
        print(f'Number of Words (Testing) = {len(self.test)}')
        
        start_sym_ind = self.stoi["<S>"]
        end_sym_ind = self.stoi["<E>"]
        
        for w_tr in self.train:
            idx_train_data.extend([start_sym_ind] + self.encoder(w_tr) + [end_sym_ind])
        self.train_data = torch.tensor(idx_train_data)

        for w_te in self.test:
            idx_test_data.extend([start_sym_ind] + self.encoder(w_te) + [end_sym_ind])
        self.test_data = torch.tensor(idx_test_data)
        
        self.count_dictionary()

    def count_dictionary(self):
        for word in self.train:
            word = ["<S>"] + list(word) + ["<E>"]  # prepend <S>, append <E> to the current word
            for ch1, ch2 in zip(word, word[1:]):
                self.count_dict[(ch1, ch2)] = self.count_dict.get((ch1, ch2), 0) + 1
        self.norm_twod_prob_mat()

    def norm_twod_prob_mat(self):
        self.mat = torch.zeros((self.vocab_size,self.vocab_size),dtype = torch.float)
        for row in range(self.vocab_size):
            for col in range(self.vocab_size):
                self.mat[row,col] = self.count_dict.get((self.itos[row],self.itos[col]),0)
        row_sum = self.mat.sum(dim=1,keepdim=True)
        row_sum[row_sum == 0] = 1
        self.norm_mat = self.mat / row_sum
        self.sequence_generation()

    def sequence_generation(self): 
        num_of_names = 10
        max_length = 10
        for name in range(num_of_names):
            myinput = ["<S>"]
            while True:
                ind_input = self.stoi[myinput[-1]]
                row = self.norm_mat[ind_input,:]
                sample = torch.multinomial(row,num_samples=1)
                next_ch = self.itos[sample.item()]
                if (next_ch == "<E>") or (len(myinput) > max_length):
                    break
                myinput.append(next_ch)
            print("".join(myinput[1:]))

    def model_evaluation(self):
        total_log_prob = 0.0
        neg_log_likelihood = 0
        num_biagrams = 0
        for ind,tst_ind in enumerate(self.test_data[:-1]):
            num_biagrams += 1
            prob = self.norm_mat[tst_ind,self.test_data[ind+1]]
            prob = torch.clamp(prob,1e-10)
            total_log_prob += torch.log(prob)
        neg_log_likelihood = -total_log_prob / num_biagrams
        return neg_log_likelihood
            

b = BigramModel(file_path,train_ratio)
nll = b.model_evaluation()
print(f"Negative Log Likelihood = {nll:.4f}")