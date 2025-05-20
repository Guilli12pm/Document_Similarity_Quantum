import pandas as pd
import os
import sys

import torch
import torch.utils.data
from torch import nn
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cleanMethod import cleanText

_ = torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
NUM_LAYER = 2
NUM_QUBITS = 4
EPOCHS = 10
LEARNING_RATE = 0.005

data_dir = "./data_Quora/questions.csv"
data_dir_rest = "./data_Quora/rest.csv"

data_qu = pd.read_csv(data_dir)
data_rest = pd.read_csv(data_dir_rest)

data = pd.concat([data_qu,data_rest])
data = data_qu

print("Number of duplicate =",len(data[(data['is_duplicate']==1)]))
print("Number of non-duplicate =",len(data[(data['is_duplicate']==0)]))

remove_non_english = False

def give_dic(dataframe, remove_non_english):
    unique_words = set("UNK")

    for index, row in dataframe.iterrows():
        q1 = row['question1']
        q2 = row['question2']
        combined_text = f"{q1} {q2}"
        combined_text = cleanText(combined_text, remove_non_english)
        unique_words.update(combined_text.split())

    word_index_dict = {word: idx for idx, word in enumerate(unique_words)}
    return word_index_dict

print(f"Building dictionary -> remove non english words: {remove_non_english}")
dictionary = give_dic(data, remove_non_english)
dic_length = len(dictionary)
print(f"Dictionary build of size {dic_length}\n")

def word2index(word):
    if word in dictionary:
        return dictionary[word]
    else: return 0


def sentence2tensor(sentence, num_words):
    tensor = torch.zeros(num_words, dtype=torch.long)
    split_sent = sentence.split()[:num_words]
    #print(split_sent)
    for i, word in enumerate(split_sent):
        #print(word)
        tensor[i] = word2index(word)
    return tensor

tensor_sentences = []
target_labels = []

PADDING = 15

print(f"Building the tensors of padding {PADDING}")
for index, row in data.iterrows():
    tensor_c1 = sentence2tensor(row['question1'], PADDING)
    #print("test")
    tensor_c2 = sentence2tensor(row['question2'], PADDING)
    tup = torch.stack((tensor_c1, tensor_c2))
    tensor_sentences.append(tup)
    target_labels.append(torch.tensor(row['is_duplicate'], dtype=torch.long))
    
print(f"Padding built\n")

tensor_sentences = torch.stack(tensor_sentences)
target_labels = torch.stack(target_labels)

test_size=0.1
print(f"Dividing data into Train/Test - test size ratio = {test_size}")
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    range(len(target_labels)), 
    test_size=test_size, 
    shuffle=True, 
    stratify=target_labels,
    random_state=42
)

print("Starting Batching")
batch_size = BATCH_SIZE
train_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_sentences[train_idx], target_labels[train_idx]), batch_size, drop_last=True)
test_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_sentences[test_idx], target_labels[test_idx]), batch_size, drop_last=True)
print("Finish Batching\n")

print("Split data successful")
print(f"Train size: {len(train_dataset)} batches")
print(f"Test size: {len(test_dataset)} batches\n")

hidden = NUM_QUBITS
layer = NUM_LAYER
vocab_length = dic_length
num_layers = layer
hidden_size = 2**hidden
print("Layer = ",num_layers)
print("Hidden Size =",hidden_size)
num_class = 2
learning_rate = LEARNING_RATE

EPOCH = EPOCHS

class SimilarityModel(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size,num_layers):
        super(SimilarityModel, self).__init__()
        self.rnn = nn.RNN(vocab_size, hidden_size,num_layers=num_layers,batch_first=True) 
        
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, input, hidden):
        input = torch.nn.functional.one_hot(input, self.vocab_size).to(torch.float)
        input = input.unsqueeze(1)
        _, hidden = self.rnn(input,hidden)
        return hidden

class Classifier(torch.nn.Module):
    def __init__(self, hidden_size) -> None:
        super(Classifier,self).__init__()
        self.fc = nn.Linear(hidden_size * 2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,res):
        res = res[0]
        output = self.fc(res)
        return output


model = SimilarityModel(vocab_length,hidden_size,num_layers)
classifier = Classifier(hidden_size)
optim = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def evaluate(model:SimilarityModel, classif:Classifier, inputs):
    hidden1 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
    hidden2 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)

    input_sent1 = inputs[:,0]
    input_sent2 = inputs[:,1]

    for i in range(input_sent1.shape[1]):
        input = input_sent1[:,i]
        hidden1 = model.forward(input, hidden1)
    
    for i in range(input_sent2.shape[1]):
        input = input_sent2[:,i]
        hidden2 = model.forward(input, hidden2)

    res = torch.cat((hidden1,hidden2),dim=-1)
    results = classif.forward(res)
    return results

def calculate_accuracy(model, classif, dataset):
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence_tensor, true_output in tqdm.tqdm(dataset):
            output = evaluate(model, classif, sentence_tensor)
            _, predicted = torch.max(output, 1)
            total += true_output.size(0)
            correct += (predicted == true_output).sum().item()
    return correct / total
    


import time 
import math
def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

start = time.time()
for epoch in range(EPOCH):
    start_epoch = time.time()
    correct = 0
    total = 0
    for i, (sentence_tensor, true_output) in enumerate(tqdm.tqdm(train_dataset)):
        optim.zero_grad()
        output = evaluate(model, classifier, sentence_tensor)
        loss = criterion(output, true_output)
        loss.backward()
        optim.step()

        _, predicted = torch.max(output, 1)
        total += true_output.size(0)
        correct += (predicted == true_output).sum().item()

    # Calculate accuracy on test dataset
    train_accuracy = correct / total
    print(f'Epoch: {epoch} Train Accuracy: {train_accuracy * 100:.2f}% - time start: {timeSince(start)}')

    # Calculate accuracy on test dataset
    test_accuracy = calculate_accuracy(model,classifier, test_dataset)
    print(f'Epoch: {epoch} Test Accuracy: {test_accuracy * 100:.2f}% - time start: {timeSince(start)}')
