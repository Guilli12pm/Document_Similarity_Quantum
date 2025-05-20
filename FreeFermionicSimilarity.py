
import pandas as pd
import os
import sys

import torch
import torch.utils.data
from torch import nn
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import FFQuantumDevice
from cleanMethod import cleanText

_ = torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 200
NUM_LAYER = 2
NUM_QUBITS = 4
EPOCHS = 10
LEARNING_RATE = 0.005

data_dir = "./data_Quora/questions.csv"

data = pd.read_csv(data_dir)

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
    for i, word in enumerate(split_sent):
        tensor[i] = word2index(word)
    return tensor

tensor_sentences = []
target_labels = []

PADDING = 15

print(f"Building the tensors of padding {PADDING}")
for index, row in data.iterrows():
    tensor_c1 = sentence2tensor(row['question1'], PADDING)
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

num_layer = NUM_LAYER
num_qubits = NUM_QUBITS
print("Layer =",num_layer)
print("Qubits =",num_qubits)

num_angles = num_layer * (2 * num_qubits - 1)
learning_rate = LEARNING_RATE
number_words = dic_length
num_class = 2

EPOCH = EPOCHS


class QRNN(torch.nn.Module):
    def __init__(self, input_size, num_angles):
        super(QRNN, self).__init__()
        self.input_size = input_size
        self.L = torch.nn.Linear(self.input_size, num_angles)

    def forward(self, input, circuit:FFQuantumDevice):
        input = torch.nn.functional.one_hot(input, self.input_size).to(torch.float)
        angles = self.L(input)

        ang = 0
        for _ in range(num_layer):
            
            circuit.rxx_layer(angles[:,ang:ang+num_qubits-1])
            ang += num_qubits - 1
            circuit.rz_layer(angles[:,ang:ang+num_qubits])
            ang += num_qubits

        return circuit

class Measure(torch.nn.Module):
    def __init__(self, num_qubits, num_class):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_class = num_class

    def forward(self, circuit:FFQuantumDevice):
        meas = circuit.z_exp_all() 
        return meas

class Dist(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, x1, x2):
        return self.cos(x1,x2)

class Classifier(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.distance = Dist()
        self.output_size = output_size
        self.fc = torch.nn.Linear(1, output_size)

    def forward(self, meas1, meas2):
        dist = self.distance(meas1, meas2).view(batch_size, 1)
        output = self.fc(dist)
        return output


model = QRNN(number_words, num_angles)
measure = Measure(num_qubits, num_class)
classifier = Classifier(num_class)
optim = torch.optim.Adam(list(model.parameters()) + list(measure.parameters()) + list(classifier.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def evaluate(model, classifier, inputs):
    hidden1 = FFQuantumDevice(num_qubits, batch_size, device=device)
    hidden2 = FFQuantumDevice(num_qubits, batch_size, device=device)

    input_sent1 = inputs[:,0]
    input_sent2 = inputs[:,1]

    for i in range(input_sent1.shape[1]):
        input = input_sent1[:,i]
        hidden1 = model.forward(input, hidden1)
    for i in range(input_sent2.shape[1]):
        input = input_sent2[:,i]
        hidden2 = model.forward(input, hidden2)

    output1 = measure.forward(hidden1)
    output2 = measure.forward(hidden2)
    
    res = classifier.forward(output1, output2)
    return res

def calculate_accuracy(model, classifier, dataset):
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence_tensor, true_output in tqdm.tqdm(dataset):
            output = evaluate(model, classifier, sentence_tensor)
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
    for i, (sentences_tensor, true_output) in enumerate(tqdm.tqdm(train_dataset)):
        optim.zero_grad()
        output = evaluate(model, classifier, sentences_tensor)
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
    test_accuracy = calculate_accuracy(model, classifier, test_dataset)
    print(f'Epoch: {epoch} Test Accuracy: {test_accuracy * 100:.2f}% - time start: {timeSince(start)}')