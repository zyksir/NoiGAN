import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
class ARGS():
    def __init__(self):
        self.cuda = True
        self.nentity = 14951
        self.nrelation = 1345
        self.countries = False
        self.test_batch_size = 16
        self.batch_size = 1024
        self.cpu_num = 8
        self.test_log_steps = 1000
        self.negative_sample_size = 256
        self.data_path = "../data/wn18"
        self.model_path = "../models/TransE_wn18_fake20_256"

class TrainDataset(Dataset):

    def __init__(self, true_triples, fake_triples, entity_embedding, relation_embedding):
        self.true_triples = true_triples
        self.fake_triples = fake_triples
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.negative_sample_size = 1

    def __len__(self):
        return len(self.true_triples)

    def __getitem__(self, idx):
        h, r, t = self.true_triples[idx]
        # negative_samples = [self.fake_triples[idx]]
        negative_samples = random.sample(self.fake_triples, self.negative_sample_size)
        positive_sample, negative_sample = [], []
        h = self.entity_embedding[h].view(1, -1)
        r = self.relation_embedding[r].view(1, -1)
        t = self.entity_embedding[t].view(1, -1)
        positive_sample = torch.cat([h, r, t], dim=0)
        for h, r, t in negative_samples:
            h = self.entity_embedding[h]
            r = self.relation_embedding[r]
            t = self.entity_embedding[t]
            negative_sample.append(torch.stack([h, r, t], dim=0))
        positive_sample = torch.FloatTensor(positive_sample)
        negative_sample = torch.stack(negative_sample, dim=0)
        return positive_sample, negative_sample

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.cat([_[1] for _ in data], dim=0)
        batch_size = negative_sample.size(0)
        negative_sample = negative_sample[:batch_size, :, :]
        return positive_sample, negative_sample


class TrainIterator(object):
    def __init__(self, dataloader):
        self.dataloader = self.one_shot_iterator(dataloader)

    def __next__(self):
        return next(self.dataloader)

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

def ComplEx(head, relation, tail, mode="single"):
    re_head, im_head = torch.chunk(head, 2, dim=1)
    re_relation, im_relation = torch.chunk(relation, 2, dim=1)
    re_tail, im_tail = torch.chunk(tail, 2, dim=1)

    if mode == 'head-batch':
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        score = re_head * re_score + im_head * im_score
    else:
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

    return score

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, output_dim=1, n_layers=1):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_dim

        self.F1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.F2 = nn.Linear(hidden_dim, 1)
        # self.F3 = nn.Linear(10, 1)

    def forward(self, positive_sample, negative_sample):
        '''
        :param positive_sample: batch_size * len * input_dim
        :param negative_sample: batch_size * len * input_dim
        :return: batch_size
        '''
        positive_score = ComplEx(positive_sample[:, 0, :], positive_sample[:, 1, :], positive_sample[:, 2, :])
        positive_score = torch.cat([positive_score, -positive_score], -1)
        # positive_score = torch.cat([positive_sample[:, 0, :], positive_sample[:, 1, :],positive_sample[:, 2, :]], -1)
        positive_score = self.F1(positive_score)
        positive_score = self.dropout(torch.tanh(positive_score))
        positive_score = self.F2(positive_score)
        # positive_score = self.F3(torch.tanh(positive_score))
        positive_score = torch.sigmoid(positive_score)

        negative_score = ComplEx(negative_sample[:, 0, :], negative_sample[:, 1, :], negative_sample[:, 2, :])
        negative_score = torch.cat([negative_score, -negative_score], -1)
        # negative_score = torch.cat([negative_sample[:, 0, :], negative_sample[:, 1, :], negative_sample[:, 2, :]], -1)
        negative_score = self.F1(negative_score)
        negative_score = self.dropout(torch.tanh(negative_score))
        negative_score = self.F2(negative_score)
        # negative_score = self.F3(torch.tanh(negative_score))
        negative_score = torch.sigmoid(negative_score)

        return positive_score, negative_score


def get_true_head_and_tail(triples):
    '''
    Build a dictionary of true triples that will
    be used to filter these true triples for negative sampling
    '''

    true_head = {}
    true_tail = {}

    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

    return true_head, true_tail

args = ARGS()
args.model_path = "../models/ComplEx_wn18_fake10"
negative_triples = pickle.load(open(os.path.join(args.data_path, "negative10.pkl"), "rb"))
fake_triples = pickle.load(open(os.path.join(args.data_path, "fake10.pkl"), "rb"))
true_all_triples = pickle.load(open(os.path.join(args.data_path, "TrueALL_triples.pkl"), "rb"))
all_triples = true_all_triples + fake_triples
true_head, true_tail = get_true_head_and_tail(all_triples)

entity_embedding = np.load(os.path.join(args.model_path, "entity_embedding.npy"))
relation_embedding = np.load(os.path.join(args.model_path, "relation_embedding.npy"))
entity_embedding = torch.from_numpy(entity_embedding)
relation_embedding = torch.from_numpy(relation_embedding)

true_all_triples_len, fake_triples_len = len(true_all_triples), len(fake_triples)

train_triples = random.sample(all_triples, len(negative_triples))
train_dataset = TrainDataset(train_triples, negative_triples, entity_embedding, relation_embedding)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=5, collate_fn=TrainDataset.collate_fn)
train_iterator = TrainIterator(train_dataloader)


model = Classifier(1000, hidden_dim=10).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs, losses = 1000, []
print_loss = 0
for i in range(epochs):
    if i == 2000:
         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # elif i == 12000:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    positive_sample, negative_sample = next(train_iterator)
    positive_score, negative_score = model(positive_sample.cuda(), negative_sample.cuda())
    target = torch.cat([torch.ones(positive_score.size()[0]), torch.zeros(negative_score.size()[0])]).cuda()
    loss = F.binary_cross_entropy(torch.cat([positive_score.view(-1), negative_score.view(-1)]), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print_loss += loss.item()
    if i % 100 == 0:
        print("loss in epoch %d: %f" % (i, print_loss/100))
        print("positive score : %f" % positive_score.mean().item())
        print("negative score : %f" % negative_score.mean().item())
        print_loss = 0

pos, neg = [], []
for h, r, t in tqdm(all_triples):
    h = entity_embedding[h].view(1, -1)
    r = relation_embedding[r].view(1, -1)
    t = entity_embedding[t].view(1, -1)
    pos.append(torch.cat([h, r, t], dim=0))

for h, r, t in fake_triples:
    h = entity_embedding[h].view(1, -1)
    r = relation_embedding[r].view(1, -1)
    t = entity_embedding[t].view(1, -1)
    neg.append(torch.cat([h, r, t], dim=0))

neg = torch.stack(neg, dim=0).cuda()
all_score1, all_score2, i, batch_size = 0, 0, 0, len(fake_triples)
while i < len(all_triples):
    j = min(i + batch_size, len(all_triples))
    pos_ = torch.stack(pos[i: j]).cuda()
    pos_score, neg_score = model(pos_, neg)
    all_score1 += (pos_score>0.5).sum().item()
    all_score2 += (pos_score).sum().item()
    i = j
print((neg_score>0.5).sum().item()/all_score1)
print((neg_score).sum().item()/all_score2)

