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
        self.nentity = 123182
        self.nrelation = 37
        self.countries = False
        self.test_batch_size = 16
        self.batch_size = 1024
        self.cpu_num = 8
        self.test_log_steps = 1000
        self.negative_sample_size = 256
        self.hidden_dim = 250
        self.data_path = "../data/YAGO3-10"

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
        r_ = r.repeat(1, 2)
        t = self.entity_embedding[t].view(1, -1)
        positive_sample = torch.cat([h, r_, t], dim=0)
        for h, r, t in negative_samples:
            h = self.entity_embedding[h]
            r = self.relation_embedding[r]
            r_ = r.repeat(2)
            t = self.entity_embedding[t]
            negative_sample.append(torch.stack([h, r_, t], dim=0))
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

def RotatE(head, relation, tail, mode="simple"):
    pi = 3.14159265358979323846

    re_head, im_head = torch.chunk(head, 2, dim=1)
    re_tail, im_tail = torch.chunk(tail, 2, dim=1)

    # Make phases of relations uniformly distributed in [-pi, pi]
    embed_model = (24+2) / args.hidden_dim
    phase_relation = relation / (embed_model / pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    if mode == 'head-batch':
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head
    else:
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

    score = torch.stack([re_score, im_score], dim=0)
    score = score.norm(dim=0)

    # score = embed_model.gamma.item() - score.sum(dim=2)
    return score

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

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
        positive_score = RotatE(positive_sample[:, 0, :], positive_sample[:, 1, :args.hidden_dim], positive_sample[:, 2, :])
        positive_score = torch.cat([positive_score, -positive_score], -1)
        positive_score = self.F1(positive_score)
        positive_score = self.dropout(torch.tanh(positive_score))
        positive_score = self.F2(positive_score)
        # positive_score = self.F3(torch.tanh(positive_score))
        positive_score = torch.sigmoid(positive_score)

        negative_score = RotatE(negative_sample[:, 0, :], negative_sample[:, 1, :args.hidden_dim], negative_sample[:, 2, :])
        negative_score = torch.cat([negative_score, -negative_score], -1)
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
args.model_path = "../models/RotatE_YAGO3-10_fake40_5"
negative_triples = pickle.load(open(os.path.join(args.data_path, "fakePath70.pkl"), "rb"))
fake_triples = pickle.load(open(os.path.join(args.data_path, "fake40.pkl"), "rb"))
negative_triples = list(set(negative_triples) - set(fake_triples))
with open(os.path.join(args.data_path, 'entities.dict')) as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)
with open(os.path.join(args.data_path, 'relations.dict')) as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)
true_all_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
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


model = Classifier(args.hidden_dim * 2, hidden_dim=2).cuda()
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
    r = r.repeat(1, 2)
    t = entity_embedding[t].view(1, -1)
    pos.append(torch.cat([h, r, t], dim=0))

for h, r, t in fake_triples:
    h = entity_embedding[h].view(1, -1)
    r = relation_embedding[r].view(1, -1)
    r = r.repeat(1, 2)
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

