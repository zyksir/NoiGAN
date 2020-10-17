import os
import sys
import numpy as np
from tqdm import tqdm
import math
import pickle
from IPython import embed
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import defaultdict
import heapq
class TopKHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)

    def topk(self):
        return [heapq.heappop(self.data)[1] for _ in range(len(self.data))]


class ARGS(object):
    def __init__(self):
        self.model = "RotatE"
        self.train_file = "NS100"

        self.cuda = True
        self.countries = False
        self.test_batch_size = 16
        self.batch_size = 1024
        self.cpu_num = 8
        self.test_log_steps = 1000
        self.negative_sample_size = 256
        self.hidden_dim = 200


class ClassifierDataset(Dataset):
    def __init__(self, positive_triples, negative_triples, entity_embedding, relation_embedding):
        self.positive_triples = positive_triples
        self.negative_triples = negative_triples
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.negative_sample_size = 1

    def __len__(self):
        return len(self.positive_triples)

    def __getitem__(self, idx):
        # choice one triple, generate tensor of (1, 3)
        positive_triple = torch.LongTensor(self.positive_triples[idx])
        negative_triple = torch.LongTensor(self.negative_triples[idx])
        return positive_triple, negative_triple

    @staticmethod
    def collate_fn(data):
        # get tensor of (batch_size, 3)
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        return positive_sample, negative_sample


class TrainIterator(object):
    def __init__(self, dataloader):
        self.dataloader = self.one_shot_iterator(dataloader)

    def __next__(self):
        return next(self.dataloader)

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            random.shuffle(dataloader.dataset.negative_triples)
            for data in dataloader:
                yield data


class Classifier(nn.Module):
    def __init__(self, args, input_dim, hidden_dim=10, output_dim=1, entity_embedding=None, relation_embedding=None):
        super(Classifier, self).__init__()
        self.model = args.model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_dim
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.embedding_range = 11 / 200

        self.F1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.F2 = nn.Linear(hidden_dim, 1)
        # self.F3 = nn.Linear(10, 1)

    def TransE(self, head, relation, tail):
        score = head + (relation - tail)
        return score

    def DistMult(self, head, relation, tail):
        # score = head * (relation * tail)
        # return score
        return head * tail

    def ComplEx(self, head, relation, tail):
        re_head, im_head = torch.chunk(head, 2, dim=1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=1)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        score = re_head * re_score + im_head * im_score

        return score

    def RotatE(self, head, relation, tail):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=1)  # batch_size, dim

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation / (self.embedding_range / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head

        score = torch.stack([re_score, im_score], dim=0)  # 2, batch_size, dim
        score = score.norm(dim=0)  # batch_size, dim

        return score

    def get_score(self, sample):
        head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0])
        relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1])
        tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2])
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE
        }

        if self.model in model_func:
            score = model_func[self.model](head, relation, tail)
        else:
            raise ValueError('model %s not supported' % self.args.model)
        return score

    def forward(self, sample):
        '''
        :param sample: (batch_size, 3)
        :return: batch_size
        '''
        score = self.get_score(sample)
        score = self.F1(score)
        score = self.dropout(torch.tanh(score))
        score = self.F2(score)
        score = torch.sigmoid(score)
        return score


def find_threshold(all_score):
    all_score.sort()
    delta = all_score[100:] - all_score[:-100]
    threshold_delta = math.sqrt(delta[0])
    for i, score in enumerate(all_score):
        if delta[i] > threshold_delta:
            return score
    return all_score.mean() - 2 * all_score.var()

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


def quickselect(start, end, A, k):
    if start == end:
        return A[start]

    mid = partition(start, end, A)

    if mid == k:
        return A[k]
    elif mid > k:
        return quickselect(start, mid - 1, A, k)
    else:
        return quickselect(mid + 1, end, A, k)


def partition(start, end, A):
    pivotIndex = random.randrange(start, end + 1)
    pivot = A[pivotIndex]
    A[end], A[pivotIndex] = A[pivotIndex], A[end]
    mid = start
    for i in range(start, end):
        if A[i] > pivot:
            A[mid], A[i] = A[i], A[mid]
            mid += 1
    A[mid], A[end] = A[end], A[mid]
    return mid

def find_positive_triples(train_triples, classifier, k):
    global args
    # return random.sample(train_triples, len(train_triples) // 10)
    # topk_heap = TopKHeap(k)
    triple_score = []
    i = 0
    while i < len(train_triples):
        sys.stdout.write("%d in %d\r" % (i, len(train_triples)))
        sys.stdout.flush()
        j = min(i + 2048, len(train_triples))
        sample = torch.LongTensor(train_triples[i: j]).cuda()
        if args.model in ["ComplEx", "DistMult"]:
            score = classifier.get_score(sample).sum(dim=1)
        else:
            score = -classifier.get_score(sample).sum(dim=1)
        for x, triple in enumerate(train_triples[i: j]):
            triple_score.append((score[x].item(), triple))
        i = j
    random.shuffle(triple_score)
    quickselect(0, len(triple_score)-1, triple_score, k)
    topk_triple_score = triple_score[:k]
    return [triple for score, triple in topk_triple_score]
    # return topk_heap.topk()
    # return random.sample(train_triples, len(train_triples) // 10)

def find_negative_triples(train_triples, args):
    true_head, true_tail, true_relation = \
        defaultdict(lambda: set()), defaultdict(lambda: set()), defaultdict(lambda: set())
    relation2tail, relation2head = defaultdict(lambda: set()), defaultdict(lambda: set())
    for h, r, t in train_triples:
        true_head[(r, t)].add(h)
        true_tail[(h, r)].add(t)
        true_relation[(h, t)].add(r)
        relation2tail[r].add(t)
        relation2head[r].add(h)
    all_entities, all_relations = set(range(args.nentity)), set(range(args.nrelation))
    negative_triples = set()
    triples_for_generate = random.sample(train_triples, len(train_triples) // 10 + 1000)
    for triple in tqdm(triples_for_generate):
        h, r, t = triple
        if random.uniform(0, 1) < 0.5:
            t_ = random.choice(list(all_entities - true_tail[(h, r)]))
            negative_triples.add((h, r, t_))
        else:
            h_ = random.choice(list(all_entities - true_head[(r, t)]))
            negative_triples.add((h_, r, t))
    return list(negative_triples)

def draw_pic(true_score, fake_score, output_file):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5), dpi=80)
    ax1 = plt.subplot(121)
    true_score.sort()
    x = list(range(true_score.size))
    plt.title("true")
    plt.xlabel('score')  # make axis labels
    plt.ylabel('num')
    plt.plot(true_score, x)
    ax2 = plt.subplot(122)
    fake_score.sort()
    x = list(range(fake_score.size))
    plt.title("fake")
    plt.xlabel('score')  # make axis labels
    plt.ylabel('num')
    plt.plot(fake_score, x)
    plt.show()
    plt.savefig(output_file, format="eps")
    print("save pic to %s" % output_file)

args = ARGS()
for args.dataset in ["wn18rr"]:     # , "FB15k-237"
    for args.train_file in ["NS100", "NS10"]:
        args.data_path = "./data/%s" % args.dataset
        for args.model in ["RotatE"]:
            args.model_path = "./models/%s_%s_%s" % (args.model, args.dataset, args.train_file)
            print("when dataset is %s, model is %s" % (args.dataset, args.model))
            with open(os.path.join(args.data_path, 'entities.dict')) as fin:
                entity2id = dict()
                for line in fin:
                    eid, entity = line.strip().split('\t')
                    entity2id[entity] = int(eid)
                args.nentity = len(entity2id)
            with open(os.path.join(args.data_path, 'relations.dict')) as fin:
                relation2id = dict()
                for line in fin:
                    rid, relation = line.strip().split('\t')
                    relation2id[relation] = int(rid)
                args.nrelation = len(relation2id)

            true_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
            train_triples = read_triple(os.path.join(args.data_path, args.train_file), entity2id, relation2id)
            fake_triples = list(set(train_triples) - set(true_triples))

            entity_embedding = np.load(os.path.join(args.model_path, "entity_embedding.npy"))
            relation_embedding = np.load(os.path.join(args.model_path, "relation_embedding.npy"))
            entity_embedding = torch.from_numpy(entity_embedding).cuda()
            relation_embedding = torch.from_numpy(relation_embedding).cuda()
            print(entity_embedding.size(), relation_embedding.size())

            k = len(train_triples) // 10
            model = Classifier(args, args.hidden_dim, hidden_dim=5, entity_embedding=entity_embedding,
                               relation_embedding=relation_embedding)
            if args.dataset == "wn18rr":
                model.embedding_range = 8 / 200 * 4
            else:
                model.embedding_range = 11 / 200
            model = model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            positive_triples = find_positive_triples(train_triples, model, k=k)
            negative_triples = find_negative_triples(train_triples, args)
            train_dataset = ClassifierDataset(positive_triples, negative_triples, entity_embedding, relation_embedding)
            train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=5, collate_fn=ClassifierDataset.collate_fn)
            train_iterator = TrainIterator(train_dataloader)

            # print("start to train")
            epochs, losses = 1500, []
            print_loss = 0
            for i in range(epochs):
                if i == 2000:
                     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                # elif i == 12000:
                #     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                positive_sample, negative_sample = next(train_iterator)
                positive_score = model(positive_sample.cuda())
                negative_score = model(negative_sample.cuda())
                # positive_score, negative_score = model(positive_sample.cuda(), negative_sample.cuda())
                target = torch.cat([torch.ones(positive_score.size()[0]), torch.zeros(negative_score.size()[0])]).cuda()
                loss = F.binary_cross_entropy(torch.cat([positive_score.view(-1), negative_score.view(-1)]), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print_loss += loss.item()
                if i % 500 == 0:
                    print("loss in epoch %d: %f" % (i, print_loss/100))
                    print("positive score : %f" % positive_score.mean().item())
                    print("negative score : %f" % negative_score.mean().item())
                    print_loss = 0

            # print("start to evaluate")
            true_score = []
            i = 0
            # true_triples = list(set(true_triples) - set(positive_triples))
            while i < len(true_triples):
                j = min(i + 1024, len(true_triples))
                true_sample = torch.LongTensor(true_triples[i: j]).cuda()
                # true_score.extend(model.get_score(true_sample).sum(dim=1).view(-1).tolist())
                true_score.extend(model(true_sample).view(-1).tolist())
                i = j

            fake_score = []
            i = 0
            # fake_triples = list(set(fake_triples) - set(positive_triples))
            while i < len(fake_triples):
                j = min(i + 1024, len(fake_triples))
                fake_sample = torch.LongTensor(fake_triples[i: j]).cuda()
                # fake_score.extend(model.get_score(fake_sample).sum(dim=1).view(-1).tolist())
                fake_score.extend(model(fake_sample).view(-1).tolist())
                i = j
            true_score = np.array(true_score)
            fake_score = np.array(fake_score)
            all_score = np.concatenate((true_score, fake_score)).reshape(-1)
            threshold = find_threshold(all_score)
            true_left, fake_left = (true_score > threshold).sum(), (fake_score > threshold).sum()
            percent = true_left / (true_left + fake_left)
            draw_pic(true_score, fake_score, args.model_path + "/" + args.model_path.split("/")[-1] + ".eps")
            print("%s:, true.mean is %f, fake.mean is %f" % (args.model_path, true_score.mean(), fake_score.mean()))
            print("%s: %f, true is %f, fake is %f" % (args.model_path, percent, true_left, fake_left))
