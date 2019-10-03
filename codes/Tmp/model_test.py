from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IPython import embed
import argparse
import tqdm
import json
import logging
import os
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch


None_relation = "NONE_RELATION"
torch.set_num_threads(8)

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset, NewTrainDataset
from dataloader import BidirectionalOneShotIterator

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--tetrad', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None,
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('--using_context', action='store_true')
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

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

def read_tetrad(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    tetrad = []
    with open(file_path) as fin:
        for line in fin:
            h, r1, r2, t = line.strip().split('\t')
            tetrad.append((entity2id[h], relation2id[r1], relation2id[r2], entity2id[t]))
    return tetrad

args = parse_args()
args.data_path = "../data/FB15k"
args.hidden_dim = 1000
args.init_checkpoint = "../models/TransE_FB15k_softmax"
args.tetrad = False

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
print("relation2id and entity2id done")
relation2id[None_relation] = len(relation2id)
nentity = len(entity2id)
nrelation = len(relation2id)
args.nentity = nentity
args.nrelation = nrelation

train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
print("triple done")

# from dataloader import preprocess
# preprocess(train_triples, entity2id, relation2id, args.negative_sample_size,
#            "head-batch", args.data_path, batch_size=64)
#
# preprocess(train_triples, entity2id, relation2id, args.negative_sample_size,
#            "tail-batch", args.data_path, batch_size=64)
#

train_dataloader_head = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TrainDataset.collate_fn
)

train_dataloader_tail = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TrainDataset.collate_fn
)
train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
    )
current_learning_rate = args.learning_rate
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, kge_model.parameters()),
    lr=current_learning_rate
)
# kge_model.train_step(kge_model, optimizer, train_iterator, args)
model_path = "../models/TransE_FB15k_fake"
# model_path = "../models/TransE_FB15k_baseline"
checkpoint = torch.load(os.path.join(model_path, 'checkpoint'))
init_step = checkpoint['step']
kge_model.load_state_dict(checkpoint['model_state_dict'])
model = kge_model
model.cuda()
model.eval()

optimizer.zero_grad()


positive_scores = []
negative_scores = []
for positive_sample, negative_sample, subsampling_weight, mode in train_dataloader_head:
    positive_score = model(positive_sample.cuda())
    positive_scores.extend(positive_score.squeeze(dim=1).tolist())
positive_score = torch.FloatTensor(positive_scores)


import pickle
train_triples = pickle.load(open("../data/FB15k/fake.pkl", "rb"))
train_dataloader_head = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TrainDataset.collate_fn
)

train_dataloader_tail = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TrainDataset.collate_fn
)
train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
positive_scores = []
negative_scores = []
for positive_sample, negative_sample, subsampling_weight, mode in train_dataloader_head:
    positive_score = model(positive_sample.cuda())
    positive_scores.extend(positive_score.squeeze(dim=1).tolist())
positive_score = torch.FloatTensor(positive_scores)

test_dataloader_head = DataLoader(
    TestDataset(
        test_triples,
        all_true_triples,
        args.nentity,
        args.nrelation,
        'head-batch'
    ),
    batch_size=args.test_batch_size,
    num_workers=max(1, args.cpu_num // 2),
    collate_fn=TestDataset.collate_fn
)

test_dataloader_tail = DataLoader(
    TestDataset(
        test_triples,
        all_true_triples,
        args.nentity,
        args.nrelation,
        'tail-batch'
    ),
    batch_size=args.test_batch_size,
    num_workers=max(1, args.cpu_num // 2),
    collate_fn=TestDataset.collate_fn
)

test_dataset_list = [test_dataloader_head, test_dataloader_tail]

logs = []

step = 0
total_steps = sum([len(dataset) for dataset in test_dataset_list])
for test_dataset in test_dataset_list:
    for _, batch in tqdm(enumerate(test_dataset), total=total_steps//2):
        positive_sample, negative_sample, filter_bias, mode = batch
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            filter_bias = filter_bias.cuda()
        batch_size = positive_sample.size(0)

        score = model((positive_sample, negative_sample), mode)
        score += filter_bias

        # Explicitly sort all the entities to ensure that there is no test exposure bias
        argsort = torch.argsort(score, dim=1, descending=True)
        if mode == 'head-batch':
            positive_arg = positive_sample[:, 0]
        elif mode == 'tail-batch':
            positive_arg = positive_sample[:, 2]
        for i in range(batch_size):
            # Notice that argsort is not ranking
            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
            assert ranking.size(0) == 1

            # ranking + 1 is the true ranking used in evaluation metrics
            ranking = 1 + ranking.item()
            logs.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                'HITS@10': 1.0 if ranking <= 10 else 0.0,
            })

        step += 1
metrics = {}
for metric in logs[0].keys():
    metrics[metric] = sum([log[metric] for log in logs])/len(logs)
print(metrics)


##############################################################################
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import KGEModel, NewKGEModel
from dataloader import TrainDataset, NewTrain
from dataloader import BidirectionalOneShotIterator
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
        self.data_path = "../data/FB15k"
        self.model_path = "../models/TransE_FB15k_DEBUG2"
args = ARGS()

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

train_triples = pickle.load(open(os.path.join(args.data_path, "TrueAndFake_triples.pkl"), "rb"))
TrueSome_triples = pickle.load(open(os.path.join(args.data_path, "TrueSome_triples.pkl"), "rb"))
fake_triples = pickle.load(open(os.path.join(args.data_path, "fake.pkl"), "rb"))
TrueAll_triples = pickle.load(open(os.path.join(args.data_path, "TrueALL_triples.pkl"), "rb"))

train_head = NewTrain(train_triples, args.nentity, args.nrelation, args.negative_sample_size, 'head-batch', TrueSome_triples)
train_tail = NewTrain(train_triples, args.nentity, args.nrelation, args.negative_sample_size, 'tail-batch', TrueSome_triples)

train_dataloader_head = DataLoader(train_head, batch_size=args.batch_size, shuffle=True, num_workers=max(1, args.cpu_num // 2), collate_fn=NewTrain.collate_fn)
train_dataloader_tail = DataLoader(train_tail, batch_size=args.batch_size, shuffle=True, num_workers=max(1, args.cpu_num // 2), collate_fn=NewTrain.collate_fn)

train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
# next(train_iterator)


# def read_triple(file_path, entity2id, relation2id):
#     '''
#     Read triples and map them into ids.
#     '''
#     triples = []
#     with open(file_path) as fin:
#         for line in fin:
#             h, r, t = line.strip().split('\t')
#             triples.append((entity2id[h], relation2id[r], entity2id[t]))
#     return triples
#
#
# train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
# train_dataloader_head = DataLoader(
#     TrainDataset(train_triples, args.nentity, args.nrelation, args.negative_sample_size, 'head-batch'),
#     batch_size=args.batch_size,
#     shuffle=True,
#     num_workers=max(1, args.cpu_num // 2),
#     collate_fn=TrainDataset.collate_fn
# )
#
# train_dataloader_tail = DataLoader(
#     TrainDataset(train_triples, args.nentity, args.nrelation, args.negative_sample_size, 'tail-batch'),
#     batch_size=args.batch_size,
#     shuffle=True,
#     num_workers=max(1, args.cpu_num // 2),
#     collate_fn=TrainDataset.collate_fn
# )
# train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
# train_iterator = NewTrain(os.path.join(args.data_path, "DEBUG.pt"))
model_path = "../models/TransE_FB15k_new_fake001"
checkpoint = torch.load(os.path.join(model_path, 'checkpoint'))
entity_embedding = np.load(os.path.join(model_path, 'entity_embedding.npy'))
relation_embedding = np.load(os.path.join(model_path, 'relation_embedding.npy'))
kge_model = NewKGEModel(
        model_name="TransE",
        nentity=14951,
        nrelation=1345,
        hidden_dim=1000,
        gamma=24,
        double_entity_embedding=False,
        double_relation_embedding=False,
        chidden_dim=1000
)
kge_model.entity_embedding = nn.Parameter(torch.from_numpy(entity_embedding))
kge_model.relation_embedding = nn.Parameter(torch.from_numpy(relation_embedding))
# kge_model = KGEModel(
#         model_name="DistMult",
#         nentity=14951,
#         nrelation=1345,
#         hidden_dim=2000,
#         gamma=24,
#         double_entity_embedding=False,
#         double_relation_embedding=False
# )
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, kge_model.parameters()),
    lr=0.0001
)
model = kge_model.cuda()
# kge_model.load_state_dict(checkpoint['model_state_dict'])
# kge_model.F1 = nn.Linear(1000, 10)
# kge_model.F2 = nn.Linear(10, 1)
# checkpoint['model_state_dict'] = kge_model.state_dict()
# torch.save(checkpoint, os.path.join(model_path, 'checkpoint'))
all_triples = train_triples + TrueSome_triples
top_triples, i = [], 0
while i < len(all_triples):
    j = min(i+1024, len(all_triples))
    score = model(torch.LongTensor(all_triples[i:j]).cuda()).view(-1).tolist()
    top_triples.extend(list(zip(all_triples[i:j], score)))
    i = j
top_triples = sorted(top_triples, key=lambda x:x[1], reverse=True)
fake_triples = set(fake_triples)
top_triples = top_triples[:len(all_triples)//100]
count = 0
for triple, score in top_triples:
    if triple in fake_triples:
        count += 1

for i in range(10):
    model.train()
    optimizer.zero_grad()
    TrueAndFake_pos, TrueAndFake_neg, _, mode, TrueSome_pos, TrueSome_neg = next(train_iterator)
    TrueAndFake_pos, TrueAndFake_neg = TrueAndFake_pos.cuda(), TrueAndFake_neg.cuda()
    TrueSome_pos, TrueSome_neg = TrueSome_pos.cuda(), TrueSome_neg.cuda()
    negative_score = model((TrueAndFake_pos, TrueAndFake_neg), mode=mode)
    negative_score = (F.softmax(negative_score, dim=1).detach() * F.logsigmoid(-negative_score)).sum(dim=1)
    positive_score = model(TrueAndFake_pos)
    positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
    subsampling_weight = torch.sigmoid(model.calculate_weight(TrueAndFake_pos).squeeze(dim=1))
    positive_sample_loss = - (subsampling_weight * positive_score).mean()
    negative_sample_loss = - negative_score.mean()
    loss = (positive_sample_loss + negative_sample_loss) / 2
    loss.backward()
    optimizer.step()
    if (i+1) % 100 == 0:
        print("in %d epoch, loss is %f"%(i+1, loss.item()))


train_iterator = NewTrain(os.path.join(args.data_path, "DEBUG.pt"))
model_path = "../models/TransE_FB15k_DEBUG2"
checkpoint = torch.load(os.path.join(model_path, 'checkpoint'))
kge_model = NewKGEModel(
        model_name="TransE",
        nentity=14951,
        nrelation=1345,
        hidden_dim=1000,
        gamma=24,
        double_entity_embedding=False,
        double_relation_embedding=False,
        chidden=1000
)
# kge_model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, kge_model.parameters()),
    lr=0.0001
)
kge_model.load_state_dict(checkpoint['model_state_dict'])
for i in range(5000):
    model = kge_model.cuda()
    model.train()
    optimizer.zero_grad()
    TrueAndFake_pos, TrueAndFake_neg, _, mode = next(train_iterator)
    TrueAndFake_pos, TrueAndFake_neg = TrueAndFake_pos.cuda(), TrueAndFake_neg.cuda()
    negative_score = model((TrueAndFake_pos, TrueAndFake_neg), mode=mode)
    negative_score = (F.softmax(negative_score, dim=1).detach() * F.logsigmoid(-negative_score)).sum(dim=1)
    positive_score = model(TrueAndFake_pos)
    positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
    positive_sample_loss = - positive_score.mean()
    negative_sample_loss = - negative_score.mean()
    loss = (positive_sample_loss + negative_sample_loss) / 2
    loss.backward()
    optimizer.step()
    if (i+1) % 100 == 0:
        print("in %d epoch, loss is %f"%(i+1, loss.item()))


#######################
