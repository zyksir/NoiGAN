######################################
# generate fake data
import random
import torch
import tqdm
import pickle
import os
import numpy as np
from collections import defaultdict


def generate_fake_data(data_path="../data/FB15k/", num=10):
    def read_triple(file_path, entity2id, relation2id):
        '''
        Read triples and map them into ids.
        '''
        triples = []
        with open(file_path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                try:
                    triples.append((entity2id[h], relation2id[r], entity2id[t]))
                except:
                    pass
        return triples


    with open(os.path.join(data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    nrelation = len(relation2id)
    nentity = len(entity2id)
    train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    # fake_triples = pickle.load(open(os.path.join(data_path, "fake%d.pkl"%num), "rb"))
    # train_triples += fake_triples
    all_triples = train_triples + valid_triples + test_triples
    true_head, true_tail, true_relation = defaultdict(lambda: set()), defaultdict(lambda: set()), defaultdict(lambda: set())
    relation2tail, relation2head = defaultdict(lambda: set()), defaultdict(lambda: set())
    for h, r, t in all_triples:
        true_head[(r, t)].add(h)
        true_tail[(h, r)].add(t)
        true_relation[(h, t)].add(r)
        relation2tail[r].add(t)
        relation2head[r].add(h)

    fake_triples = set()
    while len(fake_triples)<len(train_triples) * num // 100:
        h, r, t = random.choice(all_triples)
        t_ = random.choice(list(set(entity2id.values())-true_tail[(h, r)]))
        h_ = random.choice(list(set(entity2id.values())-true_head[(r, t)]))
        fake_triples.add((h_, r, t))
        fake_triples.add((h, r, t_))
        # try:
        #     t_ = random.choice(list(relation2tail[r] - true_tail[(h, r)]))
        #     fake_triples.add((h, r, t_))
        # except:
        #     pass
        # try:
        #     h_ = random.choice(list(relation2head[r] - true_head[(r, t)]))
        #     fake_triples.add((h_, r, t))
        # except:
        #     pass

    fake_triples = list(fake_triples)
    with open(os.path.join(data_path, "fake%d.pkl" % num), "wb") as f:
        pickle.dump(fake_triples, f)

# generate_fake_data(data_path="../data/FB15k/", num=10)
# generate_fake_data(data_path="../data/FB15k/", num=20)
# generate_fake_data(data_path="../data/FB15k/", num=40)
# generate_fake_data(data_path="../data/wn18/", num=10)
# generate_fake_data(data_path="../data/wn18/", num=20)
# generate_fake_data(data_path="../data/wn18/", num=40)
# generate_fake_data(data_path="../data/FB15k-237/", num=10)
# generate_fake_data(data_path="../data/FB15k-237/", num=20)
# generate_fake_data(data_path="../data/FB15k-237/", num=40)
# generate_fake_data(data_path="../data/NELL27k/", num=10)
# generate_fake_data(data_path="../data/NELL27k/", num=20)
# generate_fake_data(data_path="../data/NELL27k/", num=40)
# generate_fake_data(data_path="../data/YAGO3-10/", num=10)
generate_fake_data(data_path="../data/YAGO3-10/", num=20)
generate_fake_data(data_path="../data/YAGO3-10/", num=40)