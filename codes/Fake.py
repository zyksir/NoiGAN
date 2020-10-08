#!/usr/bin/python
# -*- coding:utf8 -*-
######################################
# 生成噪声
# 目前只考虑generate_fake_data
import random
import torch
import tqdm
import pickle
import os
import numpy as np
import sys
from collections import defaultdict

# 随机替换头和尾
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

    train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)

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
    while len(fake_triples) < len(train_triples) * num // 100:
        sys.stdout.write("%d in %d\r" % (len(fake_triples), len(train_triples) * num // 100))
        sys.stdout.flush()
        h, r, t = random.choice(all_triples)
        t_ = random.choice(list(set(entity2id.values())-true_tail[(h, r)]))
        h_ = random.choice(list(set(entity2id.values())-true_head[(r, t)]))
        fake_triples.add((h_, r, t))
        fake_triples.add((h, r, t_))

    fake_triples = list(fake_triples)
    with open(os.path.join(data_path, "fake%d.pkl" % num), "wb") as f:
        pickle.dump(fake_triples, f)
    print("finish generate %d percent fake data for %s" % (num, data_path))


# generate_fake_data(data_path="../data/wn18rr/", num=40)
generate_fake_data(data_path="../data/wn18rr/", num=70)
generate_fake_data(data_path="../data/wn18rr/", num=100)
# generate_fake_data(data_path="../data/FB15k-237/", num=40)
generate_fake_data(data_path="../data/FB15k-237/", num=70)
generate_fake_data(data_path="../data/FB15k-237/", num=100)
# generate_fake_data(data_path="../data/wn18/", num=40)
generate_fake_data(data_path="../data/wn18/", num=70)
generate_fake_data(data_path="../data/wn18/", num=100)
# generate_fake_data(data_path="../data/FB15k/", num=40)
generate_fake_data(data_path="../data/FB15k/", num=70)
generate_fake_data(data_path="../data/FB15k/", num=100)
generate_fake_data(data_path="../data/YAGO3-10/", num=40)
generate_fake_data(data_path="../data/YAGO3-10/", num=70)
generate_fake_data(data_path="../data/YAGO3-10/", num=100)

def generate_fakePath_data(data_path="../data/FB15k/", num=10):
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

    all_triples = set(train_triples + valid_triples + test_triples)
    true_head, true_tail, true_relation = defaultdict(lambda: set()), defaultdict(lambda: set()), defaultdict(lambda: set())
    relation2tail, relation2head, head2tail = defaultdict(lambda: set()), defaultdict(lambda: set()), defaultdict(lambda: set())
    for h, r, t in all_triples:
        true_head[(r, t)].add(h)
        true_tail[(h, r)].add(t)
        true_relation[(h, t)].add(r)
        relation2tail[r].add(t)
        relation2head[r].add(h)
        head2tail[h].add(t)

    fake_triples = set()
    all_relations = list(range(nrelation))
    all_triples_ = list(all_triples)
    while len(fake_triples)<len(train_triples) * num // 100:
        sys.stdout.write("%d in %d\r" % (len(fake_triples), len(train_triples) * num // 100))
        sys.stdout.flush()
        h, r, t = random.choice(all_triples_)
        if len(head2tail[t]) == 0:
            continue
        t_ = random.choice(list(head2tail[t]))
        r_ = random.choice(all_relations)
        if (h, r_, t_) not in all_triples:
            fake_triples.add((h, r_, t_))

    fake_triples = list(fake_triples)
    with open(os.path.join(data_path, "fakePath%d.pkl" % num), "wb") as f:
        pickle.dump(fake_triples, f)

# generate_fakePath_data(data_path='../data/FB15k-237', num=40)
# generate_fakePath_data(data_path="../data/wn18rr", num=30)
# generate_fakePath_data(data_path="../data/wn18rr", num=50)
# generate_fakePath_data(data_path="../data/wn18rr", num=70)
# generate_fakePath_data(data_path="../data/wn18rr", num=100)

# generate_fakePath_data(data_path='../data/FB15k-237', num=30)
# generate_fakePath_data(data_path='../data/FB15k-237', num=50)
# generate_fakePath_data(data_path='../data/FB15k-237', num=70)
# generate_fakePath_data(data_path='../data/FB15k-237', num=100)

# generate_fakePath_data(data_path='../data/YAGO3-10', num=30)
# generate_fakePath_data(data_path='../data/YAGO3-10', num=50)
# generate_fakePath_data(data_path='../data/YAGO3-10', num=70)
# generate_fakePath_data(data_path='../data/YAGO3-10', num=100)

#
# # 完全随机
# def generate_fakeRand_data(data_path="../data/FB15k/", num=10):
#     def read_triple(file_path, entity2id, relation2id):
#         triples = []
#         with open(file_path) as fin:
#             for line in fin:
#                 h, r, t = line.strip().split('\t')
#                 try:
#                     triples.append((entity2id[h], relation2id[r], entity2id[t]))
#                 except:
#                     pass
#         return triples
#
#     with open(os.path.join(data_path, 'entities.dict')) as fin:
#         entity2id = dict()
#         for line in fin:
#             eid, entity = line.strip().split('\t')
#             entity2id[entity] = int(eid)
#
#     with open(os.path.join(data_path, 'relations.dict')) as fin:
#         relation2id = dict()
#         for line in fin:
#             rid, relation = line.strip().split('\t')
#             relation2id[relation] = int(rid)
#     nrelation = len(relation2id)
#     nentity = len(entity2id)
#     train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
#     valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
#     test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
#     all_triples = set(train_triples + valid_triples + test_triples)
#
#     fake_triples = set()
#     while len(fake_triples)<len(train_triples) * num // 100:
#         h = random.choice(list(entity2id.values()))
#         r = random.choice(list(relation2id.values()))
#         t = random.choice(list(entity2id.values()))
#         fake_triples.add((h, r, t))
#
#     fake_triples = list(fake_triples)
#     with open(os.path.join(data_path, "fakeRand%d.pkl" % num), "wb") as f:
#         pickle.dump(fake_triples, f)
#     print(data_path)
