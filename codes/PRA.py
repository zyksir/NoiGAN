#!/usr/bin/python3

from collections import defaultdict
import time
import pickle
import copy
from random import sample
import sys
sys.setrecursionlimit(20000000)
dataset2node_number = {"FB15k": 14950, "FB15k-237": 14540, "wn18": 40942, "wn18rr": 40942, "YAGO3-10": 123181}


class KG:
    def __init__(self):
        self.head2relation_tail = {}
        self.head2tail_relation = {}
        self.relation2head = {}

    def add(self, head, relation, tail):
        if head not in self.head2relation_tail:
            self.head2relation_tail[head] = {}
        if relation not in self.head2relation_tail[head]:
            self.head2relation_tail[head][relation] = set()
        self.head2relation_tail[head][relation].add(tail)

        if head not in self.head2tail_relation:
            self.head2tail_relation[head] = {}
        if tail not in self.head2tail_relation[head]:
            self.head2tail_relation[head][tail] = set()
        self.head2tail_relation[head][tail].add(relation)

        if relation not in self.relation2head:
            self.relation2head[relation] = set()
        self.relation2head[relation].add(head)


###################################################################################
# def data_analyze(dataset):
dataset = "YAGO3-10"
kg = KG()
filename = "../data/%s/train.txt" % dataset
with open(filename, "r") as f:
    datas = f.readlines()
    for data in datas:
        [node, relation, next_node] = data.strip().split("\t")
        kg.add(node, relation, next_node)

ave_num_rel, max_num_rel, ave_num_tail, max_num_tail = 0, 0, 0, 0
ave_rel_tail, max_rel_tail, total_rel_tail = 0, 0, 0
for key, value in kg.head2relation_tail.items():
    ave_num_rel += len(value)
    max_num_rel = max(max_num_rel, len(value))
    tmp = 0
    for _key, _value in value.items():
        ave_rel_tail += len(_value)
        max_rel_tail = max(max_rel_tail, len(_value))
        tmp += len(_value)
    max_num_tail = max(max_num_tail, tmp)
    ave_num_tail += tmp

ave_rel_tail /= ave_num_rel
print("each head has average %f tail for each relation in %s" % (ave_rel_tail, dataset))
print("each head has at most %d tail for each relation in %s" % (max_rel_tail, dataset))

ave_num_rel /= len(kg.head2relation_tail)
print("each head has average %f relation in %s" % (ave_num_rel, dataset))
print("each head has at most %d relation in %s" % (max_num_rel, dataset))

ave_num_tail /= len(kg.head2relation_tail)
print("each head has average %f tail in %s" % (ave_num_tail, dataset))
print("each head has at most %d tail in %s" % (max_num_tail, dataset))


ave_rel2head = 0
for key in kg.relation2head.keys():
    ave_rel2head += len(kg.relation2head[key])
ave_rel2head /= len(kg.relation2head)

print("number of relation types in %s: %d" % (dataset, len(kg.relation2head)))
print("each relation has average %d head in %s" % (ave_rel2head, dataset))

# for dataset in dataset2node_number.keys():
#     data_analyze(dataset)

###################################################################################
# def get_reasonable_path(dataset, node_number=0):
dataset = "FB15k"

node_number = dataset2node_number[dataset]
kg = KG()
filename = "../data/%s/train.txt" % dataset
with open(filename, "r") as f:
    datas = f.readlines()
    for data in datas:
        [node, relation, next_node] = data.strip().split("\t")
        kg.add(node, relation, next_node)
for head in kg.head2relation_tail:
    all_relation = list(kg.head2relation_tail[head].keys())
    for relation in all_relation:
        tails = kg.head2relation_tail[head][relation]
        if len(tails) > 3:
            tails = sample(tails, 3)
        for tail in tails:
            second_relation_list = list(kg.head2relation_tail[tail].keys())
            if len(second_relation_list) > 3:
                second_relation_list = sample(second_relation_list, 3)
                for second_relation in second_relation_list:
                    second_tails = kg.head2relation_tail[tail][second_relation]
                    second_tails = sample(second_tails, 1)
                    kg.head2relation_tail[(relation, second_relation)] = second_tails



                ###################################################################################
# def get_reasonable_path(dataset, node_number=0):
dataset = "FB15k"
# dataset = "FB15k-237"
# dataset = "wn18"
# dataset = "wn18rr"
# dataset = "YAOGO3-10"

node_number = dataset2node_number[dataset]
all_triples = 0
all_path = 0
prune_0001 = 0
prune_triple = 0
kg = KG()
filename = "../data/%s/train.txt" % dataset
with open(filename, "r") as f:
    datas = f.readlines()
    for data in datas:
        [node, relation, next_node] = data.strip().split("\t")
        kg.add(node, relation, next_node)
num_unreachable = 0
reasonable_path = {}
for relation in kg.relation2head.keys():
    # reasonable_path[relation] = set()
    reasonable_path[relation] = {}
    for second_relation in kg.relation2head.keys():
        # reasonable_head = set()
        reasonable_head = {}
        for head in kg.relation2head[relation]:
            mid_heads = kg.relation2head[second_relation].intersection(kg.head2relation_tail[head][relation])
            if len(mid_heads) > 0:
                # reasonable_head.add(head)
                reasonable_head[head] = set()
                for mid_head in mid_heads:
                    reasonable_head[head] = reasonable_head[head].union(kg.head2relation_tail[mid_head][second_relation])
                all_triples += len(reasonable_head[head])
        head_num = len(reasonable_head)
        if head_num > 0:
            reasonable_path[relation][second_relation] = reasonable_head
            all_path += 1

print("find %d triples and %d paths in %s" % (all_triples, all_path, dataset))
print("we can prune %d triples and %d paths with alpha=0.001 in %s" % (prune_triple, prune_0001, dataset))
# return reasonable_path

# for dataset in dataset2node_number.keys():
#     reasonable_path = get_reasonable_path(dataset, node_number=dataset2node_number[dataset])
#     with open("../data/%s/reasonable_path.dict" % (dataset), "w") as f:
#         pickle.dump(reasonable_path, f)


###################################################################################
# reasonable_path[relation][second_relation][head] = set(tails)
# def combine_reasonable_path(dataset, node_number):

import pickle
dataset = "FB15k"
# dataset = "FB15k-237"
# dataset = "wn18"
# dataset = "wn18rr"
# dataset = "YAGO3-10"
def ADD(head, relation, tail, dict):
    if head not in dict:
        dict[head] = {}
    if relation not in dict[head]:
        dict[head][relation] = set()
    dict[head][relation].add(tail)

head2relation2tail = {}
tail2relation2head = {}
naive_head2relation2tail = {}
naive_tail2relation2head = {}
None_relation = "NONE_RELATION"
with open("../data/%s/train.txt" % dataset, "r") as f:
    datas = f.readlines()
    for data in datas:
        [head, relation, tail] = data.strip().split("\t")
        relation = (relation, None_relation)
        ADD(head, relation, tail, head2relation2tail)
        ADD(tail, relation, head, tail2relation2head)

with open("../data/%s/reasonable_path.dict" % dataset, "r") as f:
    reasonable_path = pickle.load(f)
    for relation in reasonable_path.keys():
        for second_relation in reasonable_path[relation].keys():
            if True:  # len(all_triple_head) > 0.05 * node_number:
                for head in reasonable_path[relation][second_relation].keys():
                    for tail in reasonable_path[relation][second_relation][head]:
                        ADD(tail, (relation, second_relation), head, tail2relation2head)
                        ADD(head, (relation, second_relation), tail, head2relation2tail)
with open("../data/%s/head2relation2tail.dict" % dataset, "w") as f:
    pickle.dump(head2relation2tail, f)
with open("../data/%s/tail2relation2head.dict" % dataset, "w") as f:
    pickle.dump(tail2relation2head, f)

###################################################################################
# now we want to change dictionary to generate tetrad
# we want to sample at most 100 neighbors for each head in FB15k/FB15k-237
# we want to sample at most 50 neighbors for each head/tail in other datasets
import pickle
from random import sample
None_relation = "NONE_RELATION"
for dataset, sample_size in [("FB15k", 50), ("FB15k-237", 50), ("wn18rr", 25), ("YAGO3-10", 25), ("wn18", 25)]:
    head_tetrad = []
    with open("../data/%s/head2relation2tail.dict" % dataset, "r") as f:
        head2relation2tail = pickle.load(f)
    with open("../data/%s/tail2relation2head.dict" % dataset, "r") as f:
        tail2relation2head = pickle.load(f)
    for head in head2relation2tail.keys():
        relation2tail = head2relation2tail[head]
        one_hop_relation_triples = []
        two_hop_relation_triples = []
        for relation in relation2tail.keys():
            if None_relation in relation:
                for tail in relation2tail[relation]:
                    one_hop_relation_triples.append((head, relation, tail))
            else:
                for tail in relation2tail[relation]:
                    two_hop_relation_triples.append((head, relation, tail))

        if len(one_hop_relation_triples) > sample_size:
            one_hop_relation_triples = sample(one_hop_relation_triples, sample_size)
        if len(two_hop_relation_triples) > sample_size:
            two_hop_relation_triples = sample(two_hop_relation_triples, sample_size)
        head_tetrad.extend(one_hop_relation_triples)
        head_tetrad.extend(two_hop_relation_triples)

    with open("../data/%s/train_head_tetrad" % dataset, "w") as f:
        for tetrad in head_tetrad:
            f.write(tetrad[0]+"\t"+tetrad[1][0]+"\t"+tetrad[1][1]+"\t"+tetrad[2]+"\n")

    tail_tetrad = []
    for tail in tail2relation2head.keys():
        relation2head = tail2relation2head[tail]
        one_hop_relation_triples = []
        two_hop_relation_triples = []
        for relation in relation2head.keys():
            if None_relation in relation:
                for head in relation2head[relation]:
                    one_hop_relation_triples.append((head, relation, tail))
            else:
                for head in relation2head[relation]:
                    two_hop_relation_triples.append((head, relation, tail))

        if len(one_hop_relation_triples) > sample_size:
            one_hop_relation_triples = sample(one_hop_relation_triples, sample_size)
        if len(two_hop_relation_triples) > sample_size:
            two_hop_relation_triples = sample(two_hop_relation_triples, sample_size)
        tail_tetrad.extend(one_hop_relation_triples)
        tail_tetrad.extend(two_hop_relation_triples)

    with open("../data/%s/train_tail_tetrad" % dataset, "w") as f:
        for tetrad in tail_tetrad:
            f.write(tetrad[0]+"\t"+tetrad[1][0]+"\t"+tetrad[1][1]+"\t"+tetrad[2]+"\n")


###################################################################################
for dataset in dataset2node_number.keys():
    with open("../data/%s/train_tail_tetrad_" % dataset, "r") as f:
        with open("../data/%s/train_tail_tetrad" % dataset, "w") as fw:
            for line in f:
                tetrad = line.strip().split("\t")
                if tetrad[2] == None_relation:
                    fw.write(tetrad[0] + "\t" + tetrad[2] + "\t" + tetrad[1] + "\t" + tetrad[3] + "\n")
                else:
                    fw.write(tetrad[0] + "\t" + tetrad[1] + "\t" + tetrad[2] + "\t" + tetrad[3] + "\n")










































# path = []
# results = []
# max_length = 5
# def DFS(kg, reasonable_path, begin, end, level):
#     if begin == end:
#         tmp = [p for p in path]
#         results.append(tmp)
#         return
#
#     if end in kg.head2tail_relation[begin]:
#         for relation in kg.head2tail_relation[begin][end]:
#             tmp = [p for p in path]
#             tmp.append(relation)
#             results.append(tmp)
#         return
#
#     if level >= max_length:
#         return
#
#     for relation in kg.head2relation_tail.keys():
#         for second_relation in reasonable_path[relation].keys():
#             if begin not in reasonable_path[relation][second_relation]:
#                 continue
#             # tails = set()
#             # mid_heads = kg.head2relation_tail[begin][relation].intersection(kg.relation2head[second_relation])
#             # for mid_head in mid_heads:
#             #     tails = tails.union(kg.head2relation_tail[mid_head][second_relation])
#             tails = reasonable_path[relation][second_relation][begin]
#             for tail in tails:
#                 path.append(relation)
#                 path.append(second_relation)
#                 DFS(kg, reasonable_path, tail, end, level=level+2)
#                 path.remove(second_relation)
#                 path.remove(relation)
#

