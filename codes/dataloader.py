#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from IPython import embed
import random
import numpy as np
import torch
import os
from collections import defaultdict
from torch.utils.data import Dataset

class NewTrain(Dataset):
    def __init__(self, all_triples, nentity, nrelation, negative_sample_size, mode, part_triples, neg_triples=None):
        # all_triples += part_triples
        self.len = len(part_triples)
        self.triples = part_triples
        self.neg_triples = neg_triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.true_head, self.true_tail = self.get_true_head_and_tail(all_triples)
        self.relation2head = defaultdict(lambda: set())
        self.relation2tail = defaultdict(lambda: set())
        self.subsampling_weights = {}
        for h, r, t in all_triples:
            self.relation2head[r].add(h)
            self.relation2tail[r].add(t)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = None

        negative_sample_list = []
        negative_sample_size = 0

        if self.neg_triples is not None:
            negative_sample_list = self.neg_triples[idx]# random.sample(self.neg_triples, self.negative_sample_size)
            negative_sample = torch.LongTensor(negative_sample_list)
        else:
            if self.mode == 'head-batch':
                candidate_list = list(self.relation2head[relation] - set(self.true_head[(relation, tail)]))
            else:
                candidate_list = list(self.relation2tail[relation] - set(self.true_tail[(head, relation)]))
            negative_sample_list = random.sample(candidate_list, min(self.negative_sample_size, len(candidate_list)))
            negative_sample_size += len(negative_sample_list)
            if negative_sample_size >= self.negative_sample_size:
                negative_sample = torch.LongTensor(negative_sample_list).view(1, 1)
            else:
                while negative_sample_size < self.negative_sample_size:
                    negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
                    if self.mode == 'head-batch':
                        mask = np.in1d(
                            negative_sample,
                            self.true_head[(relation, tail)],
                            assume_unique=True,
                            invert=True
                        )
                    elif self.mode == 'tail-batch':
                        mask = np.in1d(
                            negative_sample,
                            self.true_tail[(head, relation)],
                            assume_unique=True,
                            invert=True
                        )
                    else:
                        raise ValueError('Training batch mode %s not supported' % self.mode)
                    negative_sample = negative_sample[mask]
                    negative_sample_list.append(negative_sample)
                    negative_sample_size += negative_sample.size
                try:
                    negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
                except:
                    embed()
                negative_sample = torch.from_numpy(negative_sample).view(1, 1)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data if _[0] is not None], dim=0)
        negative_sample = torch.stack([_[1] for _ in data if _[1] is not None], dim=0)
        # subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, None, mode

    @staticmethod
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

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.subsampling_weights = {}
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        if positive_sample in self.subsampling_weights:
            subsampling_weight = self.subsampling_weights[positive_sample].view(1)
        else:
            subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)],
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)],
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode

    def update(self, score, triples):
        for i, triple in enumerate(triples):
            if score[i] > 0:
                self.subsampling_weights[tuple(triple)] *= 0.9
            else:
                self.subsampling_weights[tuple(triple)] += 0.001
            self.subsampling_weights[tuple(triple)] = self.subsampling_weights[tuple(triple)].clamp(max=1, min=0)
    
    @staticmethod
    def collate_fn(data):
        # print(data)
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
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

class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.dataloader_head = dataloader_head
        self.dataloader_tail = dataloader_tail
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

#
# class TransETrainDataset(Dataset):
#     def __init__(self, triples, nentity, nrelation, method="bern"):
#         self.len = len(triples)
#         self.triples = triples
#         self.triple_set = set(triples)
#         self.nentity = nentity
#         self.nrelation = nrelation
#         self.count = self.count_frequency(triples)
#         self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
#         self.confidence = {}
#         for triple in self.triples:
#             self.confidence[triple] = torch.Tensor([1.0])
#         self.negative_sample_method = method
#         self.random_number = [random.randint(0, 1000) for _ in range(7)]
#
#     def get_random_number(self, idx):
#         idx = idx % 7
#         self.random_number[idx] = (self.random_number[idx] * 25214903917 + 11) % 1000
#         return self.random_number[idx]
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, idx):
#         positive_sample = self.triples[idx]
#         head, relation, tail = positive_sample
#         confidence = self.confidence[positive_sample].view(1)
#
#         rand_num = self.get_random_number(idx)
#         prob = self.count[(head, relation)] / (self.count[(head, relation)] + self.count[(tail, -relation - 1)]) if self.negative_sample_method == "bern" else 0.5
#         if rand_num < prob * 1000:
#             fake_head = None
#             while fake_head is None:
#                 negative_candidate = np.random.randint(self.nentity, size=100)
#                 mask = np.in1d(negative_candidate, self.true_head[(tail, - relation - 1)], assume_unique=True, invert=True)
#                 negative_candidate = negative_candidate[mask]
#                 if negative_candidate.size > 0:
#                     fake_head = negative_candidate[rand_num % negative_candidate.size]
#             negative_sample = (fake_head, relation, tail)
#         else:
#             fake_tail = None
#             while fake_tail is None:
#                 negative_candidate = np.random.randint(self.nentity, size=100)
#                 mask = np.in1d(negative_candidate, self.true_tail[(head, relation)], assume_unique=True, invert=True)
#                 negative_candidate = negative_candidate[mask]
#                 if negative_candidate.size > 0:
#                     fake_tail = negative_candidate[rand_num % negative_candidate.size]
#             negative_sample = (head, relation, fake_tail)
#
#         positive_sample = torch.LongTensor(positive_sample)
#         negative_sample = torch.LongTensor(negative_sample)
#         return positive_sample, negative_sample, confidence
#
#     @staticmethod
#     def collate_fn(data):
#         positive_sample = torch.stack([_[0] for _ in data], dim=0)
#         negative_sample = torch.stack([_[1] for _ in data], dim=0)
#         confidence = torch.cat([_[2] for _ in data], dim=0)
#         return positive_sample, negative_sample, confidence
#
#     @staticmethod
#     def count_frequency(triples, start=4):
#         '''
#         Get frequency of a partial triple like (head, relation) or (relation, tail)
#         The frequency will be used for subsampling like word2vec
#         '''
#         count = {}
#         for head, relation, tail in triples:
#             if (head, relation) not in count:
#                 count[(head, relation)] = start
#             else:
#                 count[(head, relation)] += 1
#
#             if (tail, -relation - 1) not in count:
#                 count[(tail, -relation - 1)] = start
#             else:
#                 count[(tail, -relation - 1)] += 1
#         return count
#
#     @staticmethod
#     def get_true_head_and_tail(triples):
#         '''
#         Build a dictionary of true triples that will
#         be used to filter these true triples for negative sampling
#         '''
#
#         true_head = {}
#         true_tail = {}
#
#         for head, relation, tail in triples:
#             if (head, relation) not in true_tail:
#                 true_tail[(head, relation)] = []
#             true_tail[(head, relation)].append(tail)
#             if (relation, tail) not in true_head:
#                 true_head[(relation, tail)] = []
#             true_head[(relation, tail)].append(head)
#
#         for relation, tail in true_head:
#             true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
#         for head, relation in true_tail:
#             true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))
#
#         return true_head, true_tail
#
#
# class OneShotIterator(object):
#     def __init__(self, dataloader_head, dataloader_tail):
#         self.iterator= self.one_shot_iterator(dataloader_head)
#
#     def __next__(self):
#         data = next(self.iterator)
#         return data
#
#     @staticmethod
#     def one_shot_iterator(dataloader):
#         '''
#         Transform a PyTorch Dataloader into python iterator
#         '''
#         while True:
#             for data in dataloader:
#                 yield data
