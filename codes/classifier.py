import os
import time
import math
import sys
import numpy as np
import logging
from tqdm import tqdm
from IPython import embed
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data import Dataset
from collections import defaultdict
import heapq
from dataloader import TrainDataset, BidirectionalOneShotIterator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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

def cal_threshold(all_score):
    all_score.sort()
    delta = all_score[100:] - all_score[:-100]
    threshold_delta = math.sqrt(delta[0])
    for i, score in enumerate(all_score):
        if delta[i] > threshold_delta:
            return score
    return all_score.mean() - 2 * all_score.var()


class ClassifierDataset(Dataset):
    def __init__(self, positive_triples, negative_triples):
        self.positive_triples = positive_triples
        self.negative_triples = negative_triples
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


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, output_dim=1):
        super(SimpleNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.F1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.F2 = nn.Linear(hidden_dim, self.output_dim)
        # self.F3 = nn.Linear(10, 1)

    def forward(self, score):
        '''
        :param sample: (batch_size, 3)
        :return: batch_size
        '''
        score = self.F1(score)
        score = self.dropout(torch.tanh(score))
        score = self.F2(score)
        score = torch.sigmoid(score)
        return score





