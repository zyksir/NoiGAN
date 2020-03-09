import torch
import pickle
import random
import numpy as np
import matplotlib.pyplot as pp


weight_path = "./models/TransE_FB15k-237_CLFPath100_soft/weight"
weight = torch.load(weight_path)
fake = set(pickle.load(open("./data/FB15k-237/fakePath100.pkl", "rb")))
true_x, true_y = [], []
fake_x, fake_y = [], []
for triple, score in weight.items():
    if triple in fake:
        fake_x.append(random.uniform(0, 1))
        fake_y.append(score.item())
    else:
        true_x.append(random.uniform(0, 1))
        true_y.append(score.item())

pp.plot(true_x, true_y)