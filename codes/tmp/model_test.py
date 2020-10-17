import random
import torch
from tqdm import tqdm
import pickle
import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model import KGEModel, SimpleNN

data_path = "../data/YAGO3-10"
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
    id2entity = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)
        id2entity[int(eid)] = entity

with open(os.path.join(data_path, 'relations.dict')) as fin:
    relation2id = dict()
    id2relation = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)
        id2relation[int(rid)] = relation

nentity = len(entity2id)
nrelation = len(relation2id)
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

model_path = "../models/TransE_YAGO3-10_CLF20_hard_2"
fake_triples = pickle.load(open(os.path.join(data_path, "fake20.pkl"), "rb"))
checkpoint = torch.load(os.path.join(model_path, 'checkpoint'))


hidden_dim = 250
gamma = 12.0
kge_model = KGEModel(model_name="TransE", nentity=nentity, nrelation=nrelation, hidden_dim=hidden_dim, gamma=gamma).cuda()
classifier = SimpleNN(hidden_dim).cuda()
generator = SimpleNN(hidden_dim).cuda()
kge_model.load_state_dict(checkpoint['model_state_dict'])
try:
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
except:
    pass
distance, predict, true_label = [], [], []
# for triple in tqdm(train_triples, total=len(train_triples)):
i = 0
while i < len(train_triples):
    sys.stdout.write("%d in %d\r" % (i, len(train_triples)))
    sys.stdout.flush()
    j = min(i+1024, len(train_triples))
    sample = torch.LongTensor(train_triples[i:j]).cuda()
    h, r, t = classifier.get_embedding(kge_model, sample)
    w = classifier.forward(h + r - t).view(-1)
    d = torch.norm(h + r - t, p=1, dim=2).view(-1)
    predict.extend(w.tolist())
    distance.extend(d.tolist())
    true_label.extend([1 for _ in range(i, j)])
    i = j
# for triple in tqdm(fake_triples, total=len(fake_triples)):
i = 0
while i < len(fake_triples):
    sys.stdout.write("%d in %d\r" % (i, len(fake_triples)))
    sys.stdout.flush()
    j = min(i+1024, len(fake_triples))
    sample = torch.LongTensor(fake_triples[i:j]).cuda()
    h, r, t = classifier.get_embedding(kge_model, sample)
    w = classifier.forward(h + r - t).view(-1)
    d = torch.norm(h + r - t, p=1, dim=2).view(-1)
    predict.extend(w.tolist())
    distance.extend(d.tolist())
    true_label.extend([0 for _ in range(i, j)])
    i = j
predict = np.array(predict)
true_label = np.array(true_label)
accuracy = accuracy_score(y_true=true_label, y_pred=predict>0.5)
precision = precision_score(y_true=true_label, y_pred=predict>0.5)
recall = recall_score(y_true=true_label, y_pred=predict>0.5)
f1 = f1_score(y_true=true_label, y_pred=predict>0.5)
auc = roc_auc_score(y_true=true_label, y_score=predict)
print("accuracy = %f" % accuracy)
print("precision = %f" % precision)
print("recall = %f" % recall)
print("f1 = %f" % f1)
print("auc = %f" % auc)
l = len(train_triples)
# x = np.sort(predict[:l])
# print( (predict[:l]>0.375).sum() / (predict>0.375).sum() )
accuracy = accuracy_score(y_true=1-true_label, y_pred=predict<0.5)
precision = precision_score(y_true=1-true_label, y_pred=predict<0.5)
recall = recall_score(y_true=1-true_label, y_pred=predict<0.5)
f1 = f1_score(y_true=1-true_label, y_pred=predict<0.5)
auc = roc_auc_score(y_true=1-true_label, y_score=1-predict)
print("accuracy = %f" % accuracy)
print("precision = %f" % precision)
print("recall = %f" % recall)
print("f1 = %f" % f1)
print("auc = %f" % auc)

x = np.argsort(predict[:l])
false_negative = []
while len(false_negative) < 50:
    index = random.choice(list(range(10000)))
    h, r, t = train_triples[x[index]]
    h, r, t = id2entity[h], id2relation[r], id2entity[t]
    print("%s\t%s\t%s : %f" % (h, r, t, predict[x[index]]))
    false_negative.append((h, r, t, x[index]))

true_positive = []
while len(true_positive) < 50:
    index = random.choice(list(range(100000)))
    h, r, t = train_triples[x[-index-1]]
    h, r, t = id2entity[h], id2relation[r], id2entity[t]
    print("%s\t%s\t%s : %f" % (h, r, t, predict[x[-index-1]]))
    true_positive.append((h, r, t, x[-index-1]))

x = np.argsort(predict[l:])
true_negative = []
while len(true_negative) < 50:
    index = random.choice(list(range(10000)))
    h, r, t = fake_triples[x[-index-1]]
    h, r, t = id2entity[h], id2relation[r], id2entity[t]
    print("%s\t%s\t%s : %f" % (h, r, t, predict[l + x[-index-1]]))
    true_negative.append((h, r, t, x[-index-1]))


false_positive = []
while len(false_positive) < 50:
    index = random.choice(list(range(10000)))
    h, r, t = fake_triples[x[index]]
    h, r, t = id2entity[h], id2relation[r], id2entity[t]
    print("%s\t%s\t%s : %f" % (h, r, t, predict[l + x[index]]))
    false_positive.append((h, r, t, x[index]))


















mid2name = pickle.load(open("../data/FB15k/mid2name.pkl", "rb"))
mid2type = pickle.load(open("../data/FB15k/mid2type.pkl", "rb"))
r = relation2id["/people/profession/people_with_this_profession"]
for mid in entity2id.keys():
    if mid in mid2name and mid2name[mid] == "Michael Dobson":
        print(entity2id[mid])
        h = entity2id[mid]
for i, j in enumerate(train_triples):
    h_, r_, t_ = j
    if h == t_ and r_ == r:
        print(i)
        print(mid2name[id2entity[h_]])



pattern = set()
for h, r, t in all_triples:
    h, r, t = id2entity[h], id2relation[r], id2entity[t]
    try:
        h_name, h_type, t_name, t_type = mid2name[h], mid2type[h], mid2name[t], mid2type[t]
        pattern.add((h_type, r, t_type))
    except:
        pass

x = np.argsort(predict[:l])
print("False Negative")
false_negative = []
while len(false_negative) < 100:
    index = random.choice(list(range(10000)))
    h, r, t = train_triples[x[index]]
    h, r, t = id2entity[h], id2relation[r], id2entity[t]
    try:
        h_name, h_type, t_name, t_type = mid2name[h], mid2type[h], mid2name[t], mid2type[t]
        print("%s(%s)\t%s\t%s(%s) : %f" % (h_name, h_type, r, t_name, t_type, predict[x[index]]))
        false_negative.append((h_name, h_type, r, t_name, t_type, index))
    except:
        pass

print("True Positive : ")
true_positive = []
while len(true_positive) < 50:
    index = random.choice(list(range(1000)))
    h, r, t = train_triples[x[-index-1]]
    h, r, t = id2entity[h], id2relation[r], id2entity[t]
    try:
        h_name, h_type, t_name, t_type = mid2name[h], mid2type[h], mid2name[t], mid2type[t]
        print("%s(%s)\t%s\t%s(%s) : %f" % (h_name, h_type, r, t_name, t_type, predict[x[-index-1]]))
        true_positive.append((h_name, h_type, r, t_name, t_type, index))
    except:
        pass
# mistake = defaultdict(lambda : 0)
# for _, _, r, _, _, _ in false_negative:
#     mistake[r.split("/")[1]] += 1

x = np.argsort(predict[l:])
print("False Positive")
false_positive = []
while len(false_positive) < 50:
    index = random.choice(list(range(500)))
    h, r, t = fake_triples[x[index]]
    h, r, t = id2entity[h], id2relation[r], id2entity[t]
    try:
        h_name, h_type, t_name, t_type = mid2name[h], mid2type[h], mid2name[t], mid2type[t]
        print("%s(%s)\t%s\t%s(%s) : %f" % (h_name, h_type, r, t_name, t_type, predict[x[index]+l]))
        false_positive.append((h_name, h_type, r, t_name, t_type, index))
    except:
        pass

x = np.argsort(predict[l:])
print("True Negative : ")
true_negative = []
while len(true_negative) < 100:
    index = random.choice(list(range(1000)))
    h, r, t = fake_triples[x[index]]
    h, r, t = id2entity[h], id2relation[r], id2entity[t]
    try:
        h_name, h_type, t_name, t_type = mid2name[h], mid2type[h], mid2name[t], mid2type[t]
        print("%s(%s)\t%s\t%s(%s) : %f" % (h_name, h_type, r, t_name, t_type, predict[x[index]+l]))
        true_negative.append((h_name, h_type, r, t_name, t_type, index))
    except:
        pass
# mistake = defaultdict(lambda : 0)
# for _, _, r, _, _, _ in false_positive:
#     mistake[r.split("/")[1]] += 1

for h_name, h_type, r, t_name, t_type, index in true_negative:
    if (h_type, r, t_type) in pattern:
        print("#######%s(%s)\t%s\t%s(%s) : %f" % (h_name, h_type, r, t_name, t_type, predict[l + x[index]]))


h, r, t = train_triples[x[index]]
h, r, t = id2entity[h], id2relation[r], id2entity[t]
try:
    h_name, h_type, t_name, t_type = mid2name[h], mid2type[h], mid2name[t], mid2type[t]
    print("%s(%s)\t%s\t%s(%s) : %f" % (h_name, h_type, r, t_name, t_type, predict[x[index]]))
except:
    pass


mistake = defaultdict(lambda : 0)
type2set = defaultdict(lambda : set())
for h, r, t in train_triples:
    type = id2relation[r].split("/")[1]
    mistake[type] += 1
    type2set[type].add(h)
    type2set[type].add(t)
sorted(mistake.items(), key=lambda x:x[1])

mistake = defaultdict(lambda : 0)
for _, r,  _ in fake_triples:
    mistake[id2relation[r].split("/")[1]] += 1
sorted(mistake.items(), key=lambda x:x[1])

# evaluate_classification("../data/FB15k", "../models/TransE_FB15k_CLF10_256")

# weight = checkpoint["confidence"]
# predict, true_label = [], []
# for triple in tqdm(train_triples, total=len(train_triples)):
#     predict.append(weight[triple])
#     true_label.append(1)
# for triple in tqdm(fake_triples, total=len(fake_triples)):
#     predict.append(weight[triple])
#     true_label.append(0)
# predict = np.array(predict)
# true_label = np.array(true_label)
# accuracy = accuracy_score(y_true=true_label, y_pred=predict>0.5)
# precision = precision_score(y_true=true_label, y_pred=predict>0.5)
# recall = recall_score(y_true=true_label, y_pred=predict>0.5)
# f1 = f1_score(y_true=true_label, y_pred=predict>0.5)
# auc = roc_auc_score(y_true=true_label, y_score=predict)
# print("accuracy = %f" % accuracy)
# print("precision = %f" % precision)
# print("recall = %f" % recall)
# print("f1 = %f" % f1)
# print("auc = %f" % auc)
# accuracy = accuracy_score(y_true=1-true_label, y_pred=predict<0.5)
# precision = precision_score(y_true=1-true_label, y_pred=predict<0.5)
# recall = recall_score(y_true=1-true_label, y_pred=predict<0.5)
# f1 = f1_score(y_true=1-true_label, y_pred=predict<0.5)
# auc = roc_auc_score(y_true=1-true_label, y_score=1-predict)
# print("accuracy = %f" % accuracy)
# print("precision = %f" % precision)
# print("recall = %f" % recall)
# print("f1 = %f" % f1)
# print("auc = %f" % auc)

def show_curve(filename="./TransE_FB15k_CLF10_256/train.log", metric="HITS@10 "):
    import matplotlib.pyplot as plt

    X, Y = [], []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if metric in line and "Valid" in line:
                Y.append(float(line.split()[-1]))
                X.append(int(line.split(":")[0].split()[-1]))

    fig, ax = plt.subplots()
    ax.plot(X, Y)

    ax.set(xlabel='step', ylabel=metric,
           title=filename)
    ax.grid()

    # fig.savefig("test.png")
    plt.show()
# show_curve()

# convert tsc to data
# we want entities.dict, relations.dict, train.txt, test.txt, valid.txt
# import os
# import random
# entity2id = {}
# relation2id = {}
# all_triples = set()
# with open("../nl27k.tsv", "r") as f:
#     for line in f:
#         head, relation, tail, _ = line.strip().split()
#         if head not in entity2id:
#             entity2id[head] = len(entity2id)
#         if tail not in entity2id:
#             entity2id[tail] = len(entity2id)
#         if relation not in relation2id:
#             relation2id[relation] = len(relation2id)
#         all_triples.add((head, relation, tail))
# train_number, test_number = len(all_triples) // 7, len(all_triples) // 2
#
# train_triples = random.sample(list(all_triples), train_number)
# all_triples = all_triples - set(train_triples)
# test_triples = random.sample(list(all_triples), test_number)
# all_triples = all_triples - set(test_triples)
# valid_triples = all_triples
#
# # os.mkdir("../data/NELL27k")
# with open("../data/NELL27k/train.txt", "w") as f:
#     for head, relation, tail in train_triples:
#         f.write(head + "\t" + relation + "\t" + tail + "\n")
#
# with open("../data/NELL27k/test.txt", "w") as f:
#     for head, relation, tail in test_triples:
#         f.write(head + "\t" + relation + "\t" + tail + "\n")
#
# with open("../data/NELL27k/valid.txt", "w") as f:
#     for head, relation, tail in valid_triples:
#         f.write(head + "\t" + relation + "\t" + tail + "\n")
#
# with open("../data/NELL27k/entities.dict", "w") as f:
#     for entity, eid in entity2id.items():
#         f.write(str(eid) + "\t" + entity + "\n")
#
# with open("../data/NELL27k/relations.dict", "w") as f:
#     for relation, rid in relation2id.items():
#         f.write(str(rid) + "\t" + relation + "\n")
#
mid2name = {}
mid2type = {}
with open("../data/FB15k-237/entities.tsv", "r") as f:
    for line in f:
        if len(line.strip().split("\t")) == 5:
            mid, _, name, _, t = line.strip().split("\t")
            mid2name[mid] = name
            mid2type[mid] = t

import pickle
with open("../data/FB15k-237/mid2name.pkl", "wb") as f:
    pickle.dump(mid2name, f)

with open("../data/FB15k/mid2name.pkl", "wb") as f:
    pickle.dump(mid2name, f)

with open("../data/FB15k-237/mid2type.pkl", "wb") as f:
    pickle.dump(mid2type, f)

with open("../data/FB15k/mid2type.pkl", "wb") as f:
    pickle.dump(mid2type, f)

dataset = ""
with open("./models/TransE_%s_CLF_soft100/confidence_weight.pkl" % dataset, "rb") as f:
    confidence_weight = pickle.load(f)
