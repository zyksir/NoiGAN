# # 遍历一个文件夹下所有文件
# import os
# import re
# dirs = os.listdir("./models/")
# table = []
# for name in dirs:
#     # if len(name.split("_")) != 4:
#     #     continue
#     if 'clear' not in name:
#         continue
#     filename = "./models/%s/train.log" % name
#     with open(filename, "r") as f:
#         lines = f.read().split("\n")[-6:]
#         output = name.split("_")[:3]
#         # output[2] = int(re.findall(r"\d+",output[2])[0])
#         for line in lines:
#             if "Test" in line:
#                 output.append(line.split(":")[-1])
#         table.append(output)
# table = sorted(table)
# table = [[str(i) for i in line] for line in table]
# table = "\n".join(["\t".join(line) for line in table])
# print(table)

import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

dataset = "YAGO3-10"
fake = 10
data_path = "./data/%s" % dataset

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
fake_triples = pickle.load(open(os.path.join(data_path, "fake%s.pkl" % fake), "rb"))

model = "TransE"
with open("./models/%s_%s_CLF_soft10/confidence_weight.pkl" % (model, dataset), "rb") as f:
    confidence_weight = pickle.load(f)

predict, label = [], []
for triple in train_triples:
    predict.append(confidence_weight[triple].item())
    label.append(1)

min100_triple = np.array(predict).argsort()[:100]
with open("codes/min_score_true100.txt", "w") as fw:
    for index in min100_triple:
        h, r, t = train_triples[index]
        head, relation, tail = id2entity[h], id2relation[r], id2entity[t]
        print("%s\t%s\t%s\t%f" % (head, relation, tail, predict[index]))
        fw.write("%s\t%s\t%s\t%f\n" % (head, relation, tail, predict[index]))

max100_triple = np.array(predict).argsort()[-100:]
with open("codes/max_score_true100.txt", "w") as fw:
    for index in max100_triple:
        h, r, t = train_triples[index]
        head, relation, tail = id2entity[h], id2relation[r], id2entity[t]
        print("%s\t%s\t%s\t%f" % (head, relation, tail, predict[index]))
        fw.write("%s\t%s\t%s\t%f\n" % (head, relation, tail, predict[index]))

# for triple in fake_triples:
#     predict.append(confidence_weight[triple].item())
#     label.append(0)
#
# y_score, y_true = np.array(predict), np.array(label)
# auc = roc_auc_score(y_true=y_true, y_score=y_score)
# specificity = recall_score(y_true=1 - y_true, y_pred=y_score < 0.5)
# print("auc: %f, specificity: %f" % (auc, specificity))