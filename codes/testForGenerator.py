import os
import json
import torch
import pickle
from torch.utils.data import DataLoader
from model import KGEModel
from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
from classifier import ClassifierTrainer, LTTrainer, NoiGANTrainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ARGS(object):
    def __init__(self):
        self.cuda = True
args = ARGS()

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


model = "TransE"
dataset = "YAGO3-10"
fake = 10
data_path = "../data/%s" % dataset
save_path = "../models/%s_%s_CLF_soft%d" % (model, dataset, fake)

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

args.nentity = len(entity2id)
args.nrelation = len(relation2id)
train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
fake_triples = pickle.load(open(os.path.join(data_path, "fake%s.pkl" % fake), "rb"))
train_triples += fake_triples
all_true_triples = train_triples + valid_triples + test_triples
with open(os.path.join(save_path, 'config.json'), 'r') as fjson:
    argparse_dict = json.load(fjson)
def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(save_path, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    args.fake = argparse_dict['fake']
    args.method = argparse_dict['method']
    args.save_path = argparse_dict['save_path']

override_config(args)
checkpoint = torch.load(os.path.join(save_path, 'checkpoint'))
kge_model = KGEModel(
        model_name=model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=argparse_dict["gamma"],
        double_entity_embedding=argparse_dict["double_entity_embedding"],
        double_relation_embedding=argparse_dict["double_relation_embedding"]
)
kge_model.load_state_dict(checkpoint['model_state_dict'])
kge_model = kge_model.cuda()
trainer = NoiGANTrainer(train_triples, fake_triples, args, kge_model, False)
trainer.classifier.load_state_dict(checkpoint['classifier'])
# trainer.generator.load_state_dict(checkpoint['generator'])
true_head, true_tail = TrainDataset.get_true_head_and_tail(all_true_triples)

query_head, query_relation, query_tail, args.mode = "Joby_Talbot", "wroteMusicFor", "The_Hitchhiker's_Guide_to_the_Galaxy_(film)", "tail-batch"
head, relation, tail = entity2id[query_head], relation2id[query_relation], entity2id[query_tail]
args.negative_sample_size = 1024

negative_sample_list = []
negative_sample_size = 0

# 得分最高得几个错误选项
while negative_sample_size < args.negative_sample_size:
    negative_sample = np.random.randint(args.nentity, size=args.negative_sample_size*2)
    if args.mode == 'head-batch':
        mask = np.in1d(
            negative_sample,
            true_head[(relation, tail)],
            assume_unique=True,
            invert=True
        )
    else:
        mask = np.in1d(
            negative_sample,
            true_tail[(head, relation)],
            assume_unique=True,
            invert=True
        )
    negative_sample = negative_sample[mask]
    negative_sample_list.append(negative_sample)
    negative_sample_size += negative_sample.size

negative_sample = np.concatenate(negative_sample_list)[:args.negative_sample_size]
negative_sample = negative_sample.tolist()
candidate_sample = []
if args.mode == "head-batch":
    for nhead in negative_sample:
        candidate_sample.append((nhead, relation, tail))
else:
    for ntail in negative_sample:
        candidate_sample.append((head, relation, ntail))
candidate_tensor = torch.LongTensor(candidate_sample).cuda()
confidence_weight = trainer.classifier(trainer.get_vector(candidate_tensor)).cpu().view(-1)
max_score_fake50 = confidence_weight.argsort()[-50:]
with open("max_score_fake50.txt", "w") as fw:
    for index in max_score_fake50:
        h, r, t = candidate_sample[index]
        head, relation, tail = id2entity[h], id2relation[r], id2entity[t]
        print("%s\t%s\t%s\t%f" % (head, relation, tail, confidence_weight[index]))
        fw.write("%s\t%s\t%s\t%f\n" % (head, relation, tail, confidence_weight[index]))
