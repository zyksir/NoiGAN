import json
with open("/home/zyksir/KnowledgeGraphEmbedding/data/wn18rr/entities.dict", "r") as f, \
    open("WN18RR/entities_dict.json", "w") as fw:
    entity2id = dict()
    for line in f:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)
    json.dump(entity2id, fw)






import os
import gc
import sys
sys.path.append("./codes")
from utils import *
from trainer import *
from run import *
from classifier import *

def getTrainResultFromLog(filename, fileContent):
    lines = fileContent.split("\n")
    result = [filename]
    for line in lines:
        if "Test" not in line:
            continue
        try:
            value = float(line.split(":")[-1])
        except:
            continue
        # key = line.split(" ")[-5].strip()
        result.append("%.4f" % value)
    if len(result) != 6:
        return None
    return "\t".join(result)
    

def analyseAllLogs(analyseFunc, logFilter=["log", "train"]):
    def nameValid(name):
        for pattern in logFilter:
            if pattern not in name:
                return False
        return True
    for model_name in sorted(os.listdir("./models"), key=lambda name: (name.split("_")[0], name.split("_")[1], int(name.split("_")[2][3:]))):
        log_dir = "./models/%s/" % model_name
        for filename in os.listdir(log_dir):
            if nameValid(filename):
                with open(os.path.join(log_dir, filename), "r") as f:
                    result = analyseFunc(model_name, f.read())
                    if result is not None:
                        print(result)

# analyseAllLogs(getTrainResultFromLog)


from sklearn.metrics import average_precision_score, precision_recall_curve
def getTrainedTrainer(path):
    # path = "./models/DistMult_FB15k-237_dim1000_fake30/"
    args = parse_args()
    args.init_checkpoint = path
    override_config(args)
    args.do_train, args.init_checkpoint = False, path
    inputData = init(args)
    trainer = genModel(inputData, args)
    checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
    trainer.trainHelper.kgeModel.load_state_dict(checkpoint['model_state_dict'])
    return trainer

def calc_scores(kgeModel, inputData):
    fake_triples = torch.LongTensor(inputData.fake_triples).cuda()
    true_triples = list(set(inputData.train_triples) - set(inputData.fake_triples))
    true_triples = torch.LongTensor(true_triples).cuda()
    labels = np.array([1] * len(true_triples) + [0] * len(fake_triples))

    true_score = kgeModel(true_triples, mode="single")
    fake_score = kgeModel(fake_triples, mode="single")
    all_score = torch.vstack([true_score, fake_score]).cpu().detach().numpy()
    return labels, all_score


print_table = ""
for model_name in sorted(os.listdir("./models"), key=lambda name: (name.split("_")[0], name.split("_")[1], int(name.split("_")[2][3:]))):
    path = "./models/%s/" % model_name
    with torch.no_grad():
        try:
            trainer = getTrainedTrainer(path)
        except:
            continue
        labels, all_score = calc_scores(trainer.trainHelper.kgeModel, trainer.inputData)
    result = []
    p_auc = average_precision_score(y_true=labels, y_score=all_score)
    result.append(p_auc)
    precision, recall, thresholds = precision_recall_curve(labels, all_score)
    result.append(max(recall[precision >= 0.5]))
    result.append(max(recall[precision >= 0.6]))
    result.append(max(recall[precision >= 0.7]))
    result.append(max(recall[precision >= 0.8]))
    result.append(max(recall[precision >= 0.9]))
    result = [model_name] + ["%.4f" % score for score in result]
    print_table += "\t".join(result) + "\n"
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


# label 1: original train triples
# label 0: noise triples
