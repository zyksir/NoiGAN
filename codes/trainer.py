from classifier import *
from utils import *

class BaseTrainer(object):
    def __init__(self, inputData, args, trainHelper):
        self.name = None
        self.inputData = inputData
        self.args = args
        self.trainHelper = trainHelper
        self.true_triples = list(set(inputData.train_triples) - set(inputData.fake_triples))
        self.threshold = 0.5
        self.trainingLogs = []

    # basic train functions
    def periodicCheck(self, step):
        args, inputData, trainHelper = self.args, self.inputData, self.trainHelper
        if step % args.log_steps == 0:
            metrics = {}
            for metric in self.trainingLogs[0].keys():
                metrics[metric] = sum([log[metric] for log in self.trainingLogs]) / len(self.trainingLogs)
            log_metrics('Training average', step, metrics)
            self.trainingLogs = []

        self.trainHelper.warmUpDecreaseLR(step)

        if step % args.save_checkpoint_steps == 0:
            save_variable_list = {
                'step': step,
                'current_learning_rate': trainHelper.lr,
                'warm_up_steps': trainHelper.warm_up_steps
            }
            save_model(trainHelper.kgeModel, trainHelper.optimizer, save_variable_list, args, self)

        if args.do_valid and step % args.valid_steps == 0:
            logging.info('Evaluating on Valid Dataset...')
            kgeModel = trainHelper.kgeModel
            metrics = kgeModel.test_step(kgeModel, inputData.valid_triples, inputData.all_true_triples, args)
            log_metrics('Valid', step, metrics)

    def basicTrainStep(self, step):
        kgeModel = self.trainHelper.kgeModel
        log = kgeModel.train_step(kgeModel, self.trainHelper.optimizer, self.trainHelper.train_iterator, self.args)
        self.trainingLogs.append(log)
        self.periodicCheck(step)

    def TransE(self, head, relation, tail):
        score = head + (relation - tail)
        return score

    def DistMult(self, head, relation, tail):
        score = head * (relation * tail)
        return score

    def ComplEx(self, head, relation, tail):
        re_head, im_head = torch.chunk(head, 2, dim=1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=1)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        score = re_head * re_score + im_head * im_score

        return score

    def RotatE(self, head, relation, tail):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=1)  # batch_size, dim

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation / (self.embedding_model.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head

        score = torch.stack([re_score, im_score], dim=0)    # 2, batch_size, dim
        score = score.norm(dim=0)   # batch_size, dim

        return score

    def get_vector(self, sample, mode="single"):
        if mode == "single":
            head = torch.index_select(self.embedding_model.entity_embedding, dim=0, index=sample[:, 0])
            relation = torch.index_select(self.embedding_model.relation_embedding, dim=0, index=sample[:, 1])
            tail = torch.index_select(self.embedding_model.entity_embedding, dim=0, index=sample[:, 2])
        elif mode == "head-batch":
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.embedding_model.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.embedding_model.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.embedding_model.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
        elif mode == "tail-batch":
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.embedding_model.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.embedding_model.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.embedding_model.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE
        }

        if self.embedding_model.model_name in model_func:
            score = model_func[self.embedding_model.model_name](head, relation, tail)
        else:
            raise ValueError('model %s not supported' % self.args.model)
        return score

    def find_positive_triples(self):
        k = len(self.train_triples) // 10
        # topk_heap = TopKHeap(len(self.train_triples) // 10)
        i = 0
        triple_score = []
        while i < len(self.train_triples):
            sys.stdout.write("find positive triples %d in %d\r" % (i, len(self.train_triples)))
            sys.stdout.flush()
            j = min(i + 1024, len(self.train_triples))
            sample = torch.LongTensor(self.train_triples[i: j])
            if self.args.cuda:
                sample = sample.cuda()
            score = self.embedding_model(sample).cpu()
            torch.cuda.empty_cache()
            for x, triple in enumerate(self.train_triples[i: j]):
                triple_score.append((score[x].item(), triple))
                # topk_heap.push((score[x].item(), triple))
            i = j
        random.shuffle(triple_score)
        quickselect(0, len(triple_score) - 1, triple_score, k)
        topk_triple_score = triple_score[:k]
        return [triple for score, triple in topk_triple_score]
        # return #topk_heap.topk()
        # return random.sample(train_triples, len(train_triples) // 10)

    @staticmethod
    def test_ave_score(trainer):
        true_score = []
        i = 0
        true_triples = trainer.true_triples
        while i < len(true_triples):
            j = min(i + 1024, len(true_triples))
            true_sample = torch.LongTensor(true_triples[i: j])
            if trainer.args.cuda:
                true_sample = true_sample.cuda()
            true_score.extend(trainer.classifier(trainer.get_vector(true_sample)).view(-1).tolist())
            i = j

        fake_score = []
        i = 0
        fake_triples = trainer.fake_triples
        while i < len(fake_triples):
            j = min(i + 1024, len(fake_triples))
            fake_sample = torch.LongTensor(fake_triples[i: j])
            if trainer.args.cuda:
                fake_sample = fake_sample.cuda()
            fake_score.extend(trainer.classifier(trainer.get_vector(fake_sample)).view(-1).tolist())
            i = j
        all_score = np.array(true_score + fake_score)
        true_score = np.array(true_score)
        fake_score = np.array(fake_score)
        # trainer.threshold = cal_threshold(all_score)
        trainer.threshold = 0.5
        true_left, fake_left = (true_score >= trainer.threshold).sum(), (fake_score >= trainer.threshold).sum()
        y_true = np.hstack([np.ones_like(true_score), np.zeros_like(fake_score)])
        y_score = np.hstack([true_score, fake_score])
        return {
            "true mean": true_score.mean(),
            "fake mean": fake_score.mean(),
            "true left": true_left,
            "fake_left": fake_left,
            "percent true left": true_left / (true_left+fake_left),
            "specificity": recall_score(y_true=1 - y_true, y_pred=y_score < 0.5),
            "auc": roc_auc_score(y_true=y_true, y_score=y_score)
        }

    def cal_confidence_weight(self):
        i = 0
        while i < len(self.train_triples):
            sys.stdout.write("cal confidence weight: %d in %d\r" % (i, len(self.train_triples)))
            sys.stdout.flush()
            j = min(i + 1024, len(self.train_triples))
            sample = torch.LongTensor(self.train_triples[i: j])
            if self.args.cuda:
                sample = sample.cuda()
            confidence_weight = self.classifier(self.get_vector(sample)).cpu()
            if self.hard:
                confidence_weight = confidence_weight >= self.threshold
            torch.cuda.empty_cache()
            for x, triple in enumerate(self.train_triples[i: j]):
                self.confidence_weight[triple] = confidence_weight[x]
            i = j

class LTTrainer(BaseTrainer):
    def __init__(self, train_triples, fake_triples, args, embedding_model):
        super(LTTrainer, self).__init__(train_triples, fake_triples, args, embedding_model)
        self.name = "LT"
        self.confidence_weight = {}
        for triple in self.train_triples:
            self.confidence_weight[triple] = torch.FloatTensor([1])

    def update(self, score, triples):
        for i, triple in enumerate(triples):
            if score[i] > 0:
                self.confidence_weight[tuple(triple)] *= 0.9
            else:
                self.confidence_weight[tuple(triple)] += 0.001
            self.confidence_weight[tuple(triple)] = self.confidence_weight[tuple(triple)].clamp(max=1, min=0)

class ClassifierTrainer(BaseTrainer):
    def __init__(self, train_triples, fake_triples, args, embedding_model, hard=False):
        super(ClassifierTrainer, self).__init__(train_triples, fake_triples, args, embedding_model)
        self.name = "CLF"
        self.embedding_model = embedding_model
        self.classifier = SimpleNN(args.hidden_dim, hidden_dim=5)
        if args.cuda:
            self.classifier = self.classifier.cuda()
        self.clf_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.01)
        self.positive_triples = self.train_triples[:len(train_triples)//10]
        self.negative_triples = None
        self.train_dataset = None
        self.train_dataloader = None
        self.train_iterator = None
        self.confidence_weight = {}
        self.threshold = 0.5
        self.hard = hard
        for triple in self.train_triples:
            self.confidence_weight[triple] = torch.FloatTensor([1])

    def find_negative_triples(self):
        true_head, true_tail, true_relation = \
            defaultdict(lambda: set()), defaultdict(lambda: set()), defaultdict(lambda: set())
        relation2tail, relation2head = defaultdict(lambda: set()), defaultdict(lambda: set())
        for h, r, t in self.train_triples:
            true_head[(r, t)].add(h)
            true_tail[(h, r)].add(t)
            true_relation[(h, t)].add(r)
            relation2tail[r].add(t)
            relation2head[r].add(h)
        all_entities, all_relations = set(range(self.args.nentity)), set(range(self.args.nrelation))
        negative_triples = set()
        triples_for_generate = random.sample(self.train_triples, len(self.train_triples) // 10 + 1000)
        for triple in tqdm(triples_for_generate):
            h, r, t = triple
            t_ = random.choice(list(all_entities - true_tail[(h, r)]))
            negative_triples.add((h, r, t_))
            h_ = random.choice(list(all_entities - true_head[(r, t)]))
            negative_triples.add((h_, r, t))
            # if len(relation2tail[r] - true_tail[(h, r)]) > 0:
            #     t__ = random.choice(list(relation2tail[r] - true_tail[(h, r)]))
            #     negative_triples.add((h, r, t__))
            # if len(relation2head[r] - true_head[(r, t)]) > 0:
            #     h__ = random.choice(list(relation2head[r] - true_head[(r, t)]))
            #     negative_triples.add((h__, r, t))
            # if len(all_relations - true_relation[(h, t)]) > 0:
            #     r__ = random.choice(list(all_relations - true_relation[(h, t)]))
            #     negative_triples.add((h, r__, t))
            # # negative_triples.add(random.choice(candidate_negative_triples))
        return list(negative_triples)

    @staticmethod
    def train_classifier(trainer):
        trainer.positive_triples = trainer.find_positive_triples()
        trainer.negative_triples = trainer.find_negative_triples()
        trainer.train_dataset = ClassifierDataset(trainer.positive_triples, trainer.negative_triples)
        trainer.train_dataloader = DataLoader(trainer.train_dataset, batch_size=128, shuffle=True, num_workers=5,
                                              collate_fn=ClassifierDataset.collate_fn)
        trainer.train_iterator = TrainIterator(trainer.train_dataloader)
        epochs = 1500
        trainer.classifier.train()
        trainer.embedding_model.eval()
        avg_loss, avg_pos_score, avg_neg_score = 0, 0, 0
        metrics = {}
        for i in range(epochs):
            positive_sample, negative_sample = next(trainer.train_iterator)
            if trainer.args.cuda:
                positive_sample, negative_sample = positive_sample.cuda(), negative_sample.cuda()
            positive_score = trainer.classifier(trainer.get_vector(positive_sample))
            negative_score = trainer.classifier(trainer.get_vector(negative_sample))
            target = torch.cat([torch.ones_like(positive_score), torch.zeros_like(negative_score)])
            if trainer.args.cuda:
                target = target.cuda()
            loss = F.binary_cross_entropy(torch.cat([positive_score.view(-1), negative_score.view(-1)]), target.view(-1))
            trainer.clf_optimizer.zero_grad()
            loss.backward()
            trainer.clf_optimizer.step()
            avg_loss += loss.item()
            avg_pos_score += positive_score.mean().item()
            avg_neg_score += negative_score.mean().item()
            if i % 500 == 0:
                metrics["loss_%d" % i] = avg_loss / 500
                metrics["positive_score_%d" % i] = avg_pos_score / 500
                metrics["negative_score_%d" % i] = avg_neg_score / 500
                avg_loss, avg_pos_score, avg_neg_score = 0, 0, 0
        # logging.info(trainer.positive_triples[:100])
        trainer.classifier.eval()

        return metrics

class NoiGANTrainer(BaseTrainer):
    def __init__(self, train_triples, fake_triples, args, embedding_model, hard=False):
        super(NoiGANTrainer, self).__init__(train_triples, fake_triples, args, embedding_model)
        self.name = "NOIGAN"
        self.classifier = SimpleNN(args.hidden_dim, hidden_dim=5)
        self.generator = SimpleNN(args.hidden_dim, hidden_dim=5)
        if args:
            self.classifier = self.classifier.cuda()
            self.generator = self.generator.cuda()
        self.clf_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.01)
        self.gen_optimizer = torch.optim.SGD(self.generator.parameters(), lr=0.01)

        self.confidence_weight = {}
        self.threshold = 0.5
        self.hard = hard
        for triple in self.train_triples:
            self.confidence_weight[triple] = torch.FloatTensor([1])

    def generate(self, pos, neg, mode, n_sample=1):
        batch_size, negative_sample_size = neg.size(0), neg.size(1)
        neg_vector = self.get_vector((pos, neg), mode=mode)
        neg_scores = self.generator(neg_vector)
        neg_probs = torch.softmax(neg_scores, dim=1).reshape(batch_size, negative_sample_size)
        row_idx = torch.arange(0, batch_size).type(torch.LongTensor).unsqueeze(1).expand(batch_size, n_sample)
        sample_idx = torch.multinomial(neg_probs, n_sample, replacement=True)
        sample_neg = neg[row_idx, sample_idx.data.cpu()].view(batch_size, n_sample)
        return pos, sample_neg, neg_scores, sample_idx, row_idx

    def discriminate(self, pos, neg, mode):
        self.classifier.train()
        self.clf_optimizer.zero_grad()
        pos_vector = self.get_vector(pos)
        neg_vector = self.get_vector((pos, neg), mode=mode)
        pos_scores = self.classifier(pos_vector)
        neg_scores = self.classifier(neg_vector).reshape_as(pos_scores)

        target = torch.cat([torch.ones(pos_scores.size()), torch.zeros(neg_scores.size())])
        if self.args.cuda:
            target = target.cuda()
        loss = F.binary_cross_entropy(torch.cat([pos_scores, neg_scores]), target)
        loss.backward()
        self.clf_optimizer.step()
        return loss, torch.tanh((neg_scores - pos_scores).sum())

    @staticmethod
    def train_NoiGAN(trainer):
        trainer.embedding_model.eval()

        st = time.time()
        trainer.positive_triples = trainer.find_positive_triples()
        et = time.time()
        print("take %d s to find positive triples" % (et - st))

        trainer.train_dataset_head = TrainDataset(trainer.train_triples,
                                                  trainer.args.nentity,
                                                  trainer.args.nrelation,
                                                  trainer.args.negative_sample_size,
                                                  "head-batch")
        trainer.train_dataset_head.triples = trainer.positive_triples
        trainer.train_dataset_tail = TrainDataset(trainer.train_triples,
                                                  trainer.args.nentity,
                                                  trainer.args.nrelation,
                                                  trainer.args.negative_sample_size,
                                                  "tail-batch")
        trainer.train_dataset_tail.triples = trainer.positive_triples
        trainer.train_dataloader_head = DataLoader(trainer.train_dataset_head,
                                                   batch_size=128,
                                                   shuffle=True,
                                                   num_workers=5,
                                                   collate_fn=TrainDataset.collate_fn)
        trainer.train_dataloader_tail = DataLoader(trainer.train_dataset_tail,
                                                   batch_size=128,
                                                   shuffle=True,
                                                   num_workers=5,
                                                   collate_fn=TrainDataset.collate_fn)
        trainer.train_iterator = BidirectionalOneShotIterator(
                                    trainer.train_dataloader_head,
                                    trainer.train_dataloader_tail
                                )
        epochs = 1500
        epoch_reward, epoch_loss, avg_reward = 0, 0, 0
        for i in range(epochs):
            trainer.generator.train()
            positive_sample, negative_sample, subsampling_weight, mode = next(trainer.train_iterator)
            if trainer.args.cuda:
                positive_sample = positive_sample.cuda()    # [batch_size, 3]
                negative_sample = negative_sample.cuda()    # [batch_size, negative_sample_size]
            #$ embed()
            pos, neg, scores, sample_idx, row_idx = trainer.generate(positive_sample, negative_sample, mode)
            loss, rewards = trainer.discriminate(pos, neg, mode)
            epoch_reward += torch.sum(rewards)
            epoch_loss += loss
            rewards = rewards - avg_reward

            trainer.generator.zero_grad()
            log_probs = F.log_softmax(scores, dim=1)
            reinforce_loss = torch.sum(Variable(rewards) * log_probs[row_idx.cuda(), sample_idx.data])
            reinforce_loss.backward()
            trainer.gen_optimizer.step()
            trainer.generator.eval()

