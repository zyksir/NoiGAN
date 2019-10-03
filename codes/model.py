# !/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import logging

import numpy as np
from torch.nn.init import xavier_normal_, xavier_normal
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import TestDataset

import heapq
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
        return [x for x in reversed([heapq.heappop(self.data) for x in range(len(self.data))])]

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        self.init()

    def init(self):
        logging.info("xavier_normal_ the parameters")
        xavier_normal_(self.entity_embedding)
        xavier_normal_(self.relation_embedding)

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    def get_embedding(self, model, sample, mode="single"):
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                model.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                model.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                model.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        return head, relation, tail

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, generator=None):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        # embed()
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        if generator is not None:
            positive_sample, negative_sample = generator.generate(model, positive_sample, negative_sample, mode,
                                                                  train=False, n_sample=args.negative_sample_size//2,
                                                                  model_name=args.model)

        negative_score = model((positive_sample, negative_sample), mode=mode)
        positive_score = model(positive_sample)
        if args.method == "LT":
            tmp = (negative_score.mean(dim=1) - positive_score.squeeze(dim=1) + 1.0).tolist()
            # train_iterator.dataloader_tail.dataset.update(tmp, positive_sample.tolist())
            train_iterator.dataloader_head.dataset.update(tmp, positive_sample.tolist())
            # embed()

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics

    @staticmethod
    def find_topK_triples(model, train_iterator, fake_triples, k=None, model_name="TransE"):
        fake_triples = set(fake_triples)
        model.eval()
        if k is None:
            k = len(train_iterator.dataloader_head.dataset.triples) // 5
        topk_heap = TopKHeap(k)
        i = 0
        all_triples = train_iterator.dataloader_head.dataset.triples
        while i < len(all_triples):
            sys.stdout.write("%d in %d\r" % (i, len(train_iterator.dataloader_head.dataset.triples)))
            sys.stdout.flush()
            j = min(i + 1024, len(train_iterator.dataloader_head.dataset.triples))
            sample = torch.LongTensor(train_iterator.dataloader_head.dataset.triples[i: j]).cuda()
            h, r, t = model.get_embedding(model, sample)
            if model_name == "TransE":
                s = h + r - t
            elif model_name == "ComplEx":
                s = ComplEx(h, r, t)
            score = (-torch.norm(s, p=1, dim=1)).view(-1).detach().cpu().tolist()
            for x, triple in enumerate(train_iterator.dataloader_head.dataset.triples[i: j]):
                topk_heap.push((score[x], triple))
            i = j
        topk_list = topk_heap.topk()
        _, topk_triples = list(zip(*topk_list))
        num_triples = len(all_triples) // 100
        num_fake = len(set(topk_triples[:num_triples]).intersection(fake_triples))
        logging.info("fake in top 1     %d / %d  %f" % (num_triples, num_fake, num_fake / num_triples))
        num_triples = len(all_triples) // 20
        num_fake = len(set(topk_triples[:num_triples]).intersection(fake_triples))
        logging.info("fake in top 5     %d / %d  %f" % (num_triples, num_fake, num_fake/num_triples) )
        num_triples = len(all_triples) // 10
        num_fake = len(set(topk_triples[:num_triples]).intersection(fake_triples))
        logging.info("fake in top 10    %d / %d  %f" % (num_triples, num_fake, num_fake / num_triples))
        num_triples = len(all_triples) // 5
        num_fake = len(set(topk_triples[:num_triples]).intersection(fake_triples))
        logging.info("fake in top 20    %d / %d  %f" % (num_triples, num_fake, num_fake / num_triples))

def ComplEx(head, relation, tail, mode="single"):
    re_head, im_head = torch.chunk(head, 2, dim=2)
    re_relation, im_relation = torch.chunk(relation, 2, dim=2)
    re_tail, im_tail = torch.chunk(tail, 2, dim=2)

    if mode == 'head-batch':
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        score = re_head * re_score + im_head * im_score
    else:
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

    # score = score.sum(dim=2)
    return score

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=10):
        super(SimpleNN, self).__init__()
        self.F1 = nn.Linear(input_dim, hidden_dim)
        self.F2 = nn.Linear(hidden_dim, 1)

    def get_embedding(self, model, sample, mode="single"):
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                model.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                model.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                model.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                model.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        return head, relation, tail

    def forward(self, score):
        batch, negative_sample_size, dim = score.size(0), score.size(1), score.size(2)
        score = score.view(-1, dim)
        score = self.F1(score)
        score = torch.tanh(score)
        score = self.F2(score)
        return torch.sigmoid(score)

    def predict(self, model, sample, model_name="TransE"):
        h, r, t = self.get_embedding(model, sample)
        if model_name == "TransE":
            return self.forward(h + r - t)
        elif model_name == "ComplEx":
            return self.forward(ComplEx(h, r, t))
        else:
            raise Exception("TransE or ComplEx??")

    @staticmethod
    def train_classifier_step(embed_model, classifier, clf_opt, train_iterator, args, generator, model_name="TransE"):
        embed_model.eval()
        classifier.eval()
        clf_opt.zero_grad()

        pos_sample, neg_sample, subsampling_weight, mode = next(train_iterator)
        if args.cuda:
            pos_sample, neg_sample, subsampling_weight = pos_sample.cuda(), neg_sample.cuda(), subsampling_weight.cuda()
        if generator is not None:
            pos_sample, neg_sample = generator.generate(embed_model, pos_sample, neg_sample, mode, train=False, model_name=model_name)
        else:
            neg_sample = neg_sample[:, 0].view(-1, 1)
        pos_head, pos_relation, pos_tail = classifier.get_embedding(embed_model, pos_sample)
        neg_head, neg_relation, neg_tail = classifier.get_embedding(embed_model, (pos_sample, neg_sample), mode=mode)
        if model_name == "TransE":
            pos_score = classifier(pos_head + pos_relation - pos_tail)
            neg_score = classifier(neg_head + neg_relation - neg_tail)
        elif model_name == "ComplEx":
            pos_score = classifier(ComplEx(pos_head, pos_relation, pos_tail))
            neg_score = classifier(ComplEx(neg_head, neg_relation, neg_tail))
        else:
            raise Exception("TransE or ComplEx??")
        if args.cuda:
            target = torch.cat([torch.ones(pos_score.size()), torch.zeros(neg_score.size())]).cuda()
        else:
            target = torch.cat([torch.ones(pos_score.size()), torch.zeros(neg_score.size())])
        loss = F.binary_cross_entropy(torch.cat([pos_score, neg_score]), target)
        loss.backward()
        clf_opt.step()
        log = {
            'positive_sample_mean_score': pos_score.mean().item(),
            'negative_sample_mean_score': neg_score.mean().item(),
            'loss': loss.item()
        }
        return log

    def generate(self, embed_model, pos, neg, mode, n_sample=1, temperature=1.0, train=True, model_name="TransE"):
        batch_size, negative_sample_size = neg.size(0), neg.size(1)

        # pos_head, pos_relation, pos_tail = self.get_embedding(embed_model, pos)
        neg_head, neg_relation, neg_tail = self.get_embedding(embed_model, (pos, neg), mode=mode)
        model_func = {'TransE': embed_model.TransE, 'DistMult': embed_model.DistMult, 'ComplEx': embed_model.ComplEx}
        scores = model_func[embed_model.model_name](neg_head, neg_relation, neg_tail, mode="head-batch")
        # if model_name == "TransE":
        #     scores = self.forward(neg_head + neg_relation - neg_tail).view(batch_size,
        #                                                                    negative_sample_size) / temperature
        # elif model_name == "ComplEx":
        #     scores = self.forward(ComplEx(neg_head, neg_relation, neg_tail)).view(batch_size,
        #                                                                    negative_sample_size) / temperature
        # else:
        #     raise Exception("TransE or ComplEx??")
        probs = torch.softmax(scores, dim=1)
        row_idx = torch.arange(0, batch_size).type(torch.LongTensor).unsqueeze(1).expand(batch_size, n_sample)
        sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        sample_neg = neg[row_idx, sample_idx.data.cpu()].view(batch_size, n_sample)
        if train:
            return pos, sample_neg, scores, sample_idx, row_idx
        else:
            return pos, sample_neg

    def discriminate_step(self, embed_model, pos, neg, mode, clf_opt, model_name="TransE"):
        self.train()
        clf_opt.zero_grad()

        pos_head, pos_relation, pos_tail = self.get_embedding(embed_model, pos)
        neg_head, neg_relation, neg_tail = self.get_embedding(embed_model, (pos, neg), mode=mode)
        if model_name == "TransE":
            negative_score = self.forward(neg_head + neg_relation - neg_tail)
            positive_score = self.forward(pos_head + pos_relation - pos_tail)
        elif model_name == "ComplEx":
            negative_score = self.forward(ComplEx(neg_head, neg_relation, neg_tail))
            positive_score = self.forward(ComplEx(pos_head, pos_relation, pos_tail))
        else:
            raise Exception("TransE or ComplEx??")
        target = torch.cat([torch.ones(positive_score.size()), torch.zeros(negative_score.size())]).cuda()
        loss = F.binary_cross_entropy(torch.cat([torch.sigmoid(positive_score), torch.sigmoid(negative_score)]), target)
        self.zero_grad()
        loss.backward()
        clf_opt.step()
        return loss, torch.tanh((negative_score - positive_score).sum())

    @staticmethod
    def train_GAN_step(embed_model, generator, discriminator, opt_gen, opt_dis, train_iterator, epoch_reward, epoch_loss, avg_reward, args, model_name="TransE"):
        embed_model.eval()
        generator.train()
        discriminator.train()
        MightFake_pos, MightFake_neg, subsampling_weight, mode = next(train_iterator)
        if args.cuda:
            MightFake_pos, MightFake_neg = MightFake_pos.cuda(), MightFake_neg.cuda()

        MightFake_pos, MightFake_neg, scores, sample_idx, row_idx = generator.generate(embed_model, MightFake_pos, MightFake_neg, mode, model_name=model_name)
        loss, rewards = discriminator.discriminate_step(embed_model, MightFake_pos, MightFake_neg, mode, opt_dis, model_name=model_name)
        epoch_reward += torch.sum(rewards)
        epoch_loss += loss
        rewards = rewards - avg_reward

        generator.zero_grad()
        log_probs = F.log_softmax(scores, dim=1)
        reinforce_loss = torch.sum(Variable(rewards) * log_probs[row_idx.cuda(), sample_idx.data])
        reinforce_loss.backward()
        opt_gen.step()
        return epoch_reward, epoch_loss, MightFake_pos.size(0)

    @staticmethod
    def find_topK_triples(model, classifier, train_iterator, clf_iterator, gen_iterator, k=None, soft=True, model_name="TransE"):
        model.eval()
        if k is None:
            k = len(clf_iterator.dataloader_head.dataset)
        all_weight = 0
        topk_heap = TopKHeap(k)
        bottomK_heap = TopKHeap(k)
        i = 0
        while i < len(train_iterator.dataloader_head.dataset.triples):
            sys.stdout.write("%d in %d\r" % (i, len(train_iterator.dataloader_head.dataset.triples)))
            sys.stdout.flush()
            j = min(i + 1024, len(train_iterator.dataloader_head.dataset.triples))
            sample = torch.LongTensor(train_iterator.dataloader_head.dataset.triples[i: j]).cuda()
            h, r, t = classifier.get_embedding(model, sample)
            if model_name == "TransE":
                s = h + r - t
            elif model_name == "ComplEx":
                s = ComplEx(h, r, t)
            if soft:
                weight = classifier.forward(s).view(-1).detach().cpu()
            else:
                weight = (classifier.forward(s).view(-1).detach().cpu() > 0.5).type(torch.float32) + 0.00001
            all_weight += weight.sum().item()
            score = (-torch.norm(h + r - t, p=1, dim=1)).view(-1).detach().cpu().tolist()
            for x, triple in enumerate(train_iterator.dataloader_head.dataset.triples[i: j]):
                train_iterator.dataloader_head.dataset.subsampling_weights[triple] = weight[x]
                topk_heap.push((score[x], triple))
                bottomK_heap.push((-score[x], triple))
            i = j
        topk_list = topk_heap.topk()
        bottomK_list = bottomK_heap.topk()
        _, clf_iterator.dataloader_head.dataset.triples = list(zip(*topk_list))
        clf_iterator.dataloader_tail.dataset.triples = [tri for tri in clf_iterator.dataloader_head.dataset.triples]
        _, gen_iterator.dataloader_head.dataset.triples = list(zip(*bottomK_list))
        gen_iterator.dataloader_tail.dataset.triples = [tri for tri in gen_iterator.dataloader_head.dataset.triples]

        return all_weight
