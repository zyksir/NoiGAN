#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import DataLoader

from model import KGEModel

from utils import *
from dataloader import *
from trainer import *

class TrainHelper(object):
    def __init__(self, inputData, args, kgeModel):
        self.kgeModel = kgeModel
        self.train_dataloader_head = DataLoader(
            TrainDataset(inputData.train_triples, args.nentity, args.nrelation,
                         args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        self.train_dataloader_tail = DataLoader(
            TrainDataset(inputData.train_triples, args.nentity, args.nrelation,
                         args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        self.train_iterator = BidirectionalOneShotIterator(self.train_dataloader_head, self.train_dataloader_tail)

        self.lr = args.learning_rate
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.kgeModel.parameters()),
            lr=self.lr
        )
        self.warm_up_steps = args.warm_up_steps if args.warm_up_steps else args.max_steps # adjust learning rate

    def warmUpDecreaseLR(self, step):
        if step >= self.warm_up_steps:
            self.lr = self.lr / 10
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.kgeModel.parameters()),
                lr=self.lr
            )
            self.warm_up_steps = self.warm_up_steps * 3
            logging.info('Change learning_rate to %f at step %d' % (self.lr, step))

def genModel(inputData, args):
    kgeModel = KGEModel(
        model_name=args.model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    if args.cuda:
        kgeModel = kgeModel.cuda()

    # if args.init_checkpoint:    # Restore model from checkpoint directory
    #     logging.info('Loading checkpoint %s...' % args.init_checkpoint)
    #     checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
    #     kgeModel.load_state_dict(checkpoint['model_state_dict'])
    #     # if args.do_train:
    #     #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # else:
    #     logging.info('Ramdomly Initializing %s Model...' % args.model)

    trainHelper = TrainHelper(inputData, args, kgeModel) # generate data loader and optimizer
    trainer = BaseTrainer(inputData, args, trainHelper)
    # if args.method == "CLF":
    #     trainer = ClassifierTrainer(inputData.train_triples, inputData.fake_triples, args, kge_model, args.hard)
    # elif args.method == "LT":
    #     trainer = LTTrainer(inputData.train_triples, inputData.fake_triples, args, kge_model)
    # elif args.method == "NoiGAN":
    #     trainer = NoiGANTrainer(inputData.train_triples, inputData.fake_triples, args, kge_model, args.hard)

    logging.info('Model Parameter Configuration:')
    for name, param in kgeModel.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    return trainer


def main(args):
    checkArgsValidation(args)
    inputData = init(args)
    trainer = genModel(inputData, args)
    kgeModel = trainer.trainHelper.kgeModel

    logging.info('Start Training...')
    logging.info(f"args is {args.__dict__}")

    if args.do_train:
        trainHelper = trainer.trainHelper
        logging.info('learning_rate = %f' % trainHelper.lr)
        for step in range(args.max_steps):
            trainer.basicTrainStep(step)

        save_variable_list = {
            'step': step, 
            'current_learning_rate': trainHelper.lr,
            'warm_up_steps': trainHelper.warm_up_steps
        }
        save_model(trainHelper.kgeModel, trainHelper.optimizer, save_variable_list, args, trainer)
        
    # if trainer is not None:
    #     logging.info("Evaluating Classifier on Train Dataset")
    #     metrics = trainer.test_ave_score(trainer)
    #     log_metrics('Train', step, metrics)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kgeModel.test_step(kgeModel, inputData.valid_triples, inputData.all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kgeModel.test_step(kgeModel, inputData.test_triples, inputData.all_true_triples, args)
        log_metrics('Test', step, metrics)
        # logging.info("\t".join([metric for metric in metrics.values()]))
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kgeModel.test_step(kgeModel, inputData.train_triples, inputData.all_true_triples, args)
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    main(parse_args())
