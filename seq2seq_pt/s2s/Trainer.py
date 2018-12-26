import os
import math
import time

import torch
from torch import nn

import s2s
from s2s.xloss import generate_copy_loss_function, coverage_loss_function
from s2s.Metrics import Statistics
from s2s.Evaluator import CNNDMRougeEvaluator

try:
    import ipdb
except ImportError:
    pass


class Trainer(object):
    def __init__(self, logger, opt):
        self.logger = logger
        self.opt = opt
        self.dicts = None
        self.train_data = None
        self.dev_data = None
        self.model = None
        self.optim = None
        self.translator = None
        self.num_epoch = 0
        self.num_batch = 0
        self.evaluator: s2s.Evaluator.Evaluator = None

    def get_logger_and_opt(self):
        return self.logger, self.opt

    def prepare_data(self):
        logger, opt = self.get_logger_and_opt()
        import onlinePreprocess
        onlinePreprocess.lower = self.opt.lower_input
        onlinePreprocess.MAX_SRC_LENGTH = opt.max_src_length
        onlinePreprocess.MAX_TGT_LENGTH = opt.max_tgt_length
        if opt.truncate_sentence:
            onlinePreprocess.TRUNCATE = True
        onlinePreprocess.shuffle = 1 if opt.process_shuffle else 0
        from onlinePreprocess import prepare_data_online
        dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab)

        dict_checkpoint = opt.train_from if opt.train_from else opt.train_from_state_dict
        if dict_checkpoint:
            logger.info('Loading dicts from checkpoint at %s' % dict_checkpoint)
            checkpoint = torch.load(dict_checkpoint)
            dataset['dicts'] = checkpoint['dicts']

        train_data = s2s.Dataset(dataset['train']['src'], dataset['train']['tgt'],
                                 dataset['train']['extended_src'], dataset['train']['extended_tgt'],
                                 dataset['train']['extend_vocab_size'],
                                 opt.batch_size, opt.gpus)
        dicts = dataset['dicts']
        logger.info(' * vocabulary size. source = %d; target = %d' %
                    (dicts['src'].size(), dicts['tgt'].size()))
        logger.info(' * number of training sentences. %d' %
                    len(dataset['train']['src']))
        logger.info(' * maximum batch size. %d' % opt.batch_size)

        self.train_data = train_data
        self.dicts = dicts

        if opt.dev_input_src and opt.dev_ref:
            # from onlinePreprocess import load_dev_data
            # self.dev_data = load_dev_data(opt, translator, opt.dev_input_src, opt.dev_ref)
            dev_dataset = prepare_data_online(opt.dev_input_src, opt.src_vocab, opt.dev_ref, opt.tgt_vocab)
            self.dev_data = s2s.Dataset(dev_dataset['train']['src'], dev_dataset['train']['tgt'],
                                        dev_dataset['train']['extended_src'], dev_dataset['train']['extended_tgt'],
                                        dev_dataset['train']['extend_vocab_size'],
                                        opt.batch_size, opt.gpus)

    def build_model(self):
        logger = self.logger
        opt = self.opt
        dicts = self.dicts
        logger.info('Building model...')

        encoder = s2s.Models.Encoder(opt, dicts['src'])
        decoder = s2s.Models.Decoder(opt, dicts['tgt'])
        dec_initer = s2s.Models.DecInit(opt)

        generator = nn.Sequential(
            nn.Linear(opt.dec_rnn_size, dicts['tgt'].size()),
            # nn.LogSoftmax(dim=1)
            nn.Softmax(dim=1)
        )

        model = s2s.Models.NMTModel(encoder, decoder, dec_initer)
        model.generator = generator

        if len(opt.gpus) >= 1:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        self.model = model
        if opt.test_during_training:
            self.translator = s2s.Translator(opt, model, self.dicts)
            self.evaluator = CNNDMRougeEvaluator(opt.dev_input_src, opt.dev_ref, self.translator)

    def restore_model(self):
        """
        logger = self.logger
        opt = self.opt
        if opt.train_from:
            logger.info('Loading model from checkpoint at %s' % opt.train_from)
            chk_model = checkpoint['model']
            generator_state_dict = chk_model.generator.state_dict()
            model_state_dict = {k: v for k, v in chk_model.state_dict().items() if 'generator' not in k}
            model.load_state_dict(model_state_dict)
            generator.load_state_dict(generator_state_dict)
            self.opt.start_epoch = checkpoint['epoch'] + 1

        if opt.train_from_state_dict:
            logger.info('Loading model from checkpoint at %s' % opt.train_from_state_dict)
            model.load_state_dict(checkpoint['model'])
            generator.load_state_dict(checkpoint['generator'])
            opt.start_epoch = checkpoint['epoch'] + 1
        """
        raise NotImplemented

    def build_optimizer(self):
        logger, opt = self.get_logger_and_opt()

        if not opt.train_from_state_dict and not opt.train_from:
            for pr_name, p in self.model.named_parameters():
                logger.info(pr_name)
                # p.data.uniform_(-opt.param_init, opt.param_init)
                if p.dim() == 1:
                    # p.data.zero_()
                    p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
                else:
                    nn.init.xavier_normal_(p, math.sqrt(3))

            self.model.encoder.load_pretrained_vectors(opt)
            self.model.decoder.load_pretrained_vectors(opt)

            optim = s2s.Optim(
                opt,
                opt.optim, opt.learning_rate,
                max_grad_norm=opt.max_grad_norm,
                max_weight_value=opt.max_weight_value,
                lr_decay=opt.learning_rate_decay,
                start_decay_at=opt.start_decay_at,
                decay_bad_count=opt.halve_lr_bad_count,
                tune_direction=opt.tune_direction
            )
        else:
            logger.info('Loading optimizer from checkpoint:')
            # optim = checkpoint['optim']
            # logger.info(optim)

        optim.set_parameters(self.model.parameters())

        # if opt.train_from or opt.train_from_state_dict:
        #     optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())

        # optim = s2s.Optim(
        #     opt,
        #     opt.optim, opt.learning_rate,
        #     max_grad_norm=opt.max_grad_norm,
        #     max_weight_value=opt.max_weight_value,
        #     lr_decay=opt.learning_rate_decay,
        #     start_decay_at=opt.start_decay_at,
        #     decay_bad_count=opt.halve_lr_bad_count
        # )
        self.optim = optim

    def get_regular_epoch_dump_path(self) -> str:
        logger, opt = self.get_logger_and_opt()
        save_model_path = 'model'
        if opt.save_path:
            if not os.path.exists(opt.save_path):
                os.makedirs(opt.save_path)
            save_model_path = opt.save_path + os.path.sep + save_model_path
        save_model_path = '{0}_epoch{1}.pt'.format(save_model_path, self.num_epoch)
        return save_model_path

    def get_dev_dump_path(self, metric) -> str:
        base_path = self.get_regular_epoch_dump_path()
        res = "{0}_dev_metric_{1}.pt".format(base_path, round(metric, 4))
        return res

    def save_model(self, save_path: str):
        logger, opt = self.get_logger_and_opt()
        model = self.model

        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(
            opt.gpus) > 1 else model.generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': self.dicts,
            'opt': opt,
            'epoch': self.num_epoch,
            'optim': self.optim
        }
        torch.save(checkpoint, save_path)
        logger.info('Dump a checkpoint to {0}'.format(save_path))

    def _forward_one_batch(self, batch, criterion):
        logger, opt = self.get_logger_and_opt()
        model = self.model
        model.zero_grad()
        g_outputs, c_outputs, c_gate_values, all_attn, all_coverage = model(batch)
        extended_src_batch = batch[1][0]
        extended_vocab_size = batch[1][1]
        targets = batch[2][0][1:]  # exclude <s> from targets
        extended_tgt_batch = batch[2][1][1:]  # exclude <s> from targets
        tgt_mask = targets.eq(s2s.Constants.PAD).float()

        """
        def generate_copy_loss_function(g_outputs, c_gate_values, c_outputs, g_targets, tgt_mask,
                            extended_src_batch, extended_tgt_batch, extended_vocab_size,
                            generator, crit):
        """
        loss, res_loss, num_correct = generate_copy_loss_function(
            g_outputs, c_gate_values, c_outputs, targets, tgt_mask,
            extended_src_batch, extended_tgt_batch, extended_vocab_size,
            model.generator, criterion)
        if opt.use_coverage:
            coverage_loss = coverage_loss_function(all_coverage, all_attn, tgt_mask)
            loss = loss + coverage_loss

        if math.isnan(res_loss) or res_loss > 1e20:
            logger.info('catch NaN')
            ipdb.set_trace()
        loss_value = res_loss
        num_words = targets.data.ne(s2s.Constants.PAD).sum().item()
        num_src_words = batch[0][-1].data.sum()
        return loss, loss_value, num_words, num_src_words, num_correct

    def _train_one_epoch(self):
        logger, opt = self.get_logger_and_opt()
        train_data = self.train_data
        model = self.model
        model.train()
        criterion = nn.NLLLoss(size_average=False, reduce=False)
        stat = Statistics()
        start_time = time.time()
        if opt.extra_shuffle and self.num_epoch > opt.curriculum:
            logger.info('Shuffling...')
            train_data.shuffle()

        # shuffle mini batch order
        batch_order = torch.randperm(len(train_data))

        for i in range(len(train_data)):
            self.num_batch += 1
            """
            (wrap(srcBatch), lengths), \
               (simple_wrap(extended_src_batch), max(extended_vocab_size)), \
               (wrap(tgtBatch), wrap(extended_tgt_batch),), \
               indices
            """
            batch_idx = batch_order[i] if self.num_epoch > opt.curriculum else i
            batch = train_data[batch_idx][:-1]  # exclude original indices

            loss, loss_value, num_words, num_src_words, num_correct = self._forward_one_batch(batch, criterion)

            # update the parameters
            loss.backward()
            self.optim.step()

            stat.update(loss_value, num_src_words, num_words, num_correct)
            if i % opt.log_interval == -1 % opt.log_interval:
                report_string = stat.to_string(self.num_epoch, i, len(train_data), self.num_batch, start_time)
                logger.info(report_string)

            if opt.test_during_training and self.num_batch % opt.eval_per_batch == -1 % opt.eval_per_batch \
                    and self.num_batch >= opt.start_eval_batch:
                model.eval()
                logger.warning("Set model to {0} mode".format('train' if model.training else 'eval'))
                tune_score, all_scores, system_outputs, normed_output = self.evaluator.evaluate()
                model.train()
                logger.warning("Set model to {0} mode".format('train' if model.training else 'eval'))
                logger.info('Validation Scores: {0}'.format(str(all_scores)))
                logger.info('Validation Tune Score: {0}'.format(tune_score))
                if tune_score >= self.optim.best_metric:
                    dump_path = self.get_dev_dump_path(tune_score)
                    self.save_model(dump_path)
                self.optim.updateLearningRate(tune_score, self.num_epoch)
        self.num_epoch += 1
        return stat

    def _forward_dev_set(self):
        logger, opt = self.get_logger_and_opt()
        train_data = self.dev_data
        model = self.model
        model.eval()
        criterion = nn.NLLLoss(size_average=False, reduce=False)
        stat = Statistics()
        for i in range(len(train_data)):
            self.num_batch += 1
            batch = train_data[i][:-1]  # exclude original indices
            loss, loss_value, num_words, num_src_words, num_correct = self._forward_one_batch(batch, criterion)
            stat.update(loss_value, num_src_words, num_words, num_correct)
        return stat

    def train(self):
        logger, opt = self.get_logger_and_opt()
        model = self.model

        logger.info(model)
        model.train()
        logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))

        for epoch in range(opt.start_epoch, opt.epochs + 1):
            logger.info('')
            #  (1) train for one epoch on the training set
            train_stat = self._train_one_epoch()
            logger.info('Train perplexity: %g' % train_stat.ppl())
            logger.info('Train accuracy: %g' % (train_stat.accuracy() * 100))
            logger.info('Saving checkpoint for epoch {0}...'.format(epoch))
            save_path = self.get_regular_epoch_dump_path()
            self.save_model(save_path)
            if self.dev_data is not None:
                dev_stat = self._forward_dev_set()
                dev_ppl = dev_stat.ppl()
                logger.info('Dev perplexity: %g' % dev_ppl)
                logger.info('Dev accuracy: %g' % (dev_stat.accuracy() * 100))
                logger.info('Saving checkpoint for epoch {0}...'.format(epoch))
                save_path = self.get_dev_dump_path(dev_ppl)
                self.save_model(save_path)
                self.optim.updateLearningRate(dev_ppl, self.num_epoch)
