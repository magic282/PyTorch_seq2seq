import os
import argparse
import torch
from torch import cuda
import time
import logging

try:
    import ipdb
except ImportError:
    pass

import s2s
import xargs


def init_train():
    parser = argparse.ArgumentParser(description='train.py')
    xargs.add_data_options(parser)
    xargs.add_model_options(parser)
    xargs.add_train_options(parser)
    xargs.add_dev_options(parser)

    opt = parser.parse_args()

    logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
    log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
    if opt.log_home:
        if not os.path.exists(opt.log_home):
            os.makedirs(opt.log_home)
        log_file_name = os.path.join(opt.log_home, log_file_name)
    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
    logging.root.addHandler(file_handler)
    logger = logging.getLogger(__name__)

    logger.info('My PID is {0}'.format(os.getpid()))
    logger.info('PyTorch version: {0}'.format(str(torch.__version__)))
    logger.info(opt)

    if torch.cuda.is_available() and not opt.gpus:
        logger.info("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    if opt.gpus:
        if opt.cuda_seed > 0:
            torch.cuda.manual_seed(opt.cuda_seed)
        cuda.set_device(opt.gpus[0])

    logger.info('My seed is {0}'.format(torch.initial_seed()))
    logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))
    return logger, opt


def main():
    logger, opt = init_train()
    trainer = s2s.Trainer(logger, opt)
    trainer.prepare_data()
    trainer.build_model()
    trainer.build_optimizer()
    trainer.train()


if __name__ == "__main__":
    main()
