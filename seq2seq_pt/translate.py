import torch
import argparse
import math
import time
import logging

import s2s
import xargs
from PyRouge.Rouge import Rouge


def translate_init():
    logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
    file_handler = logging.FileHandler(time.strftime("%Y%m%d-%H%M%S") + '.log.txt', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
    logging.root.addHandler(file_handler)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='translate.py')
    xargs.add_translate_options(parser)
    opt = parser.parse_args()
    return logger, opt


def main():
    logger, opt = translate_init()
    logger.info(opt)
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = s2s.Translator(opt)
    src_file = opt.src
    ref_file = None if opt.tgt is None else opt.tgt

    system_outputs = translator.translate_small_file(src_file, ref_file, opt.batch_size)
    with open(opt.output, 'w', encoding='utf-8') as writer:
        for line in system_outputs:
            writer.write(line.strip() + '\n')

    if ref_file is not None:
        normed_output = [x.replace('<t>', '').replace('</t>', '').replace('[[', '').replace(']]', '') for x in
                         system_outputs]
        refs = []
        with open(ref_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                refs.append(line.strip().replace('<t>', '').replace('</t>', '').replace('[[', '').replace(']]', ''))
        rouge_calculator = Rouge.Rouge()
        scores = rouge_calculator.compute_rouge(refs, normed_output)
        logger.info(str(scores))


if __name__ == "__main__":
    main()
