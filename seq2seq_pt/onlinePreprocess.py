import logging
import torch
import s2s
from s2s.data_utils import src_to_ids, tgt_to_ids

try:
    import ipdb
except ImportError:
    pass

lower = True
MAX_SRC_LENGTH = 100
MAX_TGT_LENGTH = 100
TRUNCATE = False
report_every = 100000
shuffle = 1

logger = logging.getLogger(__name__)


def makeVocabulary(filenames, size):
    vocab = s2s.Dict([s2s.Constants.PAD_WORD, s2s.Constants.UNK_WORD,
                      s2s.Constants.BOS_WORD, s2s.Constants.EOS_WORD], lower=lower)
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().split(' '):
                    vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    logger.info('Created dictionary of size %d (pruned from %d)' %
                (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        logger.info('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = s2s.Dict(lower=lower)
        vocab.loadFile(vocabFile)
        logger.info('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        logger.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    logger.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt = [], []
    extended_src, extended_tgt = [], []
    extend_vocab_size = []
    sizes = []
    count, ignored = 0, 0
    src_truncate_count, tgt_truncate_count = 0, 0

    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            logger.info('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        srcWords = sline.split(' ')
        tgtWords = tline.split(' ')

        if TRUNCATE:
            if len(srcWords) > MAX_SRC_LENGTH:
                srcWords = srcWords[:MAX_SRC_LENGTH]
                src_truncate_count += 1
            if len(tgtWords) > MAX_TGT_LENGTH:
                tgtWords = tgtWords[:MAX_TGT_LENGTH]
                tgt_truncate_count += 1

        if len(srcWords) <= MAX_SRC_LENGTH and len(tgtWords) <= MAX_TGT_LENGTH:
            src += [srcDicts.convertToIdx(srcWords,
                                          s2s.Constants.UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          s2s.Constants.UNK_WORD,
                                          s2s.Constants.BOS_WORD,
                                          s2s.Constants.EOS_WORD)]
            extended_src_ids, src_oovs = src_to_ids(srcWords, srcDicts)
            extend_tgt_ids = tgt_to_ids(tgtWords, tgtDicts, src_oovs, s2s.Constants.BOS_WORD, s2s.Constants.EOS_WORD)
            extended_src += [torch.LongTensor(extended_src_ids)]
            extended_tgt += [torch.LongTensor(extend_tgt_ids)]
            extend_vocab_size.append(len(src_oovs))

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if shuffle == 1:
        logger.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        extended_src = [extended_src[idx] for idx in perm]
        extended_tgt = [extended_tgt[idx] for idx in perm]
        extend_vocab_size = [extend_vocab_size[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    logger.info('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    extended_src = [extended_src[idx] for idx in perm]
    extended_tgt = [extended_tgt[idx] for idx in perm]
    extend_vocab_size = [extend_vocab_size[idx] for idx in perm]

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(src), ignored, MAX_SRC_LENGTH))
    logger.info('{0} source truncated'.format(src_truncate_count))
    logger.info('{0} target truncated'.format(tgt_truncate_count))
    return src, tgt, extended_src, extended_tgt, extend_vocab_size


def prepare_data_online(train_src, src_vocab, train_tgt, tgt_vocab):
    dicts = {}
    dicts['src'] = initVocabulary('source', [train_src], src_vocab, 0)
    dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, 0)

    logger.info('Preparing training ...')
    train = {}
    train['src'], train['tgt'], \
    train['extended_src'], train['extended_tgt'], train['extend_vocab_size'] = makeData(train_src,
                                                                                        train_tgt,
                                                                                        dicts['src'],
                                                                                        dicts['tgt'])

    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
               }
    return dataset
