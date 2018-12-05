from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import s2s


class Dataset(object):
    def __init__(self, srcData, tgtData, extended_src_data, extended_tgt_data, extend_vocab_size_data, batchSize, cuda):
        self.src = srcData
        self.extended_src = extended_src_data
        self.extend_vocab_size = extend_vocab_size_data
        if tgtData:
            self.tgt = tgtData
            # copy switch should company tgt label
            self.extended_tgt = extended_tgt_data
            assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None
            self.extended_tgt = None
        self.device = torch.device("cuda" if cuda else "cpu")

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(s2s.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)
        extended_src_batch = self._batchify(
            self.extended_src[index * self.batchSize:(index + 1) * self.batchSize])
        extended_vocab_size = self.extend_vocab_size[index * self.batchSize:(index + 1) * self.batchSize]

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index * self.batchSize:(index + 1) * self.batchSize])
            extended_tgt_batch = self._batchify(
                self.extended_tgt[index * self.batchSize:(index + 1) * self.batchSize])

        else:
            tgtBatch = None
            extended_tgt_batch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        if tgtBatch is None:
            batch = zip(indices, srcBatch, extended_src_batch, extended_vocab_size)
        else:
            batch = zip(indices, srcBatch, extended_src_batch, extended_vocab_size, tgtBatch, extended_tgt_batch)
        # batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch, extended_src_batch, extended_vocab_size = zip(*batch)
        else:
            indices, srcBatch, extended_src_batch, extended_vocab_size, tgtBatch, extended_tgt_batch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            b = b.to(self.device)
            return b

        def simple_wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).contiguous()
            b = b.to(self.device)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)

        return (wrap(srcBatch), lengths), \
               (simple_wrap(extended_src_batch), max(extended_vocab_size)), \
               (wrap(tgtBatch), wrap(extended_tgt_batch),), \
               indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(
            zip(self.src, self.tgt, self.extended_src, self.extended_tgt, self.extend_vocab_size))
        self.src, self.tgt, self.extended_src, self.extended_tgt, self.extend_vocab_size = zip(
            *[data[i] for i in torch.randperm(len(data))])
