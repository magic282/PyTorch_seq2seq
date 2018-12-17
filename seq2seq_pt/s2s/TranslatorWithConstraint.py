import s2s
import torch.nn as nn
import torch
from torch.autograd import Variable
from s2s.data_utils import src_to_ids, tgt_to_ids

try:
    import ipdb
except ImportError:
    pass


class TranslatorWithConstraint(object):
    def __init__(self, opt, model=None, dataset=None):
        self.opt = opt

        if model is None:

            checkpoint = torch.load(opt.model)

            model_opt = checkpoint['opt']
            self.src_dict = checkpoint['dicts']['src']
            self.tgt_dict = checkpoint['dicts']['tgt']

            self.enc_rnn_size = model_opt.enc_rnn_size
            self.dec_rnn_size = model_opt.dec_rnn_size
            encoder = s2s.Models.Encoder(model_opt, self.src_dict)
            decoder = s2s.Models.Decoder(model_opt, self.tgt_dict)
            decIniter = s2s.Models.DecInit(model_opt)
            model = s2s.Models.NMTModel(encoder, decoder, decIniter)

            generator = nn.Sequential(
                nn.Linear(model_opt.dec_rnn_size // model_opt.maxout_pool_size, self.tgt_dict.size()),
                nn.Softmax())  # TODO pay attention here

            model.load_state_dict(checkpoint['model'])
            generator.load_state_dict(checkpoint['generator'])

            if opt.cuda:
                model.cuda()
                generator.cuda()
            else:
                model.cpu()
                generator.cpu()

            model.generator = generator
        else:
            self.src_dict = dataset['dicts']['src']
            self.tgt_dict = dataset['dicts']['tgt']

            self.enc_rnn_size = opt.enc_rnn_size
            self.dec_rnn_size = opt.dec_rnn_size
            self.opt.cuda = True if len(opt.gpus) >= 1 else False
            self.opt.n_best = 1
            self.opt.replace_unk = False

        self.tt = torch.cuda if opt.cuda else torch
        self.model = model
        self.model.eval()

        self.copyCount = 0

    def buildData(self, srcBatch, goldBatch):
        srcData = [self.src_dict.convertToIdx(b,
                                              s2s.Constants.UNK_WORD) for b in srcBatch]
        extended_src_ids_data, src_oovs_data = map(list, zip(*[src_to_ids(b, self.src_dict) for b in srcBatch]))
        extended_src_ids_data = list(map(torch.LongTensor, extended_src_ids_data))
        extended_vocab_size = list(map(len, src_oovs_data))
        tgtData = None
        extend_tgt_ids_data = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                                                  s2s.Constants.UNK_WORD,
                                                  s2s.Constants.BOS_WORD,
                                                  s2s.Constants.EOS_WORD) for b in goldBatch]
            extend_tgt_ids_data = [
                tgt_to_ids(b, self.tgt_dict, src_oovs_data[idx], s2s.Constants.BOS_WORD, s2s.Constants.EOS_WORD) for
                idx, b in enumerate(goldBatch)]
            extend_tgt_ids_data = list(map(torch.LongTensor, extend_tgt_ids_data))

        return s2s.Dataset(srcData, tgtData, extended_src_ids_data, extend_tgt_ids_data, extended_vocab_size,
                           self.opt.batch_size, self.opt.cuda), src_oovs_data

    def buildTargetTokens(self, pred, src, src_oov, isCopy, copyPosition, attn):
        pred_word_ids = [x.item() for x in pred]
        tokens = self.tgt_dict.convertToLabels(pred_word_ids, s2s.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        copied = False
        for i in range(len(tokens)):
            if isCopy[i]:
                tokens[i] = '[[{0}]]'.format(src_oov[copyPosition[i] - self.tgt_dict.size()])
                copied = True
        if copied:
            self.copyCount += 1
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == s2s.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, srcBatch, extBatch, tagBatch):
        batchSize = srcBatch[0].size(1)
        beamSize = self.opt.beam_size
        extended_src_batch = extBatch[0]
        extended_vocab_size = extBatch[1]
        extend_zeros = None
        if extended_vocab_size > 0:
            extend_zeros = torch.zeros((beamSize * batchSize, extended_vocab_size)).to(srcBatch[0].device)
            extended_src_batch = extended_src_batch.repeat(beamSize, 1)

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(srcBatch)
        srcBatch = srcBatch[0]  # drop the lengths needed for encoder

        decStates = self.model.decIniter(encStates[1])  # batch, dec_hidden

        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        context = context.data.repeat(1, beamSize, 1)
        decStates = decStates.unsqueeze(0).data.repeat(1, beamSize, 1)
        att_vec = self.model.make_init_att(context)
        padMask = srcBatch.data.eq(s2s.Constants.PAD).transpose(0, 1).unsqueeze(0).repeat(beamSize, 1, 1).float()

        cur_coverage = torch.zeros((context.size(1), context.size(0))).to(context.device)  # (beam*batch, seq)

        beam = [s2s.Beam(beamSize, self.tgt_dict.size(), self.opt.cuda) for k in range(batchSize)]
        batchIdx = list(range(batchSize))
        remainingSents = batchSize

        tags = torch.FloatTensor(tagBatch[0]).to(padMask.device)

        for i in range(self.opt.max_sent_length):
            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).transpose(0, 1).contiguous().view(1, -1)
            g_outputs, c_outputs, copyGateOutputs, decStates, attn, coverage, att_vec = \
                self.model.decoder(input, decStates, context, padMask.view(-1, padMask.size(2)), att_vec, cur_coverage)

            g_outputs = g_outputs[0]
            c_outputs = c_outputs[0]

            # c_outputs = torch.mul(c_outputs, tags)
            # renorm = torch.sum(c_outputs, dim=1)
            # c_outputs = c_outputs / renorm.view(renorm.size(0), 1).expand_as(c_outputs)

            attn = attn[0]
            coverage = coverage[0]
            copyGateOutputs = copyGateOutputs[0]
            # g_outputs: 1 x (beam*batch) x numWords
            copyGateOutputs = copyGateOutputs.view(-1, 1)
            # g_outputs = g_outputs.squeeze(0)
            g_out_prob = self.model.generator.forward(g_outputs)
            g_out_prob = g_out_prob * ((1 - copyGateOutputs).expand_as(g_out_prob))
            # c_outputs = c_outputs.squeeze(0)
            c_prob = c_outputs * (copyGateOutputs.expand_as(c_outputs))  # (beam, 400)

            c_prob = torch.mul(c_prob, tags) * 2

            if extended_vocab_size > 0:
                extend_prob = torch.cat((g_out_prob, extend_zeros), 1)
                extend_prob = extend_prob.scatter_add(1, extended_src_batch, c_prob)
            else:
                extend_prob = g_out_prob

            extend_prob = extend_prob + 1e-8
            log_extend_prob = torch.log(extend_prob)
            # batch x beam x numWords
            log_extend_prob = log_extend_prob.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            father_idx = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(log_extend_prob.data[idx], attn.data[idx]):
                    active += [b]
                    father_idx.append(beam[b].prevKs[-1])  # this is very annoying

            if not active:
                break

            # to get the real father index
            real_father_idx = []
            for kk, idx in enumerate(father_idx):
                real_father_idx.append(idx * len(father_idx) + kk)

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t, rnnSize):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return view.index_select(1, activeIdx).view(*newSize)

            decStates = updateActive(decStates, self.dec_rnn_size)
            context = updateActive(context, self.enc_rnn_size)
            att_vec = updateActive(att_vec, self.enc_rnn_size)
            padMask = padMask.index_select(1, activeIdx)
            if extend_zeros is not None:
                extend_zeros = extend_zeros.view(beamSize, remainingSents, -1)
                extend_zeros = extend_zeros.index_select(1, activeIdx)
                extend_zeros = extend_zeros.view(-1, extend_zeros.size(2))
                extended_src_batch = extended_src_batch.view(beamSize, remainingSents, -1)
                extended_src_batch = extended_src_batch.index_select(1, activeIdx)
                extended_src_batch = extended_src_batch.view(-1, extended_src_batch.size(2))
            cur_coverage = coverage.view(beamSize, remainingSents, -1)
            cur_coverage = cur_coverage.index_select(1, activeIdx)
            cur_coverage = cur_coverage.view(-1, cur_coverage.size(2))

            # set correct state for beam search
            previous_index = torch.stack(real_father_idx).transpose(0, 1).contiguous()
            decStates = decStates.view(-1, decStates.size(2)).index_select(0, previous_index.view(-1)).view(
                *decStates.size())
            att_vec = att_vec.view(-1, att_vec.size(1)).index_select(0, previous_index.view(-1)).view(*att_vec.size())
            cur_coverage = cur_coverage.view(-1, cur_coverage.size(1)).index_select(0, previous_index.view(-1)).view(
                *cur_coverage.size())


            remainingSents = len(active)

        # (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        allIsCopy, allCopyPosition = [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            valid_attn = srcBatch.data[:, b].ne(s2s.Constants.PAD).nonzero().squeeze(1)
            hyps, isCopy, copyPosition, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            allHyp += [hyps]
            allAttn += [attn]
            allIsCopy += [isCopy]
            allCopyPosition += [copyPosition]

        return allHyp, allScores, allIsCopy, allCopyPosition, allAttn, None

    def translate(self, srcBatch, goldBatch, tagBatch):
        #  (1) convert words to indexes
        dataset, src_oovs = self.buildData(srcBatch, goldBatch)
        """
        (wrap(srcBatch), lengths), \
               (simple_wrap(extended_src_batch), max(extended_vocab_size)), \
               (wrap(tgtBatch), wrap(extended_tgt_batch),), \
               indices
        """
        src, extend, tgt, indices = dataset[0]
        if len(tagBatch) > 1:
            raise ValueError("Now only support batch size 1. Fail.")
        new_tag_batch = [tagBatch[idx] for idx in indices]

        #  (2) translate
        pred, predScore, predIsCopy, predCopyPosition, attn, _ = self.translateBatch(src, extend, new_tag_batch)
        pred, predScore, predIsCopy, predCopyPosition, attn = list(zip(
            *sorted(zip(pred, predScore, predIsCopy, predCopyPosition, attn, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], src_oovs[b],
                                        predIsCopy[b][n], predCopyPosition[b][n], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, None
