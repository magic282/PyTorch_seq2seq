from __future__ import division

# Class for managing the internals of the beam search process.
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

import torch
import s2s

try:
    import ipdb
except ImportError:
    pass


class Beam(object):
    def __init__(self, size, vocab_size, bottom_up_coverage_penalty, length_penalty, cuda=False):

        self.size = size
        self.vocab_size = vocab_size
        self.bottom_up_coverage_penalty = bottom_up_coverage_penalty
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []
        self.all_length = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(s2s.Constants.PAD)]
        self.nextYs[0][0] = s2s.Constants.BOS
        self.nextYs_true = [self.tt.LongTensor(size).fill_(s2s.Constants.PAD)]
        self.nextYs_true[0][0] = s2s.Constants.BOS

        # The attentions (matrix) for each time.
        self.attn = []

        # is copy for each time
        self.isCopy = []

        # for bottom up coverage penalty
        self.prev_attn_sum = None

        if length_penalty == 'avg':
            def avg_panelty(cur_length, now_acc_score):
                p = cur_length.unsqueeze(1).expand_as(now_acc_score)
                return p
            self.length_penalty_func = avg_panelty
        elif length_penalty == 'wu':
            def wu_penaty(cur_length, now_acc_score):
                p = torch.pow((5 + cur_length.unsqueeze(1).expand_as(now_acc_score)) / 6, 0.9)
                return p
            self.length_penalty_func = wu_penaty
        else:
            raise ValueError('No length penalty given')

    # Get the outputs for the current timestep.
    def getCurrentState(self):
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def getCurrentOrigin(self):
        return self.prevKs[-1]

    def get_bottom_up_coverage_penalty(self, attn):
        if self.prev_attn_sum is None:
            cov = attn
        else:
            cov = self.prev_attn_sum + attn
        penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(1)
        penalty -= cov.size(1)
        self.prev_attn_sum = cov
        return penalty * 5

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.
    def advance(self, wordLk, attnOut):
        vocab_size = self.vocab_size
        numAll = wordLk.size(1)

        # self.length += 1  # TODO: some is finished so do not acc length for them
        if len(self.prevKs) > 0:
            finish_index = self.nextYs[-1].eq(s2s.Constants.EOS)
            if any(finish_index):
                # wordLk.masked_fill_(finish_index.unsqueeze(1).expand_as(wordLk), -float('inf'))
                wordLk.masked_fill_(finish_index.unsqueeze(1).expand_as(wordLk), -float('inf'))
                for i in range(self.size):
                    if self.nextYs[-1][i] == s2s.Constants.EOS:
                        wordLk[i][s2s.Constants.EOS] = 0
            # set up the current step length
            cur_length = self.all_length[-1]
            for i in range(self.size):
                cur_length[i] += 0 if self.nextYs[-1][i] == s2s.Constants.EOS else 1

        # Sum the previous scores.
        if self.bottom_up_coverage_penalty:
            bottom_up_coverage_penalty = self.get_bottom_up_coverage_penalty(attnOut)

        if len(self.prevKs) > 0:
            prev_score = self.all_scores[-1]
            now_acc_score = wordLk + prev_score.unsqueeze(1).expand_as(wordLk)
            length_penalty = self.length_penalty_func(cur_length, now_acc_score)
            beamLk = now_acc_score / length_penalty
            if self.bottom_up_coverage_penalty:
                beamLk -= bottom_up_coverage_penalty.unsqueeze(1)
        else:
            self.all_length.append(self.tt.FloatTensor(self.size).fill_(1))
            # beamLk = wordLk[0]
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numAll
        # predict = bestScoresId - prevK * numWords
        predict = bestScoresId - prevK * numAll
        isCopy = predict.ge(self.tt.LongTensor(self.size).fill_(self.vocab_size)).long()
        final_predict = predict * (1 - isCopy) + isCopy * s2s.Constants.UNK

        if len(self.prevKs) > 0:
            self.all_length.append(cur_length.index_select(0, prevK))
            self.all_scores.append(now_acc_score.view(-1).index_select(0, bestScoresId))
            if self.bottom_up_coverage_penalty:
                self.prev_attn_sum = self.prev_attn_sum.index_select(0, prevK)
        else:
            self.all_scores.append(self.scores)

        self.prevKs.append(prevK)
        self.nextYs.append(final_predict)
        self.nextYs_true.append(predict)
        self.isCopy.append(isCopy)
        self.attn.append(attnOut.index_select(0, prevK))

        # End condition is when every one is EOS.
        if all(self.nextYs[-1].eq(s2s.Constants.EOS)):
            self.done = True

        return self.done

    def sortBest(self):
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def getBest(self):
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def getHyp(self, k):
        hyp, attn = [], []
        isCopy, copyPos = [], []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            attn.append(self.attn[j][k])
            isCopy.append(self.isCopy[j][k])
            copyPos.append(self.nextYs_true[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1],  isCopy[::-1], copyPos[::-1], torch.stack(attn[::-1])
