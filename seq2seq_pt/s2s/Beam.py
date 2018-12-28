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
    def __init__(self, size, vocab_size,
                 global_scorer,
                 stepwise_penalty,
                 min_len=0,
                 cuda=False):

        self.size = size
        self.vocab_size = vocab_size
        self.done = False
        self.min_length = min_len

        self.tt = torch.cuda if cuda else torch

        self.finished = []

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []
        self.all_length = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size).fill_(s2s.Constants.PAD)]
        self.next_ys[0][0] = s2s.Constants.BOS
        self.next_ys_true = [self.tt.LongTensor(size).fill_(s2s.Constants.PAD)]
        self.next_ys_true[0][0] = s2s.Constants.BOS

        # The attentions (matrix) for each time.
        self.attn = []

        # is copy for each time
        self.isCopy = []

        # for bottom up coverage penalty
        self.prev_attn_sum = None

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}
        self.stepwise_penalty = stepwise_penalty

    # Get the outputs for the current timestep.
    def getCurrentState(self):
        return self.next_ys[-1]

    # Get the backpointers for the current timestep.
    def getCurrentOrigin(self):
        return self.prev_ks[-1]

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
    def advance(self, word_log_prob, attn_out):
        numAll = word_log_prob.size(1)

        if self.stepwise_penalty:
            self.global_scorer.update_score(self, attn_out)

        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_log_prob)):
                word_log_prob[k][s2s.Constants.EOS] = -1e20

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beamLk = word_log_prob + self.scores.unsqueeze(1).expand_as(word_log_prob)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == s2s.Constants.EOS:
                    beamLk[i] = -1e20
        else:
            beamLk = word_log_prob[0]

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

        self.prev_ks.append(prevK)
        self.next_ys.append(final_predict)
        self.next_ys_true.append(predict)
        self.isCopy.append(isCopy)
        self.attn.append(attn_out.index_select(0, prevK))
        self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == s2s.Constants.EOS:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when every one is EOS.
        # if all(self.nextYs[-1].eq(s2s.Constants.EOS)):
        #     self.done = True
        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == s2s.Constants.EOS:
            self.all_scores.append(self.scores)
            self.done = True

        return self.done

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1
        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

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
    def getHyp(self, time_step, k):
        hyp, attn = [], []
        isCopy, copyPos = [], []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prev_ks[:time_step]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])
            isCopy.append(self.isCopy[j][k])
            copyPos.append(self.next_ys_true[j + 1][k])
            k = self.prev_ks[j][k]

        return hyp[::-1], isCopy[::-1], copyPos[::-1], torch.stack(attn[::-1])


class PenaltyBuilder(object):
    """
    Returns the Length and Coverage Penalty function for Beam Search.

    Args:
        length_pen (str): option name of length pen
        cov_pen (str): option name of cov pen
    """

    def __init__(self, cov_pen, length_pen):
        self.length_pen = length_pen
        self.cov_pen = cov_pen

    def coverage_penalty(self):
        if self.cov_pen == "wu":
            return self.coverage_wu
        elif self.cov_pen == "summary":
            return self.coverage_summary
        else:
            return self.coverage_none

    def length_penalty(self):
        if self.length_pen == "wu":
            return self.length_wu
        elif self.length_pen == "avg":
            return self.length_average
        else:
            return self.length_none

    """
    Below are all the different penalty terms implemented so far
    """

    def coverage_wu(self, beam, cov, beta=0.):
        """
        NMT coverage re-ranking score from
        "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """
        penalty = -torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        return beta * penalty

    def coverage_summary(self, beam, cov, beta=0.):
        """
        Our summary penalty.
        """
        penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(1)
        penalty -= cov.size(1)
        return beta * penalty

    def coverage_none(self, beam, cov, beta=0.):
        """
        returns zero as penalty
        """
        return beam.scores.clone().fill_(0.0)

    def length_wu(self, beam, logprobs, alpha=0.):
        """
        NMT length re-ranking score from
        "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        modifier = (((5 + len(beam.next_ys)) ** alpha) /
                    ((5 + 1) ** alpha))
        return (logprobs / modifier)

    def length_average(self, beam, logprobs, alpha=0.):
        """
        Returns the average probability of tokens in a sequence.
        """
        return logprobs / len(beam.next_ys)

    def length_none(self, beam, logprobs, alpha=0., beta=0.):
        """
        Returns unmodified scores.
        """
        return logprobs


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha, beta, cov_penalty, length_penalty):
        self.alpha = alpha
        self.beta = beta
        penalty_builder = PenaltyBuilder(cov_penalty, length_penalty)
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty()
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):
        """
        Function to update scores of a Beam that is not finished
        """
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        """
        Keeps the coverage vector as sum of attentions
        :param beam:
        :return:
        """
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
            beam.global_state["coverage"] = beam.attn[-1]
        else:
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

            prev_penalty = self.cov_penalty(beam,
                                            beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty
