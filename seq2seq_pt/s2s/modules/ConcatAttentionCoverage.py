import torch
import torch.nn as nn
import torch.nn.functional as F
import math


try:
    import ipdb
except ImportError:
    pass


class ConcatAttentionCoverage(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim, use_coverage=False):
        super(ConcatAttentionCoverage, self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)
        self.use_coverage = use_coverage
        if self.use_coverage:
            self.linear_cov = nn.Linear(1, att_dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, coverage_acc=None, precompute=None):
        """
        input: (batch, dim)
        context: (batch, sourceL, dim)
        coverage_acc: (batch, sourceL)
        """
        if precompute is None:
            precompute00 = self.linear_pre(context.contiguous().view(-1, context.size(2)))
            precompute = precompute00.view(context.size(0), context.size(1), -1)  # (batch, sourceL, att_dim)
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim
        tmp10 = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim
        if self.use_coverage:
            tmp10 = tmp10 + self.linear_cov(coverage_acc.view(-1, 1)).view_as(tmp10)
        tmp20 = torch.tanh(tmp10)  # (batch, sourceL, att_dim)
        energy = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL
        if self.mask is not None:
            energy = energy * (1 - self.mask) + self.mask * (-1000000)
        score = self.sm(energy)  # (batch, sourceL)
        score_m = score.view(score.size(0), 1, score.size(1))  # (batch, 1, sourceL)

        if self.use_coverage:
            coverage_acc = coverage_acc + score  # update coverage_acc

        weightedContext = torch.bmm(score_m, context).squeeze(1)  # (batch, dim)

        return weightedContext, score, coverage_acc, precompute

    # def extra_repr(self):
    #     return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
    #            + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
    #            + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'
    #
    # def __repr__(self):
    #     return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
    #            + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
    #            + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'
