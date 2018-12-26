import sys
import time
import math


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def reset(self):
        self.loss = 0
        self.n_words = 0
        self.n_correct = 0
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, loss: float, n_src_words: int, n_words: int, n_correct: int):
        self.loss += loss
        self.n_src_words += n_src_words
        self.n_words += n_words
        self.n_correct += n_correct

    def accuracy(self):
        return self.n_correct / self.n_words

    def cross_entropy(self):
        return self.loss / self.n_words

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 16))

    def elapsed_time(self):
        return time.time() - self.start_time

    def to_string(self, epoch, batch, dataset_batch_num, acc_batch_num, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           total_batch_num (int): total batches
           start: start time of epoch.
        """
        t = self.elapsed_time()
        s = ("Epoch %2d, %5d/%5d/%6d; acc: %6.2f; ppl: %6.2f; "
             + "xent: %6.2f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") % (
            epoch, batch, dataset_batch_num,  acc_batch_num,
            self.accuracy() * 100,
            self.ppl(),
            self.cross_entropy(),
            self.n_src_words / (t + 1e-5),
            self.n_words / (t + 1e-5),
            time.time() - start)
        return s

    # def log(self, prefix, experiment, lr):
    #     t = self.elapsed_time()
    #     experiment.add_scalar_value(prefix + "_ppl", self.ppl())
    #     experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
    #     experiment.add_scalar_value(prefix + "_tgtper", self.n_words / t)
    #     experiment.add_scalar_value(prefix + "_lr", lr)
    #
    # def log_tensorboard(self, prefix, writer, lr, step):
    #     t = self.elapsed_time()
    #     writer.add_scalar(prefix + "/xent", self.xent(), step)
    #     writer.add_scalar(prefix + "/ppl", self.ppl(), step)
    #     writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
    #     writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
    #     writer.add_scalar(prefix + "/lr", lr, step)
