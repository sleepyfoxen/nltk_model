from nltk.probability import WittenBellProbDist
from .ngram import NgramModel


class LgramModel(NgramModel):

    @staticmethod
    def _estimator(fdist, bins):
        """
        Default estimator function using WB.
        """
        # can't be an instance method of NgramModel as they
        # can't be pickled either.
        res = WittenBellProbDist(fdist, fdist.B() + 1)
        return res

    def __init__(self, n, train, pad_left=False, pad_right=False,
                 estimator=None, *estimator_args, **estimator_kwargs):
        """
        NgramModel (q.v.) slightly tweaked to produce char-grams,
        not word-grams, with a WittenBell default estimator

        :param train: List of strings, which will be converted to list of lists of characters, but more efficiently
        :type train: iter(str)
        """
        if estimator is None:
            assert (not (estimator_args)) and (not (estimator_kwargs)), \
                "estimator_args (%s) or _kwargs (%s) supplied, but no estimator" % (estimator_args, estimator_kwargs)
            estimator = self._estimator
        super(LgramModel, self).__init__(n,
                                         (iter(word) for word in train),
                                         pad_left, pad_right,
                                         estimator,
                                         *estimator_args, **estimator_kwargs)
