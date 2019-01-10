# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2009 NLTK Project
# Author: Steven Bird <sb@csse.unimelb.edu.au>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT

import random, collections.abc
from itertools import chain
from math import log

from nltk.probability import (ConditionalProbDist, ConditionalFreqDist,
                              MLEProbDist, FreqDist, WittenBellProbDist)

from nltk.util import ngrams as ingrams

from nltk import compat
try:
    from api import *
except ImportError:
    from .api import *

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def discount(self):
    return float(self._N)/float(self._N + self._T)

def check(self):
    totProb=sum(self.prob(sample) for sample in self.samples())
    assert isclose(self.discount(),totProb),\
           "discount %s != totProb %s"%(self.discount(),totProb)

WittenBellProbDist.discount = discount
WittenBellProbDist.check = check

def _estimator(fdist, bins):
    """
    Default estimator function using WB.
    """
    # can't be an instance method of NgramModel as they
    # can't be pickled either.
    res=WittenBellProbDist(fdist,fdist.B()+1)
    res.check()
    return res

@compat.python_2_unicode_compatible
class NgramModel(ModelI):
    """
    A processing interface for assigning a probability to the next word.
    """

    def __init__(self, n, train, pad_left=False, pad_right=False,
                 estimator=None, *estimator_args, **estimator_kwargs):
        """
        Creates an ngram language model to capture patterns in n consecutive
        words of training text.  An estimator smooths the probabilities derived
        from the text and may allow generation of ngrams not seen during
        training.

        :param n: the order of the language model (ngram size)
        :type n: C{int}
        :param train: the training text
        :type train: C{iterable} of C{string} or C{iterable} of C{iterable} of C{string} 
        :param estimator: a function for generating a probability distribution---defaults to MLEProbDist
        :type estimator: a function that takes a C{ConditionalFreqDist} and
              returns a C{ConditionalProbDist}
        :param pad_left: whether to pad the left of each sentence with an (n-1)-gram of <s>
        :type pad_left: bool
        :param pad_right: whether to pad the right of each sentence with </s>
        :type pad_right: bool
        :param estimator_args: Extra arguments for estimator.
            These arguments are usually used to specify extra
            properties for the probability distributions of individual
            conditions, such as the number of bins they contain.
            Note: For backward-compatibility, if no arguments are specified, the
            number of bins in the underlying ConditionalFreqDist are passed to
            the estimator as an argument.
        :type estimator_args: (any)
        :param estimator_kwargs: Extra keyword arguments for the estimator
        :type estimator_kwargs: (any)
        """

        # protection from cryptic behavior for calling programs
        # that use the pre-2.0.2 interface
        assert(isinstance(pad_left, bool))
        assert(isinstance(pad_right, bool))

        # make sure n is greater than zero, otherwise print it
        assert (n > 0), n

        # For explicitness save the check whether this is a unigram model
        self.is_unigram_model = (n == 1)
        # save the ngram order number
        self._n = n
        # save left and right padding
        self._lpad = ('<s>',) * (n - 1) if pad_left else ()
        # Need _rpad even for unigrams or padded entropy will give
        #  wrong answer because '</s>' will be treated as unseen...
        self._rpad = ('</s>',) if pad_right else ()
        self._padLen = len(self._lpad)+len(self._rpad)

        self._N=0
        delta = 1+self._padLen-n        # len(sent)+delta == ngrams in sent

        if estimator is None:
            assert (estimator_args is ()) and (estimator_kwargs=={}),\
                   "estimator_args (%s) or _kwargs supplied (%s), but no estimator"%(estimator_args,estimator_kwargs)
            estimator = lambda fdist, bins: MLEProbDist(fdist)

        # Given backoff, a generator isn't acceptable
        if not isinstance(train,collections.abc.Sequence):
          train=list(train)
        self._W = len(train)
        # Coerce to list of list -- note that this means to train charGrams,
        #  requires exploding the words ahead of time 
        if train is not None:
            if isinstance(train[0], compat.string_types):
                train = [train]
                self._W=1
            elif not isinstance(train[0],collections.abc.Sequence):
                # if you mix strings and generators, you have only yourself
                #  to blame!
                for i in range(len(train)):
                    train[i]=list(train[i])

        if n == 1:
            if pad_right:
                sents=(chain(s,self._rpad) for s in train)
            else:
                sents=train
            fd=FreqDist()
            for s in sents:
                fd.update(s)
            if not estimator_args and not estimator_kwargs:
                self._model = estimator(fd,fd.B())
            else:
                self._model = estimator(fd,fd.B(),
                                        *estimator_args, **estimator_kwargs)
            self._N=fd.N()
        else:
            cfd = ConditionalFreqDist()
            self._ngrams = set()

            for sent in train:
                self._N+=len(sent)+delta
                for ngram in ingrams(chain(self._lpad, sent, self._rpad), n):
                    self._ngrams.add(ngram)
                    context = tuple(ngram[:-1])
                    token = ngram[-1]
                    cfd[context][token]+=1
            if not estimator_args and not estimator_kwargs:
                self._model = ConditionalProbDist(cfd, estimator, len(cfd))
            else:
                self._model = ConditionalProbDist(cfd, estimator, *estimator_args, **estimator_kwargs)

        # recursively construct the lower-order models
        if not self.is_unigram_model:
            self._backoff = NgramModel(n-1, train,
                                        pad_left, pad_right,
                                        estimator,
                                        *estimator_args,
                                        **estimator_kwargs)

            # Code below here in this method, and the _words_following and _alpha method, are from
            # http://www.nltk.org/_modules/nltk/model/ngram.html "Last updated on Feb 26, 2015"
            self._backoff_alphas = dict()
            # For each condition (or context)
            for ctxt in cfd.conditions():
                backoff_ctxt = ctxt[1:]
                backoff_total_pr = 0.0
                total_observed_pr = 0.0

                # this is the subset of words that we OBSERVED following
                # this context.
                # i.e. Count(word | context) > 0
                for word in self._words_following(ctxt, cfd):
                    total_observed_pr += self.prob(word, ctxt)
                    # we also need the total (n-1)-gram probability of
                    # words observed in this n-gram context
                    backoff_total_pr += self._backoff.prob(word, backoff_ctxt)
                if isclose(total_observed_pr,1.0):
                    total_observed_pr=1.0
                else:
                    assert 0.0 <= total_observed_pr <= 1.0,\
                           "sum of probs for %s out of bounds: %.10g"%(ctxt,total_observed_pr)
                # beta is the remaining probability weight after we factor out
                # the probability of observed words.
                # As a sanity check, both total_observed_pr and backoff_total_pr
                # must be GE 0, since probabilities are never negative
                beta = 1.0 - total_observed_pr

                if beta!=0.0:
                    assert (0.0 <= backoff_total_pr < 1.0), \
                           "sum of backoff probs for %s out of bounds: %s"%(ctxt,backoff_total_pr)
                    alpha_ctxt = beta / (1.0 - backoff_total_pr)
                else:
                    assert ((0.0 <= backoff_total_pr < 1.0) or
                            isclose(1.0,backoff_total_pr)), \
                           "sum of backoff probs for %s out of bounds: %s"%(ctxt,backoff_total_pr)
                    alpha_ctxt = 0.0

                self._backoff_alphas[ctxt] = alpha_ctxt

    def _words_following(self, context, cond_freq_dist):
        return cond_freq_dist[context].keys()

    def prob(self, word, context, verbose=False):
        """
        Evaluate the probability of this word in this context using Katz Backoff.

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """
        assert(isinstance(word,compat.string_types))
        context = tuple(context)
        if self.is_unigram_model:
            if not(self._model.SUM_TO_ONE):
                # Smoothing models should do the right thing for unigrams
                #  even if they're 'absent'
                return self._model.prob(word)
            else:
                try:
                    return self._model.prob(word)
                except:
                    raise RuntimeError("No probability mass assigned"
                                       "to unigram %s" % (word))
        if context + (word,) in self._ngrams:
            return self[context].prob(word)
        else:
            alpha=self._alpha(context)
            if alpha>0.0:
                if verbose:
                    print("backing off for %s"%(context+(word,),))
                return alpha * self._backoff.prob(word, context[1:],verbose)
            else:
                if verbose:
                    print('no backoff for "%s" as model doesn\'t do any smoothing so prob=0.0'%word)
                return alpha

    def _alpha(self, context,verbose=False):
        """Get the backoff alpha value for the given context
        """
        error_message = "Alphas and backoff are not defined for unigram models"
        assert not self.is_unigram_model, error_message

        if context in self._backoff_alphas:
            res = self._backoff_alphas[context]
        else:
            res = 1
        if verbose:
            print(" alpha: %s = %s"%(context,res))
        return res


    def logprob(self, word, context,verbose=False):
        """
        Evaluate the (negative) log probability of this word in this context.

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """

        return -log(self.prob(word, context,verbose), 2)

    @property
    def ngrams(self):
        return self._ngrams

    @property
    def backoff(self):
        return self._backoff

    @property
    def model(self):
        return self._model

    # NB, this will always start with same word since model
    # is trained on a single text
    def generate(self, num_words, context=()):
        '''
        Generate random text based on the language model.

        :param num_words: number of words to generate
        :type num_words: int
        :param context: initial words in generated string
        :type context: list(str)
        '''

        orig = list(context)
        res=[]
        text = list(orig) # take a copy
        for i in range(num_words):
            one=self._generate_one(text)
            text.append(one)
            if one=='</s>' or i==num_words-1:
                if self._lpad is not ():
                    res.append(list(self._lpad)[:(len(self._lpad)+len(context))-(self._n-2)]+text)
                else:
                    res.append(text)
                text=list(orig)
        return res

    def _generate_one(self, context):
        context = (self._lpad + tuple(context))[-self._n+1:]
        # print "Context (%d): <%s>" % (self._n, ','.join(context))
        if context in self:
            return self[context].generate()
        elif self._n > 1:
            return self._backoff._generate_one(context[1:])
        else:
            return self._model.max()

    def entropy(self, text, pad_left=False, pad_right=False,
                verbose=False, perItem=False):
        """
        Calculate the approximate cross-entropy of the n-gram model for a
        given evaluation text.
        This is the average log probability of each item in the text.

        :param text: items to use for evaluation
        :type text: iterable(str)
        :param pad_left: whether to pad the left of each text with an (n-1)-gram of <s> markers
        :type pad_left: bool
        :param pad_right: whether to pad the right of each sentence with an </s> marker
        :type pad_right: bool
        :param perItem: normalise for length if True
        :type perItem: bool
        """
        # This version takes account of padding for greater accuracy
        # Note that if input is a string, it will be exploded into characters 
        e = 0.0
        for ngram in ingrams(chain(self._lpad, text, self._rpad), self._n):
            context = tuple(ngram[:-1])
            token = ngram[-1]
            cost=self.logprob(token, context, verbose)  # _negative_
                                                        # log2 prob == cost!
            if verbose:
                print("p(%s|%s) = [%s-gram] %7f"%(token,context,self._n,2**-cost))
            e += cost
        if perItem:
            return e/((len(text)+self._padLen)-(self._n - 1))
        else:
            return e

    def perplexity(self, text, pad_left=False, pad_right=False, verbose=False):
        """
        Calculates the perplexity of the given text.
        This is simply 2 ** cross-entropy for the text.

        :param text: words to calculate perplexity of
        :type text: list(str)
        :param pad_left: whether to pad the left of each sentence with an (n-1)-gram of empty strings
        :type pad_left: bool
        :param pad_right: whether to pad the right of each sentence with an (n-1)-gram of empty strings
        :type pad_right: bool
        """

        return pow(2.0, self.entropy(text), pad_left=pad_left,
                   pad_right=pad_right, perItem=True)

    def dump(self, file, logBase=None, precision=7):
        """Dump this model in SRILM/ARPA/Doug Paul format

        Use logBase=10 and the default precision to get something comparable
        to SRILM ngram-model -lm output
        @param file to dump to
        @type file file
        @param logBase If not None, output logBases to the specified base
        @type logBase int|None"""
        file.write('\n\\data\\\n')
        self._writeLens(file)
        self._writeModels(file,logBase,precision,None)
        file.write('\\end\\\n')

    def _writeLens(self,file):
        if self._n>1:
            self._backoff._writeLens(file)
            file.write('ngram %s=%s\n'%(self._n,
                                        sum(len(self._model[c].samples())\
                                            for c in self._model.keys())))
        else:
            file.write('ngram 1=%s\n'%len(self._model.samples()))
            

    def _writeModels(self,file,logBase,precision,alphas):
        if self._n>1:
            self._backoff._writeModels(file,logBase,precision,self._backoff_alphas)
        file.write('\n\\%s-grams:\n'%self._n)
        if self._n==1:
            self._writeProbs(self._model,file,logBase,precision,(),alphas)
        else:
            for c in sorted(self._model.conditions()):
                self._writeProbs(self._model[c],file,logBase,precision,
                                  c,alphas)

    def _writeProbs(self,pd,file,logBase,precision,ctxt,alphas):
        if self._n==1:
            for k in sorted(pd.samples()+['<unk>','<s>']):
                if k=='<s>':
                    file.write('-99')
                elif k=='<unk>':
                    _writeProb(file,logBase,precision,1-pd.discount()) 
                else:
                    _writeProb(file,logBase,precision,pd.prob(k))
                file.write('\t%s'%k)
                if k not in ('</s>','<unk>'):
                    file.write('\t')
                    _writeProb(file,logBase,precision,alphas[ctxt+(k,)])
                file.write('\n')
        else:
            ctxtString=' '.join(ctxt)
            for k in sorted(pd.samples()):
                _writeProb(file,logBase,precision,pd.prob(k))
                file.write('\t%s %s'%(ctxtString,k))
                if alphas is not None:
                    file.write('\t')
                    _writeProb(file,logBase,precision,alphas[ctxt+(k,)])
                file.write('\n')

    def __contains__(self, item):
        item=tuple(item)
        try:
            return item in self._model
        except:
            try:
                # hack if model is an MLEProbDist, more efficient
                return item in self._model._freqdist
            except:
                return item in self._model.samples()

    def __getitem__(self, item):
        return self._model[tuple(item)]

    def __repr__(self):
        return '<NgramModel with %d %d-grams>' % (self._N, self._n)

def _writeProb(file,logBase,precision,p):
    file.write('%.*g'%(precision,
                       p if logBase is None else log(p,logBase)))


class LgramModel(NgramModel):
    def __init__(self, n, train, pad_left=False, pad_right=False,
                 estimator=None, *estimator_args, **estimator_kwargs):
        """
        NgramModel (q.v.) slightly tweaked to produce char-grams,
        not word-grams, with a WittenBell default estimator

        :param train: List of strings, which will be converted to list of lists of characters, but more efficiently
        :type train: iter(str)
        """
        if estimator is None:
            assert (not(estimator_args)) and (not(estimator_kwargs)),\
                   "estimator_args (%s) or _kwargs (%s) supplied, but no estimator"%(estimator_args,estimator_kwargs)
            estimator=_estimator
        super(LgramModel,self).__init__(n,
                                        (iter(word) for word in train),
                                        pad_left, pad_right,
                                        estimator,
                                        *estimator_args, **estimator_kwargs)

def teardown_module(module=None):
    from nltk.corpus import brown
    brown._unload()

from nltk.probability import LidstoneProbDist, WittenBellProbDist
def demo(estimator_function=LidstoneProbDist):
    from nltk.corpus import brown
    estimator = lambda fdist, bins: estimator_function(fdist, 0.2, bins+1)
    lm = NgramModel(3, brown.sents(categories='news'), estimator=estimator,
                    pad_left=True, pad_right=True)
    print("Built %s using %s as estimator"%(lm,estimator_function))
    txt="There is no such thing as a free lunch ."
    print("Computing average per-token entropy for \"%s\", showing the computation:"%txt)
    e=lm.entropy(txt.split(),True,True,True,True)
    print("Per-token average: %.2f"%e)
    text = lm.generate(100)
    import textwrap
    print("--------\nA randomly generated 100-token sequence:")
    for sent in text:
        print('\n'.join(textwrap.wrap(' '.join(sent))))
    return lm

if __name__ == '__main__':
    demo()



