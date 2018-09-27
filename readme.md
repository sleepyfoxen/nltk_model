# The NLTK Model module

**Please see nltk.lm for a supported version of the NLTK model code**

The module in `ntlk.model` was removed in NLTK version 3, however it provides some very helpful
code for text analysis. Now, however, nltk upstream has a new language model. You should use that API instead of this
one, unless you really need support for some old code -- as this is less well-maintained.


I am adapting the model code from [an older version of NLTK](https://github.com/nltk/nltk/tree/2.0.4/).
Alternatively, one could just install and use NLTK version 2.0.4.

This code is (barely) modified so that the imports resolve; there are no substantive changes to the code
other than updating for Python 3 and referencing new NLTK method names. However, I have also included the LgramModel
from even earlier versions of the NLTK, simply because it is useful in my use-case.

I have browsed the NLTK repositories in the hope of finding where the LgramModel was removed from the tree, but I
had no luck (it is before the project was using git, I believe).

It retains its original copyright licence - see
[LICENCE.txt](https://github.com/sleepyfoxen/nltk_model/blob/master/LICENCE.txt).

Please note that this *still* depends on NLTK as I am not distributing the entire NLTK package -- only a module
re-adding files from the old model module. You can install the latest version of NLTK through pip: `pip install nltk`.

## Usage

1.  Install `nltk` through `pip`: `pip install nltk`
2.  Clone this repository in your project directory: `git clone https://github.com/sleepyfoxen/nltk_model.git`
3.  Import it and use it in your code: `from nltk_model import NgramModel, LgramModel`

## Example

```python
from nltk_model import *
from nltk.corpus import brown
from nltk.probability import LidstoneProbDist


est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
lm = NgramModel(3, brown.words(categories='news'), estimator=est)
print(lm)
print(lm._backoff)
print(lm.entropy(['The', 'Fulton', 'County', 'Grand', 'Jury', 'said',
                 'Friday', 'an', 'investigation', 'of', "Atlanta's", 'recent',
                 'primary', 'election', 'produced', '``', 'no', 'evidence',
                 "''", 'that', 'any', 'irregularities', 'took', 'place', '.']))
```
