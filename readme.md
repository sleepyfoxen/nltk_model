# The NLTK Model module

The module in `ntlk.model` was removed in NLTK version 3, however it provides some very helpful
code for text analysis. Although [there are efforts](https://github.com/nltk/nltk/issues/1342) to
resuscitate the module, it is still not present in NLTK version 3.3 -- and 
[the nltk/model branch](https://github.com/nltk/nltk/tree/model) has not been active in several years.

As such, I am adapting the model code from [an older version of NLTK](https://github.com/nltk/nltk/blob/2.0.4/).
Alternatively, one could just install and use NLTK version 2.0.4. However, hopefully, the `nltk.model` will return
to the upstream.

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
