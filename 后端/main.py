from nltk import *
from nltk.corpus import *
import language_tool_python
import nltk
import gensim
import time
from Synonyms_suggestion import Synonyms_suggestion
from data_RW import data_RW
from data_treatment import data_treatment
from compositon_score import CompositionScroe


text = "i love you"

test = Synonyms_suggestion(1)
print(test.suggestion_word("love",text,1))

