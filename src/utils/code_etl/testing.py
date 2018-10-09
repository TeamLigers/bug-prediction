import re
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.mllib.feature import Word2Vec
import _pickle as pickle
import json
import numpy as np

#initialize Spark and open json file.
sc = SparkContext()
sqlCtx = SQLContext(sc)
code_lines = sqlCtx.read.option("multiLine", True).option("mode", "PERMISSIVE").json("\jsons\hi.json")
code_lines = code_lines.repartition(300)

#Process each line of code.
def split_code(input):
    strs = " ".join(input)
    patt = re.compile(r"[\w]", re.UNICODE)
    return patt.findall(strs)

#Load data into a Spark dataset.
words = code_lines\
    .rdd.map(
        lambda thing: (thing[11].split())
    )\
    .map(lambda line: [f.lower() for f in line])\
    .map(lambda line: split_code(line))\
    .filter(lambda line: line != [])

#Train Word2Vec model with dataset of code snippets.
word2vec = Word2Vec()
word2vec.setMinCount(2)    # Must be atleast 2 occurrences of a word to produce a vector.
word2vec.setVectorSize(50)  # Size of resulting vector.
model = word2vec.fit(words)

#Save word embeddings into a json file.
model_dict = {k: list(v) for k, v in dict(model.getVectors()).items()}
with open("py2vec_model.json", "w") as f:
    json.dump(model_dict, f, indent=4)

#Save as pickle for easy storing and loading.
model_dict = {k: np.array(list(v)) for k, v in dict(model.getVectors()).items()}
with open("py2vec_model.pkl", "wb") as f:
    pickle.dump(model_dict, f)
    
