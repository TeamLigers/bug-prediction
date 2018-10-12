import re
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.mllib.feature import Word2Vec
import _pickle as pickle
import json
import numpy as np
import os

conf = SparkConf().setAppName("Name01").setMaster(2)

sc = SparkContext()
sqlCtx = SQLContext(sc)
#code_lines = sqlCtx.read.option("multiLine", True).option("mode", "PERMISSIVE").json(os.getcwd() + "\jsons\hi4.json")
code_lines = sqlCtx.read.json(os.getcwd() + "\jsons\java.json")
code_lines = code_lines.repartition(300)

def split_code(input):
    strs = " ".join(input)
    patt = re.compile(r"[\w]")
    return patt.findall(strs)

words = code_lines\
    .rdd.map(
        lambda line: line[11].split(" ")
    )\
    .map(lambda line: [f.lower() for f in line])\
    .filter(lambda line: line != [])
    #.map(lambda line: split_code(line))\

word2vec = Word2Vec()
word2vec.setMinCount(2)    # Default 100
word2vec.setVectorSize(50)  # Default 100
model = word2vec.fit(words)

model_dict = {k: list(v) for k, v in dict(model.getVectors()).items()}

with open(os.getcwd() + "\py2vec\py2vec_modelJ.json", "w+") as f:
    json.dump(model_dict, f, indent=4)

model_dict = {k: np.array(list(v)) for k, v in dict(model.getVectors()).items()}

with open(os.getcwd() + "\py2vec\py2vec_modelJ.pkl", "wb+") as f:
    pickle.dump(model_dict, f)

