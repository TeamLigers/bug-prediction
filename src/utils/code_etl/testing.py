import re
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.mllib.feature import Word2Vec

sc = SparkContext()
sqlCtx = SQLContext(sc)
print('hi')
code_lines = sqlCtx.read.option("multiLine", True).option("mode", "PERMISSIVE").json("\jsons\hi.json")
code_lines = code_lines.repartition(300)

def split_code(input):
    strs = " ".join(input)
    patt = re.compile(r"[\w]", re.UNICODE)
    return patt.findall(strs)

words = code_lines\
    .rdd.map(
        lambda thing: (thing[11].split())
    )\
    .map(lambda line: [f.lower() for f in line])\
    .map(lambda line: split_code(line))\
    .filter(lambda line: line != [])

print(words)

word2vec = Word2Vec()
word2vec.setMinCount(25)    # Default 100
word2vec.setVectorSize(50)  # Default 100
model = word2vec.fit(words)

