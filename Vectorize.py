import gensim, logging


source_code = open("source.py", "r")

lines = source_code.readlines()

out = [[]]
for i in range(len(lines)):
    line = lines[i]
    line = line.replace("/n", " ")
    line = line.replace("(", " ")
    line = line.replace(")", " ")
    line = line.replace(".", " ")
    line = line.replace(",", " ")
    line = line.replace(":", " ")
    line = line.replace("[", " ")
    line = line.replace("]", " ")
    line = line.replace("=", " ")
    line = line.replace("/r", " ")
    line = line.replace("/t", " ")
    line = line.replace('"', " ")
    line = line.replace("'", " ")
    line = line.strip()
    split = line.split(" ")
    for j in range(len(split)):
        if (split[j] == "" or split[j] == None):
            continue
        out[0].append(split[j].strip())


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.Word2Vec(out, min_count=1)
print(model.wv.most_similar(positive="def"))