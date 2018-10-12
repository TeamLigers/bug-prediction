from Py2VecMod import Py2Vec
import os
import numpy as np

model = Py2Vec(os.getcwd() + "\py2vec\py2vec_model3.json")

print(str(model['if']) + "\n\n\n\n\n")


ifV = model['if']
elifV = model['elif']
print(np.sum((ifV - elifV)**2))

print(model.closest_words('if', 5))
