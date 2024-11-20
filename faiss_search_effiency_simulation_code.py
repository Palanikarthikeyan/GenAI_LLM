import pandas as pd

d = {"f1":["which location?","give me your phone","today"],"f2":["city","lastdigit","random"]}

df = pd.DataFrame(d)
print(df)

# create vectors from the given data(text)

from sentence_transformers import SentenceTransformer

text = df['f1']
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

vectors = encoder.encode(text)

# Build a FAISS index from the vector

import faiss
vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

# Create a search vector
import numpy as np

search_text = "which location?"

search_vector = encoder.encode(search_text)

res_vector =np.array([search_vector])

faiss.normalize_L2(res_vector)

# search
distances,approximate = index.search(res_vector,k=index.ntotal)

result = pd.DataFrame({'distances':distances[0],'approximate':approximate[0]})
print(result)
# distances  | approximate  - approximate nearest neighbour corresponding to distance
#  0.87      |  2
#  1.23      |  3
#  1.64      |  0 <= approxmiate 0 ->vector at position 0 in the index
#

merge_df = pd.merge(result,df,left_on='approximate',right_index=True)

print(merge_df) # see shortest distance 












