from numpy import load
import pandas as pd
import os
from IPython.display import display, HTML

# A script for making a dataframe with columns for id, title, type, shape, features, and transformer
index = 0
directory = 'txt_output' # directory of npzs
df = pd.DataFrame(columns=['id', 'title', 'type', 'shape', 'features', 'model']) # Column names

for filename in os.listdir(directory):
   f = os.path.join(directory, filename)
   # checking if it is a file
   if os.path.isfile(f) and f.split('/')[1].split('_')[0] != '.DS':
      data = load(f)
      df.loc[index] = [f.split('/')[1].split('_')[0], f.split('/')[1].split('.')[0], "local features", data['features'].shape, data['features'][0], "ViT-B/32"]
      index = index + 1

      df.loc[index] = [f.split('/')[1].split('_')[0], f.split('/')[1].split('.')[0], "global features", data['g_feature'].shape, data['g_feature'][0], "ViT-B/32"]
      index = index + 1
        
        
product = df.sort_values(['title', 'type'], ascending=[True, False]) # alphabetize values
product.to_pickle('mooney_features.pkl') # save df


#unpickled = pd.read_pickle('mooney_features.pkl')
#display(unpickled)