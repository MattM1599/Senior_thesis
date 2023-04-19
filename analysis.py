import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import ast
import math

FILE_PATH = 'mooney_features_vanilla.csv'

def main():
    
    height = 16
    width = 32
    posture = np.zeros([height,width])
    lighting = np.zeros([height,width])
    
    df = pd.read_csv(FILE_PATH)
    
    for i in range(0,32):
        q = 3*i
        print("working on " + str(df.iloc[q,2]))
        
        for j in range(3, 19):
            posture[j-3,i] = 1-(pearsonr(ast.literal_eval(df.iloc[q,j]), ast.literal_eval(df.iloc[q+2,j])))[0]
            lighting[j-3,i] = 1-(pearsonr(ast.literal_eval(df.iloc[q,j]), ast.literal_eval(df.iloc[q+1,j])))[0]
            

    posture_mean = np.mean(posture, axis=1)
    posture_std = np.std(posture, axis=1)
    lighting_mean = np.mean(lighting, axis=1)
    lighting_std = np.std(lighting, axis=1)

    print("POSTURE MEAN:")
    print(posture_mean)
    print("POSTURE STD:")
    print(posture_std)
    print("LIGHTING MEAN:")
    print(lighting_mean)
    print("LIGHTING STD:")
    print(lighting_std)

if __name__ == '__main__':
    main()