from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--csv', type=str, required=True, help='csv file for getting accuracy')
parser.add_argument('-l', '--label', type=str, required=True, help='label file')
parser.add_argument('-t', '--threshold', type=str, default=0.66, help='threadshold for predicting')
parser.add_argument('-o', '--output', default=None, help="save [option]")
args = parser.parse_args()

def main():
    data = pd.read_csv(args.csv, names=['hash', 'ypred'])
    ypred = np.where(np.array(data.ypred) > args.threshold, 1, 0)

    label = pd.read_csv(args.label, names=['hash', 'y'])
    y = label.y
    
    #get and print accuracy
    accuracy = accuracy_score(y, ypred)
    #print("accuracy : %.0000f%%" % (np.round(accuracy, decimals=4)*100))
    print(args.threshold)
    print(accuracy)
   
    #get and print matrix
    mt = confusion_matrix(y, ypred)
    t = mt[0][0]
    mt[0][0] = mt[1][1]
    mt[1][1] = t
    print(mt)

    #print FP, FN
    print("True Postive : %.0f%%" % (round(mt[0][1]/(mt[0][1]+mt[1][1]), 2)*100))
    print("False Positve : %.0f%%" % (round(mt[1][0]/(mt[0][0]+mt[1][0]), 2)*100))

    #save accuracy, mt [option]
    if args.output:
        with open(os.path.join(args.output, 'accuarcy.txt'), 'w') as f:
            accuracy.tofile(f, format='%s', sep='str')
        np.savetxt(os.path.join(args.output, 'matrix.txt'), mt)

if __name__=='__main__':
    main()
