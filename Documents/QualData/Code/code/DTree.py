# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import csv
import numpy as np
from sklearn import tree
from sklearn.externals import joblib
import graphviz
from sklearn.metrics import roc_curve, auc
from scipy import interp
#os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\pkgs\graphviz-2.38-hfd603c8_2\Library\bin'
# Set some parameters
F_n = 10 # Fold Count

# Images loading
def load_images_labels_RGB(img_folder, label_file):
    with open(label_file) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        labels = []
        images = []
        for row in reader:
            img_filename = row[0]
            img = Image.open(os.path.join(img_folder,img_filename))
            if img is not None:
                images.append(np.array(img))
                labels.append(int(row[1]))
    return np.array(images), np.array(labels)

# Import data (normalized)
Img_Data, Label_Data = load_images_labels_RGB('train/', 'train.csv')
Size_Data = Img_Data.shape
Fold_size = Size_Data[0]//F_n
Mean_RGB = np.array([128.41563722, 115.24518493, 119.38645491])
Std_RGB = np.array([38.55379149, 35.64913446, 39.07419321])
Data_Norm = (Img_Data - Mean_RGB)/Std_RGB
Data_NFlat = np.reshape(Data_Norm, (Size_Data[0], 32*32*3))
# Shullf data
per = np.random.permutation(Data_Norm.shape[0])
Shuf_Data_Norm = Data_NFlat[per, :]
Shuf_Label_Data = Label_Data[per]

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
for i in range(F_n):
    DataN_te = Shuf_Data_Norm[Fold_size*i:Fold_size*(i+1), :]
    DataN_tr_1 = Shuf_Data_Norm[0:(Fold_size*i), :]
    DataN_tr_2 = Shuf_Data_Norm[Fold_size*(i+1):, :]
    DataN_tr = np.concatenate((DataN_tr_1,DataN_tr_2))
    DataN_te_y = Shuf_Label_Data[Fold_size*i:Fold_size*(i+1)]
    DataN_tr_y_1 = Shuf_Label_Data[0:(Fold_size*i)]
    DataN_tr_y_2 = Shuf_Label_Data[Fold_size*(i+1):]
    DataN_tr_y = np.concatenate((DataN_tr_y_1,DataN_tr_y_2))
    model_name_DTree = 'Model_DTree_' + str(i+1) + '_rbf.model'
    Graph_name_DTree = 'Graph_DTree_' + str(i+1) + '_'
    # Decision Tree classfication
    clf_DTree = tree.DecisionTreeClassifier()
    clf_DTree.fit(DataN_tr,DataN_tr_y)
    score_DTree = clf_DTree.score(DataN_te,DataN_te_y)
    print("The score of DTree is : %f"%score_DTree)
    joblib.dump(clf_DTree, model_name_DTree)
    
    # Visualize the trained tree
    dot_data = tree.export_graphviz(clf_DTree, out_file=None, filled=True, rounded=True, special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render(Graph_name_DTree)
    # Plot ROC Curve
    probas_Dtree = clf_DTree.predict_proba(DataN_te)
    fpr, tpr, thresholds = roc_curve(DataN_te_y, probas_Dtree[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % ((i+1), roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Decision Tree')
plt.legend(loc="lower right")
plt.rcParams['figure.figsize'] = (6.0, 6.0)
plt.rcParams['figure.dpi'] = 300
plt.savefig('DTree_ROC_rbf.png')
#plt.show()