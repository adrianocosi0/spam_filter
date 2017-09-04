import pickle as pkl
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.style.use('seaborn')

d = pkl.load(open('final_product/data/train_test_targets.pkl','rb'))
X_train, X_test, y_train, y_test = [d[x].astype(int).values for x in d.keys()]
mod = joblib.load(open('final_product/model/spam_model.pkl','rb'))
#guesses = np.linspace(upp_bound,low_bound,50)

def plot_roc_curve(mod,X_test,y_test,guesses):
    y_1_inds = y_test==1
    zeros_and_ones = np.array([y_1_inds.size-y_1_inds.sum(),y_1_inds.sum()])
    fig, ax = plt.subplots(figsize=(18,15))
    x = []
    y = []
    for thr in guesses:
        preds = (mod.decision_function(X_test)>thr).astype(int)
        p_r_arr = (confusion_matrix(y_test,preds)/zeros_and_ones)[:,1]
        tpr,fpr = p_r_arr[1],p_r_arr[0]
        x+=[fpr]
        y+=[tpr]
    ax.axes.set_title('ROC curve')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.plot(x,y)
    return x,y,guesses,ax,fig

def find_smallest_threshold_0_fpr(X_test,y_test,fpr,guesses,target=0):
    c = mod.decision_function(X_test)
    cc = np.array(fpr)<=target
    i = np.argmax(cc)

    if cc.sum()==0 or i == len(fpr)-1:
        threshold_for_fpr_0 = guesses[-1]
        if cc.sum()==0:
            print('No value satisfy the target')
            next_bound = guesses[-1]-(2*np.std(c))
        else:
            next_bound = guesses[i]+(guesses[i]-guesses[i-1])
    else:
        threshold_for_fpr_0 = guesses[i]
        try:
            next_bound = guesses[i+1]
        except IndexError:
            next_bound = guesses[i]+(guesses[i]-guesses[i-1])

    #next_bound = guesses[i]
    guesses_new = np.linspace(threshold_for_fpr_0, next_bound,100)
    y_1_inds = y_test==1
    y_that_are_0 = y_test.shape[0]-y_1_inds.sum()
    for j,t in enumerate(guesses_new):
        preds = (mod.decision_function(X_test)>t).astype(int)
        #print(t,preds[~y_1_inds.ravel()].sum())
        if ((preds[~y_1_inds.ravel()]).sum()/y_that_are_0)>=target:
            break
    if j==len(guesses_new)-1:
        print('Nothing found for that target')
    return guesses_new[j-1]

def find_largest_threshold_99_9_tpr(X_test,y_test,tpr,guesses,target=0.993):
    c = mod.decision_function(X_test)
    cc = np.array(tpr)>=target
    i = np.argmax(cc)

    if cc.sum()==0 or i == len(tpr)-1:
        threshold_for_tpr_1 = guesses[-1]
        if cc.sum()==0:
            print('No value satisfy the target')
            next_bound = guesses[-1]-(2*np.std(c))
        else:
            next_bound = guesses[i]+(guesses[i]-guesses[i-1])
    else:
        threshold_for_tpr_1 = guesses[i]
        try:
            next_bound = guesses[i+1]
        except IndexError:
            next_bound = guesses[i]+(guesses[i]-guesses[i-1])

    guesses_new = np.linspace(threshold_for_tpr_1, next_bound,100)
    y_1_inds = y_test==1
    y_that_are_1 = y_1_inds.sum()
    for j,t in enumerate(guesses_new):
        preds = (mod.decision_function(X_test)>t).astype(int)
        #print(t,((preds[y_1_inds.ravel()]==y_test[y_1_inds.ravel()]).sum()/y_that_are_1))
        if ((preds[y_1_inds.ravel()]==y_test[y_1_inds.ravel()]).sum()/y_that_are_1)>=target:
            break
    if j==len(guesses_new)-1:
        print('Nothing found for that target')
    return guesses_new[j-1]

if __name__=='__main__':
    c = mod.decision_function(X_test)
    low_bound, upp_bound = [np.mean(c)-(2*np.std(c)),np.mean(c)+(2*np.std(c))]
    guesses = np.linspace(low_bound, upp_bound, 20)
    fpr,tpr,guesses,ax,fig = plot_roc_curve(mod,X_test,y_test,guesses)
    sm_thr = find_smallest_threshold_0_fpr(X_test,y_test,fpr,guesses,target=0)
    larg_tpr = find_largest_threshold_99_9_tpr(X_test,y_test,tpr,guesses,target=0.993)
    print(sm_thr,larg_tpr)
    ax.set_title('ROC curve. 0 fpr achieved with {}, 0.993 tpr achieved with {}'.format(sm_thr, larg_tpr))
    fig.savefig('plots/ROC_curve.png')
