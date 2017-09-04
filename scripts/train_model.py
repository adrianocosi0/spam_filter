from sklearn.svm import LinearSVC as SVC
import pickle as pkl
from sklearn.model_selection import cross_val_score
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import joblib
plt.style.use('seaborn')


d = pkl.load(open('final_product/data/train_test_targets.pkl','rb'))
X_train, X_test, y_train, y_test = [d[x].astype(int) for x in d.keys()]
# X_train, X_test = X_train.astype(int), X_test.astype(int)
# y_train, y_test = X_train.iloc[:,0], X_test.iloc[:,0]
# X_train, X_test = X_train.iloc[:,1:], X_test.iloc[:,1:]
scores = []

def cross_validate_and_plot(X_train,y_train,X_test,y_test,interval=[0.0001,0.1]):
    guesses = np.linspace(interval[0],interval[1],50)
    scores = []
    for c in guesses:
        mod = SVC(C=c,loss='hinge')
        mod.fit(X_train,y_train.values.ravel())
        v = cross_val_score(mod, X_train.values, y_train.values.ravel(),
                            scoring='roc_auc',cv=8).mean()
        scores += [v]
    fig, ax = plt.subplots(figsize=(18,15))
    ax.plot(guesses, scores)
    ax.set_xlabel('C values')
    ax.set_ylabel('ROC auc score on 8 fold cross validation')
    ax.plot(guesses[np.argmax(scores)], max(scores), marker='o', color='r')
    plt.xticks(list(plt.xticks()[0]) + [guesses[np.argmax(scores)]])
    plt.yticks(list(plt.yticks()[0]) + [max(scores)])
    ax.vlines(guesses[np.argmax(scores)],ax.get_ylim()[0],max(scores), linestyle='--',color='r')
    ax.hlines(max(scores), ax.get_xlim()[0]*2,guesses[np.argmax(scores)], linestyle='--',color='r')
    #print(scores, np.argmax(scores))
    mod = SVC(C=guesses[np.argmax(scores)],loss='hinge').fit(X_train,y_train)
    print('We have a roc_auc_score of {}'.format(roc_auc_score(y_test.values.ravel(), mod.predict(X_test.values).ravel())))
    return mod, scores, fig

if __name__=='__main__':
    mod, scores, fig = cross_validate_and_plot(X_train,y_train,X_test,y_test)
    fig.savefig('plots/Cross_validation_errors')
    joblib.dump(mod, 'final_product/model/spam_model.pkl')
