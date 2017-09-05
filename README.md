### RESULTS

The plots show the process of selecting the C parameter (regularization) for the SVM model and ROC curve and in particular the bokeh generated html figure below (download to use, too large to display on github)

[ROC curve result](plots/ROC_representation.html)

Allows to navigate the different points of the ROC curve

---

### USAGE

Input are 'bad folders' bassed to the -b flag where emails are not spam. The 'good folders' -g should only contain spam emails.

`python scripts/processing_data.py -b <list_of_folders_with_non_spam_emails> -g <list_of_folders_with_spam_emails>`

`python scripts/train_model.py`

`python scripts/evaluate_model.py`

