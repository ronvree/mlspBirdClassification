import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import rogier_data as data

labels = data.get_training_labels()


features = data.get_1d_training_data(True)

num_classes = data.get_num_classes()

print(features.shape, "Features")
print(labels.shape, "Labels")

print("Done extracting data")

classif = RandomForestClassifier(n_estimators=500, criterion='entropy',
                                     random_state=np.random.RandomState(0))
classif.fit(features, labels)

test_labels = data.get_test_labels()

test_features = data.get_1d_test_data(True)

preds = np.array(classif.predict_proba(test_features))
print(preds.shape)
print(test_labels.shape)

predictions = np.zeros(shape=(test_labels.shape[0], test_labels.shape[1]))
#
for p in range(predictions.shape[0]):
    for bird in range(predictions.shape[1]):
        predictions[p][bird] = preds[bird][p][1]
    # print(str(test_labels[p]) + " : " + str(predictions[p]))
#
from rogier_cnn import birds_correct
birds_correct(predictions, test_labels)
# print(predictions)
roc = metrics.roc_auc_score(test_labels,predictions, average='micro')
print("AUC: {}".format(roc))
