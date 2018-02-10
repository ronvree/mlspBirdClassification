import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from rogier_data import BirdData

data = BirdData()
features, labels = data.get_1d_training_data(True)

num_classes = data.get_num_classes()

print(features.shape, "Features")
print(labels.shape, "Labels")

print("Done extracting data")
for i in range(5):
    classif = RandomForestClassifier(n_estimators=500, criterion='entropy', verbose=1, n_jobs=-1)
    classif.fit(features, labels)

    test_labels, _ = data.get_test_labels()

    test_features = data.get_1d_test_data(True)

    preds = np.array(classif.predict_proba(test_features))
    classifications = classif.predict(test_features)
    # print(preds.shape)
    # predictions = preds[:, 1]
    # print(predictions.shape)
    if len(preds.shape) > 2:
        predictions = np.zeros(shape=(test_labels.shape[0], test_labels.shape[1]))
        #
        for p in range(predictions.shape[0]):
            for bird in range(predictions.shape[1]):
                predictions[p][bird] = preds[bird][p][1]
        # print(str(test_labels[p]) + " : " + str(predictions[p]))
        # birds_correct(predictions, test_labels)
    else:
        predictions = preds[:, 1]


    acc = metrics.accuracy_score(test_labels, classifications)
    roc = metrics.roc_auc_score(test_labels, predictions, average='micro')
    print("Accuracy: {}".format(acc))
    print("AUC: {}".format(roc))
