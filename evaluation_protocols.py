import numpy as np


def k_fold_cross_validation(train_d, train_l, k, model, performance_metrics):
    if len(train_d) != len(train_l): raise Exception('Data size does not match label size!')

    # Make list length divisible by k
    train_d, train_l = train_d[len(train_d) % k:], train_l[len(train_l) % k:]
    # Determine size of each sample
    sample_size = len(train_d) / k
    # Create k samples of equal size
    train_ds = [train_d[i:(i + sample_size)] for i in range(0, len(train_d), sample_size)]
    train_ls = [train_l[i:(i + sample_size)] for i in range(0, len(train_l), sample_size)]
    # Iterate through folds
    metric_scores = np.zeros(len(performance_metrics))
    for fold in range(k):
        # Separate validation / train samples
        validation_d, validation_l = train_ds[fold], train_ls[fold]
        train_d, train_l = [], []
        for sample in [train_ds[i] for i in range(k) if i != fold]: train_d.extend(sample)
        for sample in [train_ls[i] for i in range(k) if i != fold]: train_l.extend(sample)
        train_d, train_l = np.array(train_d), np.array(train_l)

        # Perform training
        model.fit(train_d, train_l)

        # Perform classification
        classifications = model.predict(validation_d)
        # classifications = [trained_model.classify(i) for i in validation_d]

        # Determine performance
        for i in range(len(performance_metrics)):
            metric_scores[i] += performance_metrics[i](validation_l, classifications)

    # Return average score
    return metric_scores / float(k)


# def hold_out_test_set(train_d, train_l, model, performance_metrics):



