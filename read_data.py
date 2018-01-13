import pandas as pd
import soundfile as sf
import numpy as np

path_to_data_set = 'data\\mlsp_contest_dataset\\'


def read_species_list():
    """
    :return: A Pandas DataFrame with all species with their class id (index) and code
    """
    with open(path_to_data_set + 'essential_data\\species_list.txt') as f:
        table = pd.read_table(f, sep=',', index_col='class_id')
    return table


def read_wav_file_names():
    """
    :return: A Pandas DataFrame with all rec_id's and their corresponding filename
    """
    with open(path_to_data_set + 'essential_data\\rec_id2filename.txt') as f:
        table = pd.read_table(f, sep=',', index_col='rec_id')
    return table


def _read_cv_folds_2():
    """
    :return: A Pandas DataFrame with all rec_id's and a flag in which fold they belong (train/test)
    """
    with open(path_to_data_set + 'essential_data\\CVfolds_2.txt') as f:
        table = pd.read_table(f, sep=',', index_col='rec_id')
    return table


def read_labels():
    """
    :return: A Pandas DataFrame with all rec_id's and string of the corresponding labels (separated with commas)
    """
    # with open(path_to_data_set + 'essential_data\\rec_labels_test_hidden.txt') as f:
    with open(path_to_data_set + 'essential_data\\bag_labels.txt') as f:
        data = {'rec_id': [], 'labels': []}
        for line in f.readlines()[1:]:      # Skip header
            line = line.replace('\n', '')   # Remove \n
            entries = line.split(',', 1)    # Split from index
            data['rec_id'].append(int(entries[0]))
            if len(entries) == 1 or entries[1] == '':
                data['labels'].append(np.array([]))
            else:
                data['labels'].append(np.array([int(s) for s in entries[1].split(',')]))
        return pd.DataFrame(data).set_index('rec_id')


def read_wav_files():
    """
    :return: A Pandas DataFrame with all rec_id's and the corresponding audio sample
    """
    names = read_wav_file_names()
    data = {'rec_id': [], 'signal': [], 'sample_rate': []}
    for index, row in names.iterrows():
        signal, sample_rate = sf.read(path_to_data_set + 'essential_data\\src_wavs\\' + row['filename'] + '.wav')
        data['rec_id'].append(index)
        data['signal'].append(signal)
        data['sample_rate'].append(sample_rate)
    return pd.DataFrame(data).set_index('rec_id')


def read_data_and_labels():
    """
    :return: A four-tuple of Pandas DataFrames: (train data, train labels, test data, test labels)
    """
    folds = _read_cv_folds_2()
    data = read_wav_files()
    labels = read_labels()
    train_indices = folds.loc[folds['fold'] == 0].index
    test_indices = folds.loc[folds['fold'] == 1].index
    return data.iloc[train_indices], labels.iloc[train_indices], data.iloc[test_indices], labels.iloc[test_indices]


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = read_data_and_labels()

    print(read_species_list())

    print('\nTrain Data')
    print(train_data.head())
    print(train_data.shape)

    print('\nTrain Labels')
    print(train_labels.head())
    print(train_labels.shape)

    print('\nTest Data')
    print(test_data.head())
    print(test_data.shape)

    print('\nTest Labels')
    print(test_labels.head())
    print(test_labels.shape)

