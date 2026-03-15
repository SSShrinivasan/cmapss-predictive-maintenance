import numpy as np

def generate_sequences(dataframe, sequence_length):

    sequences = []
    labels = []

    # remove columns not used as features
    feature_cols = dataframe.columns.difference(['unit_number','time_in_cycles','RUL'])

    for unit in dataframe['unit_number'].unique():

        unit_data = dataframe[dataframe['unit_number'] == unit]

        features = unit_data[feature_cols].values
        rul = unit_data['RUL'].values

        for i in range(len(unit_data) - sequence_length):

            seq = features[i:i+sequence_length]
            label = rul[i + sequence_length - 1]

            sequences.append(seq)
            labels.append(label)

    return np.array(sequences), np.array(labels)