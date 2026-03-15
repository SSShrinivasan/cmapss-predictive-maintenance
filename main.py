from src.data_processing import load_data, compute_rul, normalize_data
from src.sequence_generator import generate_sequences

train_path = r"data\train_FD001.txt"
test_path = r"data\Test_FD001.txt"
rul_path = r"data\RUL_FD001.txt"

WINDOW = 30

train_data, test_data, rul_data = load_data(train_path, test_path, rul_path)

train_data, test_data, rul_data = load_data(train_path, test_path, rul_path)

train_data = compute_rul(train_data)

X_train, y_train = generate_sequences(train_data, WINDOW)

print("Train sequence shape:", X_train.shape)
print("Train labels shape:", y_train.shape)