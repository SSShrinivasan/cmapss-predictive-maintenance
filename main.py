from src.data_processing import load_data, compute_rul, normalize_data

train_path = r"data\train_FD001.txt"
test_path = r"data\Test_FD001.txt"
rul_path = r"data\RUL_FD001.txt"





train_df, test_df, rul_df = load_data(train_path, test_path, rul_path)

train_df = compute_rul(train_df)

train_df, test_df, scaler = normalize_data(train_df, test_df)

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)

print(train_df.head())