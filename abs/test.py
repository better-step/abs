import pickle

# Replace 'your_file.pkl' with the path to your .pkl file
file_path = '/Users/nafiseh/Desktop/train.pkl'

# Open the file in 'rb' (read binary) mode
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Now you can work with the loaded data
print(data)
