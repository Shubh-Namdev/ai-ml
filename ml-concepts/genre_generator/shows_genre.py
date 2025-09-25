import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# need data
music_data = pd.read_csv("../data/music.csv")
# print(music_data)

# Decide the input data which to be provided to model for training and output which model generate
raw_input_data = music_data.drop(columns=["genre"])
# print(raw_input_data)

raw_output_data = music_data["genre"]
# print(raw_output_data)
y = raw_output_data.dropna()
# print(y)

# If null data , decide replacing or removal
filtered_input_data = raw_input_data.dropna(subset=["gender"])
# print(filtered_input_data)

# in case data is not consistent , we need to make it a consistent and reliable data
# filtered_input_data["column_name"] = filtered_input_data["column_name"].replace({1000:100})

# check the columns data type
filtered_input_data["age"] = filtered_input_data["age"].astype(int)
# print(filtered_input_data)

# remove duplicates
X = filtered_input_data.drop_duplicates()
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# create a model
model = DecisionTreeClassifier()

# Train model
# model.fit(X, y)
model = model.fit(X_train, y_train)

joblib.dump(model, "music-recommender.joblib")
print("model generated")

# predictions
# prediction = model.predict([[34, 1], [21, 0], [24, 1], [39, 0]])
# prediction = model.predict(X_test)
# print(prediction)

# accuracy_score = accuracy_score(y_test, prediction)
# print(accuracy_score)

