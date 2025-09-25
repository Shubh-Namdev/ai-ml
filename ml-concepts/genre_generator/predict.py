import joblib

model = joblib.load('music-recommender.joblib')

# prediction
prediction = model.predict([[32, 1]])
print(prediction)

