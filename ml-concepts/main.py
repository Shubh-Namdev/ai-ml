import joblib
import pandas as pd

from best_product.model_generator import model_generator, check_accuracy
from best_product.predict import predict_result

# df = pd.read_csv(r"data\gvsales.csv")
# print(df)
# print(df.describe())
# print(df.values)
# print(df.values.tolist())

# generate model and check prediction score
# model, inp_testing_data, out_testing_data = model_generator()
# check_accuracy(model, inp_testing_data, out_testing_data)

input_data = pd.DataFrame([{
    "Name": "Mario Kart Wii",
    "Platform": "Wii",
    "Year": 2008,
    "Genre": "Racing",
    "Publisher": "Nintendo",
    "NA_Sales": 15.85,
    "EU_Sales": 12.88,
    "JP_Sales": 3.79,
    "Other_Sales": 3.31,
    "Global_Sales": 35.82
}])

rank = predict_result(input_data)
print(rank)