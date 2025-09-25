import pandas as pd
import joblib

def predict_result(input_data) :
    # Load model
    model = joblib.load(r"D:\ShubhamN\learning\ai-ml\ml-concepts\best_product\rank_generator.joblib")

    # Apply same preprocessing as training
    # 1. lowercase text
    text_cols = ["Name", "Platform", "Genre", "Publisher"]
    input_data[text_cols] = input_data[text_cols].apply(lambda x: x.str.lower())

    # 2. one-hot encode
    # Get columns from training set
    training_columns = joblib.load(r"D:\ShubhamN\learning\ai-ml\ml-concepts\best_product\rank_generator_columns.joblib")
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=training_columns, fill_value=0)  # add missing columns

    # Predict
    rank = model.predict(input_data)

    return rank
