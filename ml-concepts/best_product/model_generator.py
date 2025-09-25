import joblib

from best_product.data_cleaner import data_cleaner
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def model_generator() :
    X, y = data_cleaner()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # update y_train
    # y_train = y_train.apply(categorize_rank)

    # create a model
    model = DecisionTreeClassifier()
    # model = DecisionTreeRegressor()

    # train the model
    model.fit(X_train, y_train)
    joblib.dump(model, r"D:\ShubhamN\learning\ai-ml\ml-concepts\best_product\rank_generator.joblib")

    # Save the column names of X_train for future prediction
    joblib.dump(X_train.columns.tolist(), r"D:\ShubhamN\learning\ai-ml\ml-concepts\best_product\rank_generator_columns.joblib")

    return model,X_test,y_test


def check_accuracy(model, inp_testing_data, out_testing_data) :
    prediction = model.predict(inp_testing_data)
    score = accuracy_score(out_testing_data, prediction)
    print(score)


def categorize_rank(rank):
    if rank <= 100:
        return "top"
    elif rank <= 500:
        return "mid"
    else:
        return "low"