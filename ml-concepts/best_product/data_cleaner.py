import pandas as pd
from sklearn.preprocessing import LabelEncoder


def data_cleaner() :
    # read data from csv
    vg_sales_data = pd.read_csv(r"D:\ShubhamN\learning\ai-ml\ml-concepts\data\gvsales.csv")

    # cleaning the data
    # drop or replace null where required
    vg_sales_input_data = vg_sales_data.dropna(subset=["Year"])
    vg_sales_input_data = vg_sales_input_data.fillna({"NA_Sales": 0, "EU_Sales": 0, "JP_Sales": 0,
                                                "Other_Sales": 0, "Global_Sales": 0})

    # replace inconsistent values
    vg_sales_input_data = vg_sales_input_data[vg_sales_input_data["Other_Sales"] <= 100]

    # standardize the input
    vg_sales_input_data[["Name", "Platform", "Genre", "Publisher"]] = vg_sales_input_data[
        ["Name", "Platform", "Genre", "Publisher"]
    ].apply(lambda x: x.str.lower())

    # change the type of data if required
    le = LabelEncoder()
    vg_sales_input_data["Name"] = le.fit_transform(vg_sales_input_data["Name"])
    vg_sales_input_data["Platform"] = le.fit_transform((vg_sales_input_data["Platform"]))
    vg_sales_input_data["Genre"] = le.fit_transform(vg_sales_input_data["Genre"])
    vg_sales_input_data["Publisher"] = le.fit_transform(vg_sales_input_data["Publisher"])


    # drop duplicate rows
    vg_sales_input_data = vg_sales_input_data.drop_duplicates()
    # print(vg_sales_input_data)
    # print(vg_sales_input_data.describe())

    # drop output data columns
    vg_sales_output_data = vg_sales_input_data["Rank"]
    vg_sales_input_data = vg_sales_input_data.drop(columns=["Rank"])

    return vg_sales_input_data,vg_sales_output_data


# inp, outp = data_cleaner()
# print(inp)
# print(outp)


