import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Indlæs datasættet
dataset_path = "./data/kaggle_diabetes_dataset/diabetes.csv"
df = pd.read_csv(dataset_path)


def check_df(dataframe):
    print("##### HEAD #####")
    print(dataframe.head())

    print("##### INFO #####")
    print(dataframe.info())

    print("##### DESCRIBE #####")
    print(dataframe.describe(include="all"))

    print("##### ISNULL() #####")
    print(dataframe.isnull().sum())

    print("##### OUTCOME VC #####")
    print(dataframe['Outcome'].value_counts())


def visualize_df():
    plt.figure(figsize=(10, 6))
    sns.histplot(df['BMI'], bins=30)
    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.title('BMI ting')
    plt.show()


check_df(df)
visualize_df()
