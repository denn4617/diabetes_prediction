import pandas as pd
from sklearn.model_selection import train_test_split

# Indlæs datasæt
dataset_path = "./data/kaggle_diabetes_dataset/diabetes.csv"
df = pd.read_csv(dataset_path)

# Erstat nulværdier med median, da dette kunne tyde på manglende data
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness',
                      'Insulin', 'BMI']

for column in columns_to_replace:
    median = df[column].median()
    df[column] = df[column].replace(0, median)

# Opdel datasættet
X = df.drop('Outcome', axis=1)  # Feature data
y = df['Outcome']  # Target variable

# Opdeling af data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Kombiner DataFrames til træning og test
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Gem DataFrames som CSV
train_data.to_csv("./data/train_data.csv", index=False)
test_data.to_csv("./data/test_data.csv", index=False)
