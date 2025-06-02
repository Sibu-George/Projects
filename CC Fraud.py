import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv(r"C:\Users\Metatron\Downloads\archive (1)\fraudTrain.csv")


numeric_data = train_data.select_dtypes(include=[np.number])
cor_mat = numeric_data.corr()
plt.imshow(cor_mat, cmap="coolwarm")
plt.colorbar()

variables = []
for col in train_data.columns:
    if train_data[col].dtypes in ["int64", "float64"]:
                variables.append(col)
plt.xticks(range(len(cor_mat)), variables, rotation=45, ha='right')
plt.yticks(range(len(cor_mat)), variables)
plt.show()

train_data.drop(axis=1, columns = ["Unnamed: 0", "cc_num", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"], inplace=True)

fraud = train_data[train_data.is_fraud == 1]
not_fraud = train_data[train_data.is_fraud == 0]

def Train_Fraud_nonFraud():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_theme()
    cat_fraud = fraud.category.value_counts().reset_index()
    cat_fraud.columns = ["Category", "Counts"]
    cat_not_fraud = not_fraud.category.value_counts().reset_index()
    cat_not_fraud.columns = ["Category", "Counts"]
    sns.barplot(x="Category", y="Counts", data=cat_fraud, ax=axes[0])
    axes[0].set_title("Number of fraudulent transactions per category")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Number of transactions")
    axes[0].tick_params(axis="x", rotation=90)
    sns.barplot(x="Category", y="Counts", data=cat_not_fraud, ax=axes[1])
    axes[1].set_title("Number of non-fraudulent transactions per category")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Number of transactions")
    axes[1].tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.show()

def Train_Fraud_nonFraud_Gender():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_theme()
    g_fraud = fraud.gender.value_counts().reset_index()
    g_fraud.columns = ["Gender", "Counts"]
    g_not_fraud = not_fraud.gender.value_counts().reset_index()
    g_not_fraud.columns = ["Gender", "Counts"]
    sns.barplot(x="Gender", y="Counts", data=g_fraud, ax=axes[0])
    axes[0].set_title("Number of fraudulent transactions per gender")
    axes[0].set_xlabel("Gender")
    axes[0].set_ylabel("Number of transactions")
    axes[0].bar_label(axes[0].containers[0])
    sns.barplot(x="Gender", y="Counts", data=g_not_fraud, ax=axes[1])
    axes[1].set_title("Number of non-fraudulent transactions per gender")
    axes[1].set_xlabel("Gender")
    axes[1].set_ylabel("Number of transactions")
    axes[1].bar_label(axes[1].containers[0])
    plt.tight_layout()
    plt.show()


train_data["trans_year"] = pd.Series(pd.to_datetime(train_data.trans_date_trans_time)).dt.year
train_data["dob"] = pd.Series(pd.to_datetime(train_data.dob)).dt.year

age = pd.Series(train_data.trans_year - train_data.dob)
train_data["age"] = age

bins = [10, 18, 35, 60, 100]
labels = ["14-18", "18-35", "35-60", "60+"]
train_data["age_group"] = pd.cut(train_data['age'], bins=bins, labels=labels, right=True)

train_data.drop(axis=1, columns=["age", "dob", "trans_year", "trans_date_trans_time"], inplace=True)

train_data.head(10)

age_fraud = train_data[train_data.is_fraud == 1].age_group.value_counts().reset_index()
age_fraud.columns = ["Age group", "Counts"]
age_not_fraud = train_data[train_data.is_fraud == 0].age_group.value_counts().reset_index()
age_not_fraud.columns = ["Age group", "Counts"]

def Train_Fraud_nonFraud_Age():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_theme()
    sns.barplot(x="Age group", y="Counts", data=age_fraud, ax=axes[0])
    axes[0].set_title("Number of fraudulent transactions per age group")
    axes[0].set_xlabel("Age group")
    axes[0].set_ylabel("Number of transactions")
    axes[0].bar_label(axes[0].containers[0])
    sns.barplot(x="Age group", y="Counts", data=age_not_fraud, ax=axes[1])
    axes[1].set_title("Number of non-fraudulent transactions per age group")
    axes[1].set_xlabel("Age group")
    axes[1].set_ylabel("Number of transactions")
    axes[1].bar_label(axes[1].containers[0])
    plt.tight_layout()
    plt.show()


state_to_region = {
            'ME': 'Northeast', 'NH': 'Northeast', 'VT': 'Northeast',
            'MA': 'Northeast', 'RI': 'Northeast', 'CT': 'Northeast',
            'NY': 'Northeast', 'NJ': 'Northeast', 'PA': 'Northeast',
            'OH': 'Midwest', 'IN': 'Midwest', 'IL': 'Midwest', 'MI': 'Midwest',
            'WI': 'Midwest', 'MN': 'Midwest', 'IA': 'Midwest', 'MO': 'Midwest',
            'ND': 'Midwest', 'SD': 'Midwest', 'NE': 'Midwest', 'KS': 'Midwest',
            'DE': 'South', 'MD': 'South', 'DC': 'South', 'VA': 'South',
            'WV': 'South', 'NC': 'South', 'SC': 'South',
            'GA': 'South', 'FL': 'South', 'AL': 'South', 'MS': 'South',
            'TN': 'South', 'KY': 'South', 'AR': 'South', 'LA': 'South',
            'OK': 'South', 'TX': 'South', 'MT': 'Rocky Mountains', 'WY': 'Rocky Mountains',
            'CO': 'Rocky Mountains', 'NM': 'Rocky Mountains', 'ID': 'Rocky Mountains',
            'UT': 'Rocky Mountains', 'WA': 'Far West', 'OR': 'Far West',
            'CA': 'Far West', 'HI': 'Far West', 'AK': 'Far West'
        }
train_data['Region'] = train_data.state.map(state_to_region)


print(train_data.head())  # This will show the first 5 rows of the train_data, including the 'Region' column



def Train_Fraud_nonFraud_Region():
    train_data = pd.read_csv(r"C:\Users\Metatron\Downloads\archive (1)\fraudTrain.csv")
    if 'Region' not in train_data.columns:
        state_to_region = {
            'ME': 'Northeast', 'NH': 'Northeast', 'VT': 'Northeast',
            'MA': 'Northeast', 'RI': 'Northeast', 'CT': 'Northeast',
            'NY': 'Northeast', 'NJ': 'Northeast', 'PA': 'Northeast',
            'OH': 'Midwest', 'IN': 'Midwest', 'IL': 'Midwest', 'MI': 'Midwest',
            'WI': 'Midwest', 'MN': 'Midwest', 'IA': 'Midwest', 'MO': 'Midwest',
            'ND': 'Midwest', 'SD': 'Midwest', 'NE': 'Midwest', 'KS': 'Midwest',
            'DE': 'South', 'MD': 'South', 'DC': 'South', 'VA': 'South',
            'WV': 'South', 'NC': 'South', 'SC': 'South',
            'GA': 'South', 'FL': 'South', 'AL': 'South', 'MS': 'South',
            'TN': 'South', 'KY': 'South', 'AR': 'South', 'LA': 'South',
            'OK': 'South', 'TX': 'South', 'MT': 'Rocky Mountains', 'WY': 'Rocky Mountains',
            'CO': 'Rocky Mountains', 'NM': 'Rocky Mountains', 'ID': 'Rocky Mountains',
            'UT': 'Rocky Mountains', 'WA': 'Far West', 'OR': 'Far West',
            'CA': 'Far West', 'HI': 'Far West', 'AK': 'Far West'
        }
        train_data['Region'] = train_data.state.map(state_to_region)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_theme()
    r_fraud = train_data[train_data.is_fraud == 1].Region.value_counts().reset_index()
    r_fraud.columns = ["Region", "Counts"]
    r_not_fraud = train_data[train_data.is_fraud == 0].Region.value_counts().reset_index()
    r_not_fraud.columns = ["Region", "Counts"]
    sns.barplot(x="Region", y="Counts", data=r_fraud, ax=axes[0])
    axes[0].set_title("Number of fraudulent transactions per region")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("Number of transactions")
    axes[0].bar_label(axes[0].containers[0])
    sns.barplot(x="Region", y="Counts", data=r_not_fraud, ax=axes[1])
    axes[1].set_title("Number of non-fraudulent transactions per region")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Number of transactions")
    axes[1].bar_label(axes[1].containers[0])
    plt.tight_layout()
    plt.show()
        
train_data.drop(axis=1, columns=["job", "merchant"], inplace=True)

train_data = pd.get_dummies(train_data, columns = ["category", "gender", "Region" ,"age_group"], drop_first=True)

train_data.drop(axis=1, columns=["city", "state"], inplace=True)

features = train_data.drop(axis=1, columns=["is_fraud"], inplace=False)
label = train_data["is_fraud"]

non_numeric_cols = features.select_dtypes(include=['object', 'category']).columns

high_cardinality_cols = [col for col in non_numeric_cols if features[col].nunique() > 100]  # Adjust threshold
print("High-cardinality columns:", high_cardinality_cols)
features = features.drop(columns=high_cardinality_cols)

non_numeric_cols = features.select_dtypes(include=['object', 'category']).columns

features = pd.get_dummies(features, columns=non_numeric_cols, drop_first=True)

print("Features preprocessing complete!")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


log_model = LogisticRegression(penalty="l2", fit_intercept=False, random_state=42, max_iter=1500)
log_model.fit(features, label)

test = pd.read_csv(r"C:\Users\Metatron\Downloads\archive (1)\fraudTest.csv")

test.columns

test.drop(axis=1, columns=["Unnamed: 0", "cc_num", "merchant", "first", "last", "street", "city", "zip", "lat", "long", "city_pop", "job", "trans_num", "unix_time", "merch_lat", "merch_long"], inplace=True)

test["Region"] = test.state.map(state_to_region)

test["trans_year"] = pd.Series(pd.to_datetime(test.trans_date_trans_time)).dt.year
test["dob"] = pd.Series(pd.to_datetime(test.dob)).dt.year
age = pd.Series(test.trans_year - test.dob)
test["age"] = age
bins = [10, 18, 35, 60, 100]
labels = ["14-18", "18-35", "35-60", "60+"]
test["age_group"] = pd.cut(test['age'], bins=bins, labels=labels, right=True)

test.drop(axis=1, columns=["dob", "trans_year", "trans_date_trans_time", "age", "state"], inplace=True)

test = pd.get_dummies(test, columns=["category", "gender", "Region", "age_group"], drop_first=True)

test_f = test.drop(axis=1, columns=["is_fraud"], inplace=False)
test_l = test["is_fraud"]

pred = log_model.predict(test_f)

# Accuracy score
accuracy = accuracy_score(test_l, pred)
accuracy

test["predictions"] = pred
actual_fraud = test.is_fraud.value_counts()[1]
pred_fraud = test[test.is_fraud == 1].predictions.value_counts()[1]
actual_fraud, pred_fraud

def Test_Fraud_ConfusionMatrix():
    cm = confusion_matrix(test_l, pred)
    matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_model.classes_)
    matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Fraud", "Fraud"])
    matrix.plot(cmap="coolwarm")
    for text in matrix.text_.ravel():
        text.set_color("black")
    plt.grid(False)
    plt.show()

def menu():
    while True:
        print("\nMenu:")
        print("1. Fraud vs Non-Fraud Transactions by Category")
        print("2. Fraud vs Non-Fraud Transactions by Gender")
        print("3. Fraud vs Non-Fraud Transactions by Age Group")
        print("4. Fraud vs Non-Fraud Transactions by Region")
        print("5. Show Confusion Matrix")  
        print("6. Exit")
        
        try:
            option = int(input("Enter your Option: "))
            if option == 1:
                Train_Fraud_nonFraud()  
            elif option == 2:
                Train_Fraud_nonFraud_Gender()  
            elif option == 3:
                Train_Fraud_nonFraud_Age()  
            elif option == 4:
                Train_Fraud_nonFraud_Region()    
            elif option == 5:
                Test_Fraud_ConfusionMatrix()  
            elif option == 6:
                print("Exiting the Program. Goodbye!")
                break
            else:
                print("Invalid option. Please select a number between 1 and 7.")
        except ValueError:
            print("Invalid input. Please enter a number.")
menu()

