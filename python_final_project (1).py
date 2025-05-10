

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder





df = pd.read_csv('Human_Resources.csv')
df.head(5)





df.info()





df.describe()





df.nunique()





df.isnull().sum()





df.duplicated().sum()





df_cleaned = df.drop(columns=["EmployeeCount", "StandardHours", "Over18"], errors='ignore')





categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
label_encoders = {}





# Encode categorical variables and display mapping
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))  
    print(f"Encoding for '{col}': {label_encoders[col]}\n")




# Step 4: Detect and Remove Outliers Using IQR Method
columns_to_clean = [
    "MonthlyIncome", "NumCompaniesWorked", "TotalWorkingYears",
    "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager"
]





def detect_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return outliers





print("\n🔹 Outliers Detected:")
for col in columns_to_clean:
    outliers = detect_outliers(df_cleaned[col])
    if not outliers.empty:
        print(f"{col}: {len(outliers)} outliers | Sample: {outliers.head(5).tolist()}")





def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df_cleaned = remove_outliers(df_cleaned, columns_to_clean)








# Visualization 7: Leavers by Business Travel Frequency
plt.figure(figsize=(6, 6))
travel_attrition = df_cleaned.groupby('BusinessTravel')['Attrition'].sum()
plt.pie(travel_attrition, labels=travel_attrition.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'steelblue', 'lightskyblue'])
plt.title('Leavers by Business Travel Frequency')
plt.tight_layout()
plt.show()



# Visualization 9: Employees Left by Gender
plt.figure(figsize=(6, 6))
gender_attrition = df_cleaned.groupby('Gender')['Attrition'].sum()
plt.pie(gender_attrition, labels=None, autopct='%1.1f%%', startangle=90,
        colors=['steelblue', 'skyblue'], wedgeprops={'width': 0.3, 'edgecolor': 'white'})
plt.title('DASHBOARD: Employees Left by Gender')
plt.legend(labels=[f"{label} ({count})" for label, count in zip(gender_attrition.index, gender_attrition)],
           title='Gender', loc='center left', bbox_to_anchor=(1, 0.5))
plt.axis('equal')
plt.tight_layout()
plt.show()







# Step 5: Save the Cleaned Dataset
cleaned_file_path = "Human_Resources_Cleaned.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)
print(f"✅ Cleaned dataset saved as: {cleaned_file_path}")




