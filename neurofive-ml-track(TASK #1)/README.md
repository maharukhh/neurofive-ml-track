# Titanic Exploratory Data Analysis (EDA)

This project is my first assignment in the **Neurofive Machine Learning Track**. The objective was to perform Exploratory Data Analysis (EDA) on the Titanic dataset using Python to understand the dataset before applying machine learning models.

---

## Dataset

**Dataset:** Titanic - Machine Learning from Disaster

Source: Kaggle

The dataset contains passenger information such as age, gender, ticket class, fare, and survival status.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Jupyter Notebook (VS Code)

---

## Tasks Completed

- Loaded the Titanic dataset using `pandas.read_csv()`
- Displayed the first five rows using `head()`
- Inspected the dataset using `info()`
- Generated statistical summaries using `describe()`
- Identified missing values
- Counted the number of rows and columns
- Distinguished numerical and categorical features
- Wrote a short summary of the dataset

---

## Key Findings

- Total Rows: **891**
- Total Columns: **12**
- Missing values found in:
  - Age (177)
  - Cabin (687)
  - Embarked (2)
- Numerical features:
  - PassengerId
  - Survived
  - Pclass
  - Age
  - SibSp
  - Parch
  - Fare
- Categorical features:
  - Name
  - Sex
  - Ticket
  - Cabin
  - Embarked

---

## Project Structure

```
neurofive-ml-track/
│── Titanic_EDA.ipynb
│── train.csv
│── README.md
```

---

## Learning Outcome

Through this project, I learned how to:

- Load datasets using Pandas
- Explore data structure
- Identify missing values
- Differentiate between numerical and categorical variables
- Perform basic Exploratory Data Analysis (EDA)
- Prepare a dataset for future machine learning tasks

---

## Author

**Mahrukh**
---

⭐ If you found this project useful, feel free to star the repository.
