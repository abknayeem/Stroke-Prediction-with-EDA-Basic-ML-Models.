# Stroke Prediction with EDA and Basic Machine Learning Models

## Project Overview
This project aims to predict the probability of a brain stroke using machine learning techniques. By leveraging medical data, seven machine learning models will be trained to recognize patterns and risk factors linked to stroke. The findings will support early detection, offering crucial insights for prevention and timely treatment.

## Dataset Overview
The dataset includes various columns such as:
- Id: Unique identification number for each patient.
- Age: The age of the patient.
- Hypertension: Whether the patient has hypertension (1 for yes, 0 for no).
- Heart Disease: Whether the patient has a history of heart disease (1 for yes, 0 for no).
- Ever Married: Whether the patient has been married (1 for yes, 0 for no).
- Work Type: The type of work the patient does (e.g., private, self-employed, government, children).
- Residence Type: Whether the patient resides in an urban or rural area.
- Glucose Level: The patient's glucose level.
- BMI (Body Mass Index): A measure of body fat based on height and weight.
- Smoking Status: The smoking habit of the patient (e.g., never smoked, formerly smoked, currently smoking).
- Stroke: The target variable indicates whether the patient had a stroke (1 for yes, 0 for no).

## Analysis Points
- **Probability Level:** Calculate the likelihood of certain outcomes, such as survival rate, based on various risk factors in the dataset. This helps estimate the probability of survival or risk for different individuals and categories.
-  **Analysis of Health Risk:** Identify which factors significantly influence lung cancer mortality risk among categories.
-  **Analysis with KNN Accuracy:** Measure the accuracy of the KNN classifier in predicting survival status based on patient health data.
-  **Analysis with Decision Tree Classifier:** Identify important health features that influence survival and assess the classifier’s accuracy in predicting survival outcomes.
-  **Analysis with Random Forest Classifier:** Improve predictive accuracy and robustness by using an ensemble of Decision Trees with the Random Forest classifier.
Analysis with AdaBoost Classifier:** Focus on instances that are difficult to classify correctly to Increase the model’s predictive power.

## Visualizations

### Correlation Heatmap of Columns
![image](https://github.com/user-attachments/assets/e16bb1dd-d269-4d9f-a0c5-105070e9c40c)
### Gender Analysis
![image](https://github.com/user-attachments/assets/50d849f2-a23f-4196-8000-ef41a1c07d84)
### Residence Type Analysis
![image](https://github.com/user-attachments/assets/475ac011-a7b8-48d9-804c-99ebdb5fb582)
### Married Status Analysis
![image](https://github.com/user-attachments/assets/7125ce4c-b3bd-4de1-8d4a-200f9e8e2ecc)
### Work Type Analysis
![image](https://github.com/user-attachments/assets/2a0356e7-9cbe-49b9-a30a-f5b642a1a8c0)
### Smoking Status Analysis
![image](https://github.com/user-attachments/assets/8a326d2b-d2dd-4857-8e7f-e19c4ac57f2f)
### Graphs for Heart Disease and Hypertension
![image](https://github.com/user-attachments/assets/ef96d168-8bd1-4b48-aca9-7eb0dad4fdbe)
![image](https://github.com/user-attachments/assets/6f07f094-4c39-4627-a4de-72e810b3f14a)
### Bivariate Analysis
![image](https://github.com/user-attachments/assets/79dcc7bc-30ac-46f9-a717-36fd7d3f4067)
![image](https://github.com/user-attachments/assets/2968551f-e475-45b3-b6c2-613aa4b8eca9)
![image](https://github.com/user-attachments/assets/8165fe17-bacc-4e7e-8959-abff5281ddef)
### Receiver Operating Characteristic
![image](https://github.com/user-attachments/assets/8e4cbba4-8de1-4de2-bede-8a74a51b5683)
![image](https://github.com/user-attachments/assets/0aaa583f-8390-4f11-bc2a-632a8c61c3da)
### Comparison of ROC Curves
![image](https://github.com/user-attachments/assets/50f501c7-4c34-4928-aa1b-f6f95ed59551)


## Conclusion
- All models perform significantly better than random chance (represented by the diagonal dashed line in the ROC curve)
- The ensemble methods consistently outperform the simpler algorithms
- There's a notable performance gap between the ensemble methods and traditional algorithms like Naive Bayes and Logistic Regression
- For future scope optimal performance, XGBoost or Random Forest would be the best choices as they demonstrate the highest and most consistent performance
- If computational resources are limited, KNN could serve as a good alternative with relatively strong performance

## Dataset Source
Kaggle: [(https://www.kaggle.com/code/saumyagupta2025/stroke-prediction-detailed-eda-7-ml-models/input)]
