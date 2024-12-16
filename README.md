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
![image](https://github.com/user-attachments/assets/d106a4e0-d1a9-41a9-a9f5-507795b71b77)
### Gender Analysis
![image](https://github.com/user-attachments/assets/02df87ce-da1c-4f3f-a399-3645898529f8)
### Residence Type Analysis
![image](https://github.com/user-attachments/assets/97df0028-51d4-4c33-bf3f-30406613450b)
### Married Status Analysis
![image](https://github.com/user-attachments/assets/750d2e89-78f5-4f63-b521-fb41e55c2e84)
### Work Type Analysis
![image](https://github.com/user-attachments/assets/b0cec264-946c-491d-93c2-76d6e09d37bd)
### Smoking Status Analysis
![image](https://github.com/user-attachments/assets/dea15d45-e79b-4ed3-aa9c-e11389b8d21b)
### Graphs for Heart Disease and Hypertension
![image](https://github.com/user-attachments/assets/080d6c3c-4535-441d-a1b4-8119205a0096)
![image](https://github.com/user-attachments/assets/9b73dd9e-accc-438b-b95d-261ff081c3dc)
### Bivariate Analysis
![image](https://github.com/user-attachments/assets/2dd3b80d-c050-492c-8031-c3ec11b4fd0e)
![image](https://github.com/user-attachments/assets/98a7b9d9-c6d6-4d9c-af8d-c57aea08c2b3)
![image](https://github.com/user-attachments/assets/27880919-afa5-45d0-9cb5-9b96dc4635a7)
### Countplots
![image](https://github.com/user-attachments/assets/7fe275b8-2e31-48fc-a790-347d3e8ad947)
![image](https://github.com/user-attachments/assets/e94dbdb7-e520-4dc8-b651-e2024869fe9f)
![image](https://github.com/user-attachments/assets/075b7530-7bf1-443c-a2a5-b6dc31140965)
![image](https://github.com/user-attachments/assets/31c36df6-043d-4957-a885-b2c982794bdd)
![image](https://github.com/user-attachments/assets/3cd3f591-ca15-4df1-866c-685450b7abcd)
### Exploring Need for Oversampling
![image](https://github.com/user-attachments/assets/7ca3d5a5-2b88-40ae-953c-1d9eb00b3687)
![image](https://github.com/user-attachments/assets/cd84e673-3f20-491e-a727-db613589f506)
### Model Training
![image](https://github.com/user-attachments/assets/a83d56ee-045b-42eb-b013-f2842adce5fb)
![image](https://github.com/user-attachments/assets/7a9c1b15-b70e-4189-ba11-20d535f8f9ff)
### Confusion Matrices of Testing Model
![image](https://github.com/user-attachments/assets/648d61b1-1059-4b25-a280-32dc9b7c2611)
![image](https://github.com/user-attachments/assets/da6bade7-6556-4487-93d6-796d96f85957)
### Receiver Operating Characteristic
![image](https://github.com/user-attachments/assets/1e1a00cc-069a-47bd-b324-fdd293919006)
![image](https://github.com/user-attachments/assets/8a7726b4-a123-4788-a58c-d9c6bca1a538)
### Comparison of ROC Curves
![image](https://github.com/user-attachments/assets/b8c806ed-18ed-4c19-8fa0-608e10819066)


## Conclusion
- All models perform significantly better than random chance (represented by the diagonal dashed line in the ROC curve)
- The ensemble methods consistently outperform the simpler algorithms
- There's a notable performance gap between the ensemble methods and traditional algorithms like Naive Bayes and Logistic Regression
- For future scope optimal performance, XGBoost or Random Forest would be the best choices as they demonstrate the highest and most consistent performance
- If computational resources are limited, KNN could serve as a good alternative with relatively strong performance

## Dataset Source
Kaggle: [(https://www.kaggle.com/code/saumyagupta2025/stroke-prediction-detailed-eda-7-ml-models/input)]
