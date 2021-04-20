# Heart Failure Prediction

This Capstone Project is the last project of the Azure Machine Learning nanodegree and I'm going to use the Heart Disease dataset to predict the disease. In this project, two models are created in the following way:
  1. First, Using AutoML model
  2. Then, using a customized model and tuning its hyperparameters with HyperDrive

After these steps are performed, we compare the performance of both models and deploy the best model. 

<hr/>

## TABLE OF CONTENTS
* [Project Set Up and Installation](#project-set-up-and-installation)
* [Projct Architecture](@project-architecture)
* [Dataset](#dataset)
  * [Overview](#overview)
  * [Task](#task)
  * [Access](#access)
* [Automated ML](#automated-ml)
  * [Results](#results)
* [Hyperparameter Tuning](#hyperparameter-tuning)
  * [Results](#results)
* [Model Deployment](#model-deployment)
* [Screen Recording](#screen-recording)
* [Standout Suggestions](standout-suggestions)
* [References](#references)
<hr/>

## Project Set Up and Installation
This project is done using Azure ML lab and a workspace was already provided to us. In order to start we need to do the following:

- Set up a compute instance, give it a name such `project-compute` with `STANDARD_DS2_V2` size.

## Dataset

### Overview
For this project, I am using the [Heart disease dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) from Kaggle. The term “heart disease” refers to several types of heart conditions. The most common type of heart disease in the United States is coronary artery disease (CAD), which affects the blood flow to the heart. Decreased blood flow can cause a heart attack. Sometimes heart disease may be “silent” and not diagnosed until a person experiences signs or symptoms of a heart attack, heart failure, or an arrhythmia. When these events happen, symptoms may include:

**Heart attack**: Chest pain or discomfort, upper back or neck pain, indigestion, heartburn, nausea or vomiting, extreme fatigue, upper body discomfort, dizziness, and shortness of breath.<br />
**Arrhythmia**: Fluttering feelings in the chest (palpitations).<br />
**Heart failure**: Shortness of breath, fatigue, or swelling of the feet, ankles, legs, abdomen, or neck veins.<br />

What are the risk factors for heart disease?
High blood pressure, high blood cholesterol, and smoking are key risk factors for heart disease. About half of Americans (47%) have at least one of these three risk factors. Several other medical conditions and lifestyle choices can also put people at a higher risk for heart disease, including:

- Diabetes
- Overweight and obesity
- Unhealthy diet
- Physical inactivity
- Excessive alcohol use

<p align="center">
<img src="heart1.png") /></p>
<p align="center">Figure 1. Heart Disease Information</p>


Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide. Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure. Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
````
Attribute information:

age - Age of the patient
anemia - Decrease of red blood cells or hemoglobin (boolean)
creatinine_phosphokinase - Level of the CPK enzyme in the blood (mcg/L)
diabetes - If the patient has diabetes (boolean)
ejection_fraction - Percentage of blood leaving the heart at each contraction (percentage)
high_blood_pressure - If the patient has hypertension (boolean)
platelets - Platelets in the blood (kiloplatelets/mL)
serum_creatinine - Level of serum creatinine in the blood (mg/dL)
serum_sodium - Level of serum sodium in the blood (mEq/L)
sex - Woman or man (binary)
smoking - If the patient smokes or not (boolean)
time - Follow-up period (days)
DEATH_EVENT - If the patient deceased during the follow-up period (boolean)
````

### Task
The goal of this project is to train the model to predict mortality caused by heart disease. (death=1, no death=0)
### Access
In Azure ML Studio, I registered the dataset from local files. I have the .csv file in my github repository and I downloaded it in the VM. For the train.py file I usef the link to my repo to create a Tabular Dataset.
![dataset](/starter_file/dataset.png)

## Automated ML
For the Compute Target, I used a 'STANDARD_D2_V2' vm_size with max_nodes=4. For the AutoML Configuration, I used the following settings:
![automl](/starter_file/automl.png)

experiment_timeout_minutes: I chose 15 minutes as the maximum amount of time to run the experiment

max_concurrent_iterations: maximum number of iterations that would be executed in parallel

n_cross_validations: To avoid overfitting, we need to user cross validation.

primary_metric: Accuracy.

task: Classification since we want a binary prediction or either 1 or 0.

#### AutoML Run

![automl-models-2](/starter_file/automl-models-2.png)
![automl-models](/starter_file/automl-models.png)

<b>AutoML Best Model</b>

![automl-best-model](/starter_file/automl-best-model.png)


#### AutoML Best model parameters

![automl-best-parameters](/starter_file/automl-best-parameters.png)

### Results
The best model was Voting Ensemble with an accuray of 0.8696. Voting Ensemble learning improves machine learning results and predictive performance by combining multiple models as opposed to using single models.

![run-details](/starter_file/run-details.png)

![run-details-2](/starter_file/run-details-2.png)

![best-model](/starter_file/best-model.png)

### Future Improvement Suggestions
To improve the autoML, I can disable the early stopping and increase experiment_timeout_minutes. Also choose another primary_metric such as AUC instead of Accuracy because AUC is better way to measure performance than accuracy because it does not bias on size of test or evaluation data.

## Hyperparameter Tuning

For this technique, I decided to choose logistic regression for the following reason:

* It is the most basic algorithm when it comes to classification and one should always start from basic models
* It is easy to understand the results and simple to train
* The execution time is very fast

C - The inverse regularization strength. max_iter | The maximum iteration to converge for the SKLearn Logistic Regression.

Then, I use random parameter sampling to sample over a discrete set of values. Random parameter sampling is great for discovery and getting hyperparameter combinations that you would not have guessed intuitively, although it often requires more time to execute.

The parameter search space used for C is [0.01, 0.1, 1.0, 10.0, 100.0] and for max_iter is [20, 50, 100, 120, 150]


### Results

![hd-run-details](/starter_file/hd-run-details.png)
![hd-run-details-2](/starter_file/hd-run-details-2.png)
![hd-run-details-3](/starter_file/hd-run-details-3.png)
![hd-run-details-4](/starter_file/hd-run-details-4.png)

As we can see from the screenshot above, HyperDrive tested many combinations of C and max_iter and the highest accuracy that our Logistic Regression Model acheived was 0.7888888888888889. To improve these results, we should handle the class imbalance in the dataset. Also, I could've increased max_total_runs so that I can optimize the parameters more and use a different primary metric to maximize, similar to the automl model. Lastly, I could use Bayesian sampling insted of Random Sampling as it picks the next sample of hyperparameters, based on how the previous samples performed.

### Future Improvement Suggestions
I could have used Bayesian sampling insted of Random Sampling as it picks the next sample of hyperparameters, based on how the previous samples performed. Also, I could have increased the the max_total_runs so it can optimize the parameters.

## Model Deployment
Based on the previous results, I chose the Voting Ensemble model as it has the best Accuracy out of the two. To successfully deploy the model, we must have an InferenceConfig and an ACI Config.

![model-deployment](/starter_file/model-deployment.png)
![model-deployment-2](/starter_file/model-deployment-2.png)

From the screenshot above, we can also verify from the azure portal that the model was successfully deployed and is in a healthy state.

After this, we tested the model endpoint by providing dummy data to see the results. Below is the screenshot of the test data used to test the endoint:

![model-deployment-3](/starter_file/model-deployment-3.png)

![model-deployment-4](/starter_file/model-deployment-4.png)

In the screenshot above, we are using values from the data to test the model. The model returns the output as 0 and 1. This means that based on Voting Ensemble model, if a patient has an output of 0 then it means she/he won't die from heart disease. However, the second output is 1, which means the patient most likely die due to heart disease.


## Screen Recording

https://youtu.be/ZXhv1zNwFdU

## Standout Suggestions
Choose another primary metric like "AUC Weighted" instead of accuracy
Choose another Classifier instead of Logistic Regression.
Feature Selection can be performed to select only thsoe features that positively contribute to the prediction of the outcome variable


## References
Centers for Disease Control and Prevention. Underlying Cause of Death, 1999–2018. CDC WONDER Online Database. Atlanta, GA: Centers for Disease Control and Prevention; 2018. Accessed March 12, 2020.

Virani SS, Alonso A, Benjamin EJ, Bittencourt MS, Callaway CW, Carson AP, et al. Heart disease and stroke statistics—2020 update: a report from the American Heart Associationexternal icon. Circulation. 2020;141(9):e139–e596.

Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020) https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5

Figure 1. Fist Choice Neurology. https://www.healthcentral.com/condition/heart-disease



