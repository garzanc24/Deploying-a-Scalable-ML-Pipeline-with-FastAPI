# Model Card

## Model Details
* Developed by: Jennifer Garza
* Model date: 3/12/2025
* Model version: 1.0
* Model type: 
* Features: Demographic data including workclass, education, marital-status, occupation, relationship, race, sex, native-country, and numeric features.


## Intended Use
* Primary intended uses: Predict whether income exceeds $50K/year based on census data.
* Primary intended users: Researchers and analysts studying socioeconomic factors.
* Out-of-scope uses: Any use that could lead to discriminatory decisions or practices based on protected attributes.

## Training Data
* Source: UCI Census Income dataset
* Training data split: 80% of the original dataset
* Preprocessing: Categorical features encoded using one-hot encoding, label binarized for the salary target.

## Evaluation Data
* Source: 20% of the original UCI Census Income dataset
* Preprocessing: Identical to training data

## Metrics
* Model performance measures:
  * Precision: 0.7962
  * Recall: 0.5372
  * F1 Score: 0.6416
* The model has higher precision than recall, meaning it's more likely to miss positive cases than to incorrectly classify negative ones.

## Ethical Considerations
* Significant performance disparities exist across different demographic groups:
  * Education level shows major disparities, with better performance for those with higher education
  * Gender disparity exists, with better performance on male samples than female
  * Occupation shows significant variance, with better performance on higher-income occupations
* These disparities could reinforce existing socioeconomic inequalities if the model is used for decision-making without acknowledging these limitations.
* Further fairness analysis and mitigation strategies are recommended before deployment.

## Caveats and Recommendations
* The model shows signs of bias particularly related to education level, gender, and occupation.
* Small sample sizes for some feature values (e.g., certain countries of origin) may lead to unreliable metrics.
* The dataset is based on census data which may not reflect current demographic and economic conditions.
* Consider:
  * Implementing fairness constraints or post-processing techniques to mitigate bias
  * Collecting more representative data for underrepresented groups
  * Using this model as part of a human-in-the-loop system rather than for automated decisions
  * Regularly retraining the model with updated data