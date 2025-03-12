# Model Card  

## Model Details  
- **Developed by:** Jennifer Garza  
- **Model date:** March 12, 2025  
- **Model version:** 1.0  
- **Model type:** Binary Classification (Predicting income level)  
- **Features:**  
  - **Categorical:** workclass, education, marital-status, occupation, relationship, race, sex, native-country  
  - **Numerical:** Age, hours per week, and other continuous variables  

## Intended Use  
- **Primary Intended Uses:**  
  - Predict whether an individual earns more than $50K/year based on demographic and socioeconomic factors.  
  - Assist researchers and analysts in studying income distribution and socioeconomic factors.  
- **Primary Intended Users:**  
  - Data scientists, economists, and policymakers analyzing income trends.  
- **Out-of-Scope Uses:**  
  - Any use that could lead to discriminatory decisions or reinforce biases based on protected attributes.  
  - Automated decision-making in hiring, lending, or other high-stakes applications without fairness mitigations.  

## Training Data  
- **Source:** UCI Census Income dataset  
- **Training Data Split:** 80% of the dataset for training  
- **Preprocessing:**  
  - Categorical variables encoded using one-hot encoding  
  - Target variable binarized (income ≤50K vs. >50K)  
  - Standardization applied to numerical features  

## Evaluation Data  
- **Source:** 20% of the UCI Census Income dataset  
- **Preprocessing:** Identical to training data  

## Model Performance  
- **Evaluation Metrics:**  
  - **Precision:** 0.7962  
  - **Recall:** 0.5372  
  - **F1 Score:** 0.6416  
- **Interpretation:**  
  - The model has high precision but lower recall, meaning it minimizes false positives but may fail to identify all high-income individuals.  
  - The imbalance between precision and recall suggests the model may underpredict income >$50K in certain cases.  

## Ethical Considerations  
- **Performance disparities across demographic groups:**  
  - **Education level:** The model performs better for individuals with higher education.  
  - **Gender:** Higher accuracy for male samples compared to female.  
  - **Occupation:** Higher-income occupations have better predictions.  
- **Risk of reinforcing socioeconomic inequalities:**  
  - Without intervention, the model could contribute to biased decisions if used in hiring, lending, or other financial assessments.  
- **Mitigation Strategies:**  
  - Conduct fairness analysis and apply bias mitigation techniques (e.g., reweighting, adversarial debiasing).  
  - Use fairness-aware post-processing to adjust predictions.  
  - Deploy in a human-in-the-loop system rather than full automation.  

## Caveats and Recommendations  
- **Potential Biases:**  
  - Biases related to gender, education, and occupation may lead to unfair outcomes.  
  - Small sample sizes for certain demographic groups could result in unreliable predictions.  
  - The dataset is based on historical census data, which may not reflect current trends.  
- **Recommendations for Improvement:**  
  - **Fairness Constraints:** Implement fairness-aware constraints during training.  
  - **Data Collection:** Gather more representative data for underrepresented groups.  
  - **Continuous Monitoring:** Regularly retrain and validate the model with updated data.  
  - **Human Oversight:** Use model predictions to assist decision-making rather than automate critical processes.  



This version improves clarity, highlights risks, and provides actionable recommendations. Let me know if you need modifications! 🚀