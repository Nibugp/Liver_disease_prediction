# Liver_disease_prediction

### Business Problem

#### Problem Context
The prevalence of liver disease has been steadily increasing, largely due to factors such as excessive alcohol consumption, exposure to harmful gases, ingestion of contaminated food, and the use of certain drugs. This dataset was collected with the aim of evaluating predictive algorithms that could potentially alleviate the workload of healthcare professionals by identifying liver disease in patients more efficiently.

#### Dataset Overview
The dataset comprises 583 records, of which 416 represent patients diagnosed with liver disease, and 167 represent patients without liver disease. These records were gathered from patients in the North East region of Andhra Pradesh, India. The dataset includes information from 441 male and 142 female patients. For any patient whose age exceeds 89, the age is recorded as 90.

#### Features:
- **Age**: Age of the patient.
- **Gender**: Gender of the patient.
- **Total Bilirubin**: A measure of bilirubin levels in the blood. It is the key indicator of liver function and damage.
- **Direct Bilirubin**: The level of direct bilirubin in the blood.
- **Alkaline Phosphatase**: An enzyme related to the bile ducts; often increased when they are blocked.
- **Alamine Aminotransferase (ALT)**: An enzyme that helps convert proteins into energy for the liver cells.
- **Aspartate Aminotransferase (AST)**: An enzyme found in various tissues, including the liver.
- **Total Proteins**: The total amount of protein in the blood.
- **Albumin**: A protein made by the liver, low levels of which may indicate liver disease.
- **Albumin and Globulin Ratio**: The ratio between albumin and globulin proteins in the blood.
- **Dataset**: A label used to classify the records into two groups: liver disease (1) or no liver disease (0).

### Mapping Business Problem to Machine Learning Problem

#### Type of Machine Learning Problem
This is a binary classification problem. Given the set of features mentioned above, the goal is to predict whether a patient has liver disease or not.

#### Evaluation Metrics (KPIs)
Since this is a binary classification problem, the following metrics will be used for evaluation:

- **Confusion Matrix**: To provide a clear view of the number of correct and incorrect predictions made by the model.
- **ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)**: This metric evaluates the model's ability to distinguish between positive and negative classes by considering the rank of the output probabilities. AUC will be used to select the best-performing model, as it is well-suited for binary classification tasks.
