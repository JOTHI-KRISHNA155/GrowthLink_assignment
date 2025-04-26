**🚀 Methodology**
This project builds a hybrid machine learning model to predict customer churn (whether a customer will exit a bank) using a combination of Random Forest and Gradient Boosting within a Voting Classifier. Here's the detailed step-by-step methodology followed:

**1. 🧹 Data Preprocessing**
        Dropped irrelevant columns: RowNumber, CustomerId, and Surname.
        Converted categorical variables (Geography, Gender) into numerical features using One-Hot Encoding.

**2. ⚙️ Feature Engineering**
        Created a new interaction feature: Age_Tenure (Age × Tenure) to capture combined effects.
        Binned the Tenure feature into categorical groups (Short, Medium, Long, etc.) and applied one-hot encoding to them.

**3. 🧪 Train-Test Split**
        Split the dataset into 80% training and 20% testing using train_test_split from scikit-learn.

**4. 📈 Feature Scaling**
        Standardized the features using StandardScaler to ensure models are trained on normalized data.

**5. ⚖️ Handling Class Imbalance**
        Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training set to balance the minority and majority classes.

**6. 🤖 Model Building**
        Built two base models:
            Random Forest Classifier (200 trees, controlled depth and minimum samples)
            Gradient Boosting Classifier (200 estimators, controlled learning rate and depth)
            Combined them into a Voting Classifier using soft voting to leverage probabilities for better performance.

**7. 🔮 Model Training and Prediction**
        Trained the Voting Classifier on the resampled (balanced) training data.
        Predicted churn on the test set.

**8. 📊 Model Evaluation**
        Evaluated the model using:
            Accuracy Score : 85%
            Classification Report (Precision, Recall, F1-Score)
            Confusion Matrix visualized using Seaborn.
