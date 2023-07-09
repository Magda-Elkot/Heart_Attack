# Heart Attack Prediction
  This Jupyter Notebook contains the code for predicting heart attacks using machine learning algorithms.

# Dependencies
  The following libraries are imported for data analysis, visualization, and machine learning:

    pandas: for data manipulation and analysis.
    
    numpy: for numerical operations.
    
    matplotlib.pyplot: for creating visualizations.
    
    seaborn: for creating statistical graphics.
    
    plotly.express: for interactive visualizations.
    
    plotly.graph_objects: for creating interactive graphs.
    
    sklearn: for machine learning algorithms, preprocessing, and evaluation metrics.
    
    warnings: to ignore any warning messages.
    
    missingno: for visualizing missing values.

# Loading the Dataset
  The dataset is loaded using the pd.read_csv() function from the pandas library. 
  The dataset is stored in a DataFrame called df.

# Exploratory Data Analysis
  ## Dataset Overview
    df.head(): displays the first 5 rows of the dataset.
    
    df.info(): provides information about the dataset, including the data types of each column and the number of non-null values.
    
    df.isnull().sum(): calculates the total number of missing values in each column.
    
    df.columns: retrieves the column names of the dataset.
    
    df.shape: returns the dimensions of the dataset (number of rows, number of columns).
    
    df.describe(): displays descriptive statistics about the numerical columns in the dataset.
    
    df.duplicated().sum(): calculates the number of duplicated rows in the dataset.
    
    df.drop_duplicates(): removes duplicated rows from the dataset.

  ## Handling Missing Values
    sns.heatmap(df.isnull(), yticklabels=False, cmap='plasma'): visualizes the null values in the dataset using a heatmap.
    
    df.isnull().sum(): calculates the total number of missing values in each column.
    
    missingno.bar(df, color="b"): displays a bar chart showing the distribution of missing values in the dataset.

  ## Handling Categorical Variables
    The categorical variables in the dataset are: 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall', 'output'.
    
    df["thall"].value_counts(): displays the count of unique values in the 'thall' column.
    
    df[df["thall"] == 0]: filters the dataset based on a condition.
    
    df["thall"] = df["thall"].replace(0, 2): replaces the value 0 with 2 in the 'thall' column.
    
    df["caa"].value_counts(): displays the count of unique values in the 'caa' column.
    
    df[df["caa"] == 4]: filters the dataset based on a condition.
    
    df["caa"] = df["caa"].replace(4, 0): replaces the value 4 with 0 in the 'caa' column.

  ## Correlation Analysis
    df[numeric_var].corr(): calculates the correlation coefficients between the numerical variables.
    
    df[numeric_var].corr().iloc[:, [-1]]: retrieves the correlation between the numerical variables and the target variable.
    
    df[categoric_var].corr(): calculates the correlation coefficients between the categorical variables.
    
    df[categoric_var].corr().iloc[:, [-1]]: retrieves the correlation between the categorical variables and the target variable.

  ## Data Visualization
  ### Numerical Variables
    Histograms: Displays the distribution of numerical variables using histograms.
    
    sns.distplot(df[i], hist_kws=dict(linewidth=1, edgecolor="k"), bins=20): plots a histogram for the variable i.
    
    plt.title(i, fontdict=title_font): sets the title for the histogram.
    
    plt.xlabel(z, fontdict=axis_font): sets the x-axis label for the histogram.
    
    plt.ylabel("Density", fontdict=axis_font): sets the y-axis label for the histogram.

  ### Categorical Variables
    Pie Charts: Displays the distribution of categorical variables using pie charts.
    
    ax.pie(total_observation_values, labels=observation_values, autopct='%1.1f%%', startangle=110, labeldistance=1.1): plots a pie chart for the variable i.
    
    plt.title((i + "(" + z + ")"), fontdict=title_font): sets the title for the pie chart.

  ### Feature Scaling
    RobustScaler: Scales the numerical features using statistics that are robust to outliers.
    
    RobustScaler().fit_transform(df[numeric_var]): applies the RobustScaler transformation to the numerical variables.

  ### Outlier Detection and Treatment
    Boxplots: Displays the distribution of numerical variables using boxplots.
    
    plt.boxplot(df["age"]): plots a boxplot for the 'age' variable.

  ### Outlier Detection:
    z_scores_trtbps = zscore(df["trtbps"]): calculates the z-scores for the 'trtbps' variable.
    
    np.where(z_scores_trtbps > threshold): identifies the outliers based on a threshold.

  ### Outlier Treatment:
  #### Winsorization:
    stats.percentileofscore(df["trtbps"], 165): calculates the percentile of a given value.
    
    winsorize(df.trtbps, (0, (1 - winsorize_percentile_trtbps))): applies winsorization to the 'trtbps' variable.

  #### Transformation Operations
    np.log(df["oldpeak_winsorize"]): applies the natural logarithm transformation to the 'oldpeak_winsorize' variable.
    
    np.sqrt(df["oldpeak_winsorize"]): applies the square root transformation to the 'oldpeak_winsorize' variable.

  #### One-Hot Encoding
    pd.get_dummies(df_copy, columns=categoric_var[:-1], drop_first=True): applies one-hot encoding to the categorical variables, excluding the last variable ('output').

# Data Splitting
  train_test_split(X, y, test_size=0.1, random_state=3): splits the dataset into training and testing sets.
  ## Machine Learning Algorithms
    The following machine learning algorithms are implemented:

  ### Logistic Regression
    LogisticRegression(): initializes a Logistic Regression model.
    
    log_reg.fit(X_train, y_train): fits the Logistic Regression model to the training data.
    
    log_reg.predict(X_test): predicts the target variable for the test data.
    
    accuracy_score(y_test, y_pred): calculates the accuracy score for the predicted values.
    
    cross_val_score(log_reg, X_test, y_test, cv=10): performs cross-validation and calculates the accuracy scores
    .
    classification_report(y_test, y_pred): generates a classification report with precision, recall, F1-score, and support.

  ### Decision Tree
    DecisionTreeClassifier(random_state=5): initializes a Decision Tree model.
    
    dec_tree.fit(X_train, y_train): fits the Decision Tree model to the training data.
    
    dec_tree.predict(X_test): predicts the target variable for the test data.
    
    accuracy_score(y_test, y_pred): calculates the accuracy


  ### Support Vector Machine Algorithm
    The Support Vector Machine (SVM) algorithm is a method used for classification, regression, and outlier detection. 
    The following steps were taken to implement the SVM algorithm in this project:

      Initialize an SVM model with SVC(random_state = 5).
      
      Fit the SVM model to the training data using svc_model.fit(X_train, y_train).
      
      Predict the target variable for the test data using y_pred = svc_model.predict(X_test).
      
      Calculate the test accuracy score of the SVM using accuracy_score(y_test, y_pred).
      
      Calculate the cross-validation accuracy scores using cross_val_score(svc_model, X_test, y_test, cv = 10). 
      
      Create a confusion matrix using confusion_matrix(y_test, y_pred) and visualize it using sns.heatmap() function.
      
      Generate a classification report using metrics.classification_report(y_test, y_pred).
      
  ### Random Forest Algorithm
    The Random Forest algorithm is an ensemble learning method that constructs a multitude of decision trees at training time 
    and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. 
    The following steps were taken to implement the Random Forest algorithm in this project:

      Initialize a Random Forest model with RandomForestClassifier(random_state = 5).
      
      Fit the Random Forest model to the training data using random_forest.fit(X_train, y_train).
      
      Predict the target variable for the test data using y_pred_RF = random_forest.predict(X_test).
      
      Calculate the test accuracy score of the Random Forest using accuracy_score(y_test, y_pred_RF).
      
      Calculate the cross-validation accuracy scores using cross_val_score(random_forest, X_test, y_test, cv = 10).
      
      Create a confusion matrix using confusion_matrix(y_test, y_pred_RF) and visualize it using sns.heatmap() function.
      
      Generate a classification report using metrics.classification_report(y_test, y_pred_RF).
      
  ### Hyperparameter Optimization (with GridSearchCV)
    Hyperparameter optimization is the process of finding the best hyperparameters for a model. 
    In this project, the GridSearchCV method was used to find the best hyperparameters for the Random Forest algorithm. 
    The following steps were taken:

      Initialize a Random Forest model with RandomForestClassifier(random_state = 5).
      
      Create a dictionary of hyperparameters to tune with parameters = 
      {"n_estimators" : [50, 100, 150, 200], "criterion" : ["gini", "entropy"], 
      'max_features': ['auto', 'sqrt', 'log2'], 'bootstrap': [True, False]}.
      
      Perform a grid search with GridSearchCV(random_forest_new, param_grid = parameters).
      
      Fit the optimized Random Forest model to the training data using random_forest_new2.fit(X_train, y_train).
      
      Predict the target variable for the test data using y_pred = random_forest_new2.predict(X_test).
      
      Calculate the test accuracy score of the Random Forest using accuracy_score(y_test, y_pred).
      
      Create a confusion matrix using confusion_matrix(y_test, y_pred) and visualize it using sns.heatmap() function.
      
      Generate a classification report using metrics.classification_report(y_test, y_pred).

  ### Voting Classifier
    The Voting Classifier is an ensemble learning method that combines multiple models to improve the overall performance. 
    In this project, the Voting Classifier was implemented using the following steps:

      Import the necessary packages with from sklearn.ensemble import VotingClassifier.
      
      Initialize three classifiers with clf1 = GradientBoostingClassifier(), clf2 = LogisticRegression(), and clf3 = AdaBoostClassifier().
      
      Initialize the Voting Classifier with eclf1 = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft').
      
      Fit the Voting Classifier to the training data using eclf1.fit(X_train, y_train).
      
      Predict the target variable for the test data using predictions = eclf1.predict(X_test).
      
      Calculate the test accuracy score of the Voting Classifier using accuracy_score(y_test, predictions).
      
      Calculate the cross-validation accuracy scores using cross_val_score(eclf1, X_test, y_test, cv = 10).
      
      Create a confusion matrix using confusion_matrix(y_test, predictions) and visualize it using sns.heatmap() function.
      
      Generate a classification report using metrics.classification_report(y_test, predictions).
