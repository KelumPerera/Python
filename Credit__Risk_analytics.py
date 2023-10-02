
"""
Explore the credit data
Begin by looking at the data set cr_loan. In this data set, loan_status shows whether the loan is currently in default with 1 being default and 0 being non-default.

You have more columns within the data, and many could have a relationship with the values in loan_status. You need to explore the data and these relationships more with further analysis to understand the impact of the data on credit loan defaults.

Checking the structure of the data as well as seeing a snapshot helps us better understand what's inside the set. Similarly, visualizations provide a high level view of the data in addition to important trends and patterns.

The data set cr_loan has already been loaded in the workspace.
#   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   person_age                  32581 non-null  int64  
 1   person_income               32581 non-null  int64  
 2   person_home_ownership       32581 non-null  object 
 3   person_emp_length           31686 non-null  float64
 4   loan_intent                 32581 non-null  object 
 5   loan_grade                  32581 non-null  object 
 6   loan_amnt                   32581 non-null  int64  
 7   loan_int_rate               29465 non-null  float64
 8   loan_status                 32581 non-null  int64  
 9   loan_percent_income         32581 non-null  float64
 10  cb_person_default_on_file   32581 non-null  object 
 11  cb_person_cred_hist_length  32581 non-null  int64  
dtypes: float64(3), int64(5), object(4)
"""

# Check the structure of the data
print(cr_loan.dtypes)

# Check the first five rows of the data
print(cr_loan.head())

# Look at the distribution of loan amounts with a histogram
n, bins, patches = plt.hist(x=cr_loan['loan_amnt'], bins='auto', color='blue',alpha=0.7, rwidth=0.85)
plt.xlabel("Loan Amount")
plt.show()

"""
Create a scatter plot of a person's income and age. In this case, income is the independent variable and age is the dependent variable
"""

print("There are 32 000 rows of data so the scatter plot may take a little while to plot.")

# Plot a scatter plot of income against age
plt.scatter(cr_loan['person_income'], cr_loan['person_age'],c='blue', alpha=0.5)
plt.xlabel('Personal Income')
plt.ylabel('Persone Age')
plt.show()




"""
Crosstab and pivot tables
Often, financial data is viewed as a pivot table in spreadsheets like Excel.

With cross tables, you get a high level view of selected columns and even aggregation like a count or average. For most credit risk models, especially for probability of default, columns like person_emp_length and person_home_ownership are common to begin investigating.

You will be able to see how the values are populated throughout the data, and visualize them. For now, you need to check how loan_status is affected by factors like home ownership status, loan grade, and loan percentage of income.

The data set cr_loan has been loaded in the workspace.
"""

# Create a cross table of the loan intent and loan status
print(pd.crosstab(cr_loan['loan_intent'], cr_loan['loan_status'], margins = True))

# Create a cross table of home ownership, loan status, and grade
print(pd.crosstab(cr_loan['person_home_ownership'],[cr_loan['loan_status'],cr_loan['loan_grade']]))

# Create a cross table of home ownership, loan status, and average percent income
print(pd.crosstab(cr_loan['person_home_ownership'], cr_loan['loan_status'],values=cr_loan['loan_percent_income'], aggfunc='mean'))


"""
You will be able to see how the values are populated throughout the data, and visualize them. For now, you need to check how loan_status is affected by factors like home ownership status, loan grade, and loan percentage of income.
"""
# Create a box plot of percentage income by loan status
cr_loan.boxplot(column = ['loan_percent_income'], by = 'loan_status')
plt.title('Average Percent Income by Loan Status')
plt.suptitle('')
plt.show()


"""
As with any machine learning problem, data preparation is the first step. But why? When our data is properly prepared we reduce the training time of our machine learning models. Also, prepared data can also have a positive impact on the performance of our model. This is important because we want our models to predict defaults correctly as often as possible. Consider this ROC chart. This shows the accuracy of three different models on the same data throughout different stages of processing. The light blue line represents a model trained on tidy and prepared data, while the orange line's model trained on raw data. The light blue line represents the most accurate model, because the curve is closest to the top left corner. We will see more graphs like this later when we check the accuracy of our models.

The first type of preparation we will look at is outlier detection and removal. Unfortunately, data entry systems producing bad data is fairly common. If the data entry specialist was tired or distracted, they can enter incorrect values into our system. It's also possible for data ingestion tools to create erroneous values in our data as a result of technical problems or system failures.

With outliers in our training data, our predictive models will have a difficult time estimating parameters like coefficients. This can cause our models to not predict as many defaults. Think of the coefficients as how much each column or feature is weighted to determine the loan status. Notice the coefficient differences in this example. It's possible that outliers in interest rate can cause that column to be weighted much more than normal. This will affect predictions.

One way we can detect outliers, is to use cross tables with aggregate functions like those from the previous video. Here, we call crosstab on our credit loan data just like before to find the average interest rate. For this example, we might expect to see the values on the left with our normal data. However, there could be some extreme outliers in the data which would result in the data on the right. This would cause problems with modeling. Imagine having an interest rate of 59,000 percent!

Another way to detect outliers is to use visuals. For this we can easily use plots like histograms and scatter plots, which we saw in the previous video. Here, we can see that a couple records have a person's employment length set at well over 100. This would suggest that two loan applicants are over 136 years old! This, for now at least, is not possible.

So, we know outliers are a problem and want to remove them, but how? We can easily use the drop method within the pandas package to remove rows from our data. In this example, we first use basic python subsetting to find rows with a person's employment length greater than 60. What this returns is the index position of that row in our data frame. From there, we call the drop method on our data frame to have it remove the rows in the data frame which match the index positions found earlier. Now, we can see visually that the outliers have been removed according to our criteria, and the data looks much more realistic.

"""

"""
Finding outliers with cross tables
Now you need to find and remove outliers you suspect might be in the data. For this exercise, you can use cross tables and aggregate functions.

Have a look at the person_emp_length column. You've used the aggfunc = 'mean' argument to see the average of a numeric column before, but to detect outliers you can use other functions like min and max.

It may not be possible for a person to have an employment length of less than 0 or greater than 60. You can use cross tables to check the data and see if there are any instances of this!

The data set cr_loan has been loaded in the workspace.
"""

# Create the cross table for loan status, home ownership, and the max employment length
print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],values=cr_loan['person_emp_length'], aggfunc='max'))

# Create an array of indices where employment length is greater than 60
indices = cr_loan[cr_loan['person_emp_length'] > 60].index

# Drop the records from the data based on the indices and create a new dataframe
cr_loan_new = cr_loan.drop(indices)

# Create the cross table from earlier and include minimum employment length
print(pd.crosstab(cr_loan_new['loan_status'],cr_loan_new['person_home_ownership'],values=cr_loan_new['person_emp_length'], aggfunc=['min','max']))


"""
Visualizing credit outliers
You discovered outliers in person_emp_length where values greater than 60 were far above the norm. person_age is another column in which a person can use a common sense approach to say it is very unlikely that a person applying for a loan will be over 100 years old.

Visualizing the data here can be another easy way to detect outliers. You can use other numeric columns like loan_amnt and loan_int_rate to create plots with person_age to search for outliers.
"""

# Create the scatter plot for age and amount
plt.scatter(cr_loan['person_age'], cr_loan['loan_amnt'], c='blue', alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()

# Use Pandas to drop the record from the data frame and create a new one
cr_loan_new = cr_loan.drop(cr_loan[cr_loan['person_age'] > 100].index)

# Create a scatter plot of age and interest rate
colors = ["blue","red"]
plt.scatter(cr_loan_new['person_age'], cr_loan_new['loan_int_rate'],
            c = cr_loan_new['loan_status'],
            cmap = matplotlib.colors.ListedColormap(colors),
            alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Interest Rate")
plt.show()



"""
2. What is missing data?
Normally, you might think of missing data as when an entire row is missing, but that is not the only way data can be missing. Data can be missing when there are null values in place of actual values. It can also be an empty string instead of a real string. For this course, we will refer to missing data as when specific values are not present, not when entire rows of data are missing. Any of the columns within our data can contain missing values. If we see a row of data with missing values in a Pandas dataframe, it will look something like this. Notice for employment length we see NAN, or not a number, instead of a value.

3. Similarities with outliers
One issue with missing data is similar to problems caused with outliers in that it negatively impacts predictive model performance. It can bias our model in unanticipated ways, which can affect how we predict defaults. This could result in us predicting a large number of defaults that are not actually defaults because the model is biased towards defaults. Also, many machine learning models in Python do not automatically ignore missing values, and will often throw errors and cease training.
4. Similarities with outliers
Here are some examples of missing data and possible results. If there are null values in numeric or string columns, the model will throw an error.

5. How to handle missing data
So, how do we handle missing data? Most often, it is handled in one of three ways. Sometimes we need to replace missing values. This could be replacing a null with the average value of that column. Other times we remove the row with missing data all together. For example, if there are nulls in loan amount, we should drop those rows entirely. We sometimes keep missing values as well. This, however, is not the case with most loan data. Understanding the data will direct you towards one of these three actions.

6. How to handle missing data
For example, if the loan status is null, it's possible that the loan was recently processed in our system. Sometimes there is a data delay, and additional time needed for processing. In this case, we should just remove the whole row. Another example is where the person's age is missing. Here, we might be able to replace the missing age values with the median of everyone's age.

7. Finding missing data
But how do we find missing data? With Pandas, we can find missing data like nulls using the isnull function and the sum function to count the rows with data missing. By combining the functions isnull, sum, and any, we count all the null values in each column. This produces a table of values show the count of records with nulls in the data.

8. Replacing Missing data
If we decide to replace missing data, we can call the fill-n-a method from Pandas along with aggregate functions. This will replace only missing values. In this example, we replace null interest rates with the average of all interest rates in the data. The result, as shown here, replaces a null interest rate with 11 percent.

9. Dropping missing data
Dropping rows with missing data is just like dropping rows with outliers like in the previous video. We use the drop method from Pandas. Here, we find the rows with missing data using isnull, and then drop the rows from the data set entirely.

"""


"""
Replacing missing credit data
Now, you should check for missing data. If you find missing data within loan_status, you would not be able to use the data for predicting probability of default because you wouldn't know if the loan was a default or not. Missing data within person_emp_length would not be as damaging, but would still cause training errors.

So, check for missing data in the person_emp_length column and replace any missing values with the median.

Print an array of column names that contain missing data using .isnull().
Print the top five rows of the data set that has missing data for person_emp_length.
Replace the missing data with the median of all the employment length using .fillna().
Create a histogram of the person_emp_length column to check the distribution.
"""

# Print a null value column array
print(cr_loan.columns[cr_loan.isnull().any()])

# Print the top five rows with nulls for employment length
print(cr_loan[cr_loan['person_emp_length'].isnull()].head())

# Impute the null values with the median value for all employment lengths
cr_loan['person_emp_length'].fillna((cr_loan['person_emp_length'].mean()), inplace=True)

# Create a histogram of employment length
n, bins, patches = plt.hist(cr_loan['person_emp_length'], bins='auto', color='blue')
plt.xlabel("Person Employment Length")
plt.show()

# Print a null value column array
print(cr_loan.columns[cr_loan.isnull().any()])

# Print the top five rows with nulls for employment length
print(cr_loan[cr_loan['person_emp_length'].isnull()].head())

# Impute the null values with the median value for all employment lengths
cr_loan['person_emp_length'].fillna((cr_loan['person_emp_length'].median()), inplace=True)

# Create a histogram of employment length
n, bins, patches = plt.hist(cr_loan['person_emp_length'], bins='auto', color='blue')
plt.xlabel("Person Employment Length")
plt.show()

"""
Print the number of records that contain missing data for interest rate.
Create an array of indices for rows that contain missing interest rate called indices.
Drop the records with missing interest rate data and save the results to cr_loan_clean.
"""

# Print the number of nulls
print(cr_loan['loan_int_rate'].isnull().sum())

# Store the array on indices
indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index
indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index

# Save the new data without missing data
cr_loan_clean = cr_loan.drop(indices)


"""
Missing data intuition
Here's an intuition check! When handling missing data, you have three options: keep, replace, and remove.

You've been looking at numeric columns, but what about a non-numeric column? How would you handle missing data in the column person_home_ownership which has string values?

The object ownership_table has already been created to show how many records occur in each unique value of person_home_ownership with the following code:

# Count the number of records for each unique value
cr_loan['person_home_ownership'].value_counts()
ownership_table and cr_loan are already loaded in the workspace.
"""


"""
1. Logistic regression for probability of default
Now that we've removed both outliers and missing data from out data set, we can begin modeling to predict the probability of default.

2. Probability of default
Recall that the probability of default is the likelihood that someone will fail to repay a loan. This is expressed as a probability which is a value between zero and one. These probabilities are associated with our loan status column where a 1 is a default, and a 0 is a non default.

3. Probability of default
The resulting predictions give us probabilities of default. The closer the value is to 1, the higher the probability of the loan being a default.

4. Predicting probabilities
To get these probabilities, we train machine learning models on our credit data columns, known as features, so the models learn how to use the data to predict the probabilities. These types of models are known as classification models, where the class is default or non-default. In the industry, two models are used frequently. These are logistic regressions, and decision trees. Both of these models can predict the probability of default, and tell us how important each column is for predictions.

5. Logistic regression
The logistic regression is like a linear regression but only produces a value between 0 and 1. Notice that the equation for the linear regression is actually part of the logistic regression. Logistic regressions perform better on data when what determines a default or non-default can vary greatly. Think about the y-intercept here, which is the log odds of non-default. This as another way of expressing the overall probability of non-default.

6. Training a logistic regression
In this course, we use the logistic regression within scikit learn. The use of the model is easy. Like any function, you can pass in parameters or not. The solver parameter is an optimizer, just like the solver in Excel. LBFGS is the default. To train the model, we call the fit method on it. Within the method, we have to provide the model with training columns and training labels. We use ravel from numpy to make the labels a one-dimensional array instead of a data frame. In our credit data, the training columns are every column except the loan status. The loan status contains the labels.

7. Training and testing
Generally, in machine learning, we split our entire data set into two individual data sets.

8. Training and testing
They are the training set and the test set. We use the majority of the data to train our models, so they learn as much as possible from the data. Our test set is used to see how our model reacts to new data that it has not seen before. This is like students learning in school. They will learn facts from one subject, and be tested on different facts from that same subject. This way, we can asses their mastery of the topic.

9. Creating the training and test sets
The first thing we do is separate our data into training columns and labels. Here, we have assigned those as X and Y. With that done, we use the test train split function within the sci-kit learn package. Let's have a look at the code. Remember how I said we need training columns and labels for our model? We need these for both the training set and the test set, which are all easily created with one line of code. Within this function, we set the percentage of the data to be used as a test set, and a number used as a random seed for reproducibility.
"""


"""
Logistic regression basics
You've now cleaned up the data and created the new data set cr_loan_clean.

Think back to the final scatter plot from chapter 1 which showed more defaults with high loan_int_rate. Interest rates are easy to understand, but what how useful are they for predicting the probability of default?

Since you haven't tried predicting the probability of default yet, test out creating and training a logistic regression model with just loan_int_rate. Also check the model's internal parameters, which are like settings, to see the structure of the model with this one column.

The data cr_loan_clean has already been loaded in the workspace.
"""

"""
Create the X and y sets using the loan_int_rate and loan_status columns.
Create and fit a logistic regression model to the training data and call it clf_logistic_single.
Print the parameters of the model with .get_params().
Check the intercept of the model with the .intercept_ attribute.
"""

# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate']]
y = cr_loan_clean[['loan_status']]

# Create and fit a logistic regression model
clf_logistic_single = LogisticRegression()
clf_logistic_single.fit(X, np.ravel(y))

# Print the parameters of the model
print(clf_logistic_single.get_params())

# Print the intercept of the model
print(clf_logistic_single.intercept_)


"""
Multivariate logistic regression
Generally, you won't use only loan_int_rate to predict the probability of default. You will want to use all the data you have to make predictions.

With this in mind, try training a new model with different columns, called features, from the cr_loan_clean data. Will this model differ from the first one? For this, you can easily check the .intercept_ of the logistic regression. Remember that this is the y-intercept of the function and the overall log-odds of non-default.
"""


"""
Create a new X data set with loan_int_rate and person_emp_length. Store it as X_multi.
Create a y data set with just loan_status.
Create and .fit() a LogisticRegression() model on the new X data. Store it as clf_logistic_multi.
Print the .intercept_ value of the model
"""

# Create X data for the model
X_multi = cr_loan_clean[['loan_int_rate','person_emp_length']]

# Create a set of y data for training
y = cr_loan_clean[['loan_status']]

# Create and train a new logistic regression
clf_logistic_multi = LogisticRegression(solver='lbfgs').fit(X_multi, np.ravel(y))

# Print the intercept of the model
print(clf_logistic_multi.intercept_)


"""
Creating training and test sets
You've just trained LogisticRegression() models on different columns.

You know that the data should be separated into training and test sets. test_train_split() is used to create both at the same time. The training set is used to make predictions, while the test set is used for evaluation. Without evaluating the model, you have no way to tell how well it will perform on new loan data.

In addition to the intercept_, which is an attribute of the model, LogisticRegression() models also have the .coef_ attribute. This shows how important each training column is for predicting the probability of default.
"""

"""
Create the data set X using interest rate, employment length, and income. Create the y set using loan status.
Use train_test_split() to create the training and test sets from X and y.
Create and train a LogisticRegression() model and store it as clf_logistic.
Print the coefficients of the model using .coef_.
"""

# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate','person_emp_length','person_income']]
y = cr_loan_clean[['loan_status']]

# Use test_train_split to create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.4, random_state=123)

# Create and fit the logistic regression model
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Print the models coefficients
print(clf_logistic.coef_)

"""
Nicely done! Do you see that three columns were used for training and there are three values in .coef_? This tells you how important each column, or feature, was for predicting. The more positive the value, the more it predicts defaults. Look at the value for loan_int_rate.
"""




"""
1. Predicting the probability of default
00:00 - 00:10
So far, we've trained a logistic regression on our credit data, and looked some attributes of the model. Now, let's discuss the structure of the model and how to create predictions.

2. Logistic regression coefficients
00:10 - 00:42
In the previous exercise, we saw the following intercept and coefficients for our model. These coefficients the importance of each column. These values are part of the logistic regression formula that calculates the probability of default which we see here. Each coefficient is multiplied by the values in the column, and then added together along with the intercept. Then, 1 is divided by the sum of 1 and e to the negative power of our intercept coefficient sums. The result is the probability of default.

3. Interpreting coefficients
00:42 - 01:05
Consider employment length as an example. I've already calculated the intercept and coefficient for a logistic regression using this one column. What this coefficient tells us is the log odds for non-default. This means that for every 1 year increase in employment length, the person is less likely to default by a factor of the coefficient.

4. Interpreting coefficients
01:05 - 01:19
Let's say we have 3 values for employment length, and we want to know how this affects our probability of default by looking at the coefficients. What we see here is that the higher a person's employment length is, the less likely they are to default.

5. Using non-numeric columns
01:19 - 01:47
Since we're talking about numbers, it's worth mentioning that so far we have only used numeric columns to train out models. Our data also contains non-numeric columns like loan intent, which uses words to describe how the person plans to use the money we lend them. In Python, unlike R, machine learning models do not know how to use these non-numeric values. So, we have to perform an operation called one-hot encoding before we can use them.

6. One-hot encoding
01:47 - 02:07
One-hot encoding sounds complicated, but it's really simple. The main idea is to represent a string with a numeric value. Here is how it works. Let's think about the loan intent column where each loan has it's own intent value as a string. This sample has education, medical, and venture.

7. One-hot encoding
02:07 - 02:34
With one-hot encoding, we get a new set of columns where each value from loan intent is now it's own column. Each new column is created by separating out the loans with each intent value and making the new column's value a 0 or 1. For example, if the loan intent was education, it is now represented with a 1 in the loan intent education column. This way, there is one hot value.

8. Get dummies
02:34 - 02:57
To one-hot encode our string columns, we use the get dummies function within pandas. First, we separate the numeric and non-numeric columns from the data into two sets. Then we use the get dummies function to one-hot encode only the non-numeric columns. We union the two sets and the result is a full data set that's ready for machine learning!

9. Predicting the future, probably
02:57 - 03:23
Once our model is trained, we use the predict proba method on test data to make predictions. This creates a set of probabilities for non-default and default. Notice the output is a series of numbers between 0 and 1. We have two for each loan. The first number is the probability of non-default, and the second number is the probability of default.

"""




"""
Changing coefficients
With this understanding of the coefficients of a LogisticRegression() model, have a closer look at them to see how they change depending on what columns are used for training. Will the column coefficients change from model to model?

You should .fit() two different LogisticRegression() models on different groups of columns to check. You should also consider what the potential impact on the probability of default might be.

The data set cr_loan_clean has already been loaded into the workspace along with the training sets X1_train, X2_train, and y_train.
"""

"""
Check the first five rows of both X training sets.
Train a logistic regression model, called clf_logistic1, with the X1 training set.
Train a logistic regression model, called clf_logistic2, with the X2 training set.
Print the coefficients for both logistic regression models.
"""

# Print the first five rows of each training set
print(X1_train.head())
print(X2_train.head())

# Create and train a model on the first training data
clf_logistic1 = LogisticRegression(solver='lbfgs').fit(X1_train, np.ravel(y_train))

# Create and train a model on the second training data
clf_logistic2 = LogisticRegression(solver='lbfgs').fit(X2_train, np.ravel(y_train))

# Print the coefficients of each model
print(clf_logistic1.coef_)
print(clf_logistic2.coef_)

"""
Interesting! Notice that the coefficient for the person_income changed when we changed the data from X1 to X2. This is a reason to keep most of the data like we did in chapter 1, because the models will learn differently depending on what data they're given!
"""


"""
One-hot encoding credit data
It's time to prepare the non-numeric columns so they can be added to your LogisticRegression() model.

Once the new columns have been created using one-hot encoding, you can concatenate them with the numeric columns to create a new data frame which will be used throughout the rest of the course for predicting probability of default.

Remember to only one-hot encode the non-numeric columns. Doing this to the numeric columns would create an incredibly wide data set!
"""

"""
Create a data set for all the numeric columns called cred_num and one for the non-numeric columns called cred_str.
Use one-hot encoding on cred_str to create a new data set called cred_str_onehot.
Union cred_num with the new one-hot encoded data and store the results as cr_loan_prep.
Print the columns of the new data set.
"""

# Create two data sets for numeric and non-numeric data
cred_num = cr_loan_clean.select_dtypes(exclude=['object'])
cred_str = cr_loan_clean.select_dtypes(include=['object'])

# One-hot encode the non-numeric columns
cred_str_onehot = pd.get_dummies(cred_str)

# Union the one-hot encoded columns to the numeric ones
cr_loan_prep = pd.concat([cred_num, cred_str_onehot], axis=1)

# Print the columns in the new data set
print(cr_loan_prep.columns)

"""
Look at all those columns! If you've ever seen a credit scorecard, the column_name_value format should look familiar. If you haven't seen one, look up some pictures during your next break!
"""


"""
Predicting probability of default
All of the data processing is complete and it's time to begin creating predictions for probability of default. You want to train a LogisticRegression() model on the data, and examine how it predicts the probability of default.

So that you can better grasp what the model produces with predict_proba, you should look at an example record alongside the predicted probability of default. How do the first five predictions look against the actual values of loan_status?

The data set cr_loan_prep along with X_train, X_test, y_train, and y_test have already been loaded in the workspace.
"""

"""
Train a logistic regression model on the training data and store it as clf_logistic.
Use predict_proba() on the test data to create the predictions and store them in preds.
Create two data frames, preds_df and true_df, to store the first five predictions and true loan_status values.
Print the true_df and preds_df as one set using .concat()
"""

# Train the logistic regression model on the training data
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Create predictions of probability for loan status using test data
preds = clf_logistic.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))

"""
Neat! We have some predictions now, but they don't look very accurate do they? It looks like most of the rows with loan_status at 1 have a low probability of default. How good are the rest of the predictions? Next, let's see if we can determine how accurate the entire model is.

"""


"""
1. Credit model performance
00:00 - 00:07
We saw predictions for probability of default against true values for loan status, but how do we analyze the performance of our model?

2. Model accuracy scoring
00:07 - 00:36
The easiest way to analyze performance is with accuracy. Accuracy is the number of correct predictions divided by the total number of predictions. One way to check this is to use the score method within scikit-learn on the logistic regression. This is used on the trained model and returns the average accuracy for the test set. Using the score method will display this accuracy as a percentage. In this example, it tells us that 81 percent of the loans were predicted correctly.

3. ROC curve charts
00:36 - 01:08
R-O-C charts are a great way to visualize the performance of our model. They plot the true positive rate, the percentage of correctly predicted defaults, against the false positive rate, the percentage of incorrectly predicted defaults. Using the roc_curve function in scikit-learn, we create these two values and the thresholds all at once. From there, we use a normal line plot to see the results. The dotted blue line represents a random prediction and the orange line represents our model's predictions.

4. Analyzing ROC charts
01:08 - 01:33
R-O-C charts are interpreted by looking at how far away the model's curve gets from the dotted blue line shown here, which represents the random prediction. This movement away from the line is called lift. The more lift we have, the larger the area under the curve gets. The A-U-C is the calculated area between the curve and the random prediction. This is a direct indicator of how well our model makes predictions.

5. Default thresholds
01:33 - 01:58
To analyze performance further, we need to decide what probability range is a default, and what is a non-default. Let's say that we decide any probability over 0.5 is a default, and anything below that is a non-default. What this means is that we will assign a new loan_status to these loans based on their probability of default and the threshold. Once we have this, we can further check the model's performance.

6. Setting the threshold
01:58 - 02:34
Once the threshold is defined, we need to relabel our loans based on that threshold. For that, we will first need to create a variable to store the predicted probabilities. Then we can create a data frame from the second column which contains the probabilities of default. Then we apply a quick function to assign a value of 1 if the probability of default is above our threshold of 0.5. The lambda is there just to tell Python that we want to use a one-time function without defining it. The result of this is a data frame with new values for loan status based on our threshold.

7. Credit classification reports
02:34 - 03:06
Another really useful function for evaluating our models is the classification report function within scikit-learn. This will show us several different evaluation metrics all at once! We use this function to evaluate our model using our true values for loan status stored in the y_test set, and our predicted loan status values from our logistic regression and the threshold we set. There are 2 really useful metrics in this table, and they are the precision and recall. For now, let's focus on recall.

8. Selecting classification metrics
03:06 - 03:30
Sometimes after generating the report, you want to select or store specific values from within the report. To do this, you can use the precision recall fscore support function within sci-kit learn. With this function, we can get the recall for defaults from by subsetting the report the way we would any array. Here we select the second value from the second set.
"""



"""
Default classification reporting
It's time to take a closer look at the evaluation of the model. Here is where setting the threshold for probability of default will help you analyze the model's performance through classification reporting.

Creating a data frame of the probabilities makes them easier to work with, because you can use all the power of pandas. Apply the threshold to the data and check the value counts for both classes of loan_status to see how many predictions of each are being created. This will help with insight into the scores from the classification report.

The cr_loan_prep data set, trained logistic regression clf_logistic, true loan status values y_test, and predicted probabilities, preds are loaded in the workspace.
"""

"""
Create a data frame of just the probabilities of default from preds called preds_df.
Reassign loan_status values based on a threshold of 0.50 for probability of default in preds_df.
Print the value counts of the number of rows for each loan_status.
Print the classification report using y_test and preds_df.
"""

# Create a dataframe for the probabilities of default
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])

# Reassign loan status based on the threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.50 else 0)

# Print the row counts for each loan status
print(preds_df['loan_status'].value_counts())

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))



"""
Selecting report metrics
The classification_report() has many different metrics within it, but you may not always want to print out the full report. Sometimes you just want specific values to compare models or use for other purposes.

There is a function within scikit-learn that pulls out the values for you. That function is precision_recall_fscore_support() and it takes in the same parameters as classification_report.

It is imported and used like this:

# Import function
from sklearn.metrics import precision_recall_fscore_support
# Select all non-averaged values from the report
precision_recall_fscore_support(y_true,predicted_values)

"""
"""
Print the classification report for y_test and predicted loan status.
"""

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))


# Print all the non-average values from the report
print(precision_recall_fscore_support(y_test, preds_df['loan_status']))


# Print the first two numbers from the report
print(precision_recall_fscore_support(y_test, preds_df['loan_status'])[:2])

"""
Great! Now we know how to pull out specific values from the report to either store later for comparison, or use to check against portfolio performance. Remember the impact of recall for defaults? This way, you can store that value for later calculations.
"""


"""
Visually scoring credit models
Now, you want to visualize the performance of the model. In ROC charts, the X and Y axes are two metrics you've already looked at: the false positive rate (fall-out), and the true positive rate (sensitivity).

You can create a ROC chart of it's performance with the following code:

fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)
plt.plot(fallout, sensitivity)

To calculate the AUC score, you use roc_auc_score().

The credit data cr_loan_prep along with the data sets X_test and y_test have all been loaded into the workspace. A trained LogisticRegression() model named clf_logistic has also been loaded into the workspace.

Create a set of predictions for probability of default and store them in preds.
Print the accuracy score the model on the X and y test sets.
Use roc_curve() on the test data and probabilities of default to create fallout and sensitivity Then, create a ROC curve plot with fallout on the x-axis.
Compute the AUC of the model using test data and probabilities of default and store it in auc.
"""


# Create predictions and store them in a variable
preds = clf_logistic.predict_proba(X_test)

# Print the accuracy score the model
print(clf_logistic.score(X_test, y_test))

# Plot the ROC curve of the probabilities of default
prob_default = preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()

# Compute the AUC and store it in a variable
auc = roc_auc_score(y_test, prob_default)


"""
I wasn't worried about your .score() on this exercise! So the accuracy for this model is about 80% and the AUC score is 76%. Notice that what the ROC chart shows us is the tradeoff between all values of our false positive rate (fallout) and true positive rate (sensitivity).
"""


"""
1. Model discrimination and impact
00:00 - 00:07
We've looked at some ways to evaluate our logistic regression. Let's talk more about thresholds and their impact on portfolio performance.

2. Confusion matrices
00:07 - 00:34
Another way to analyze our model's performance is with the confusion matrix. These will show us all our correct and incorrect predictions for loan status. The confusion matrix has four sections: true positives, false positives, false negatives, and true negatives. We've looked at recall for defaults within classification reports. That formula and where it resides in the confusion matrix are shown here.

3. Default recall for loan status
00:34 - 00:57
The definition of default recall, also called sensitivity, is the proportion of actual positives correctly predicted. Before, we retrieved this value from the classification report without understanding how it's calculated. Recall is found by taking the number of true defaults and dividing it by the sum of true defaults and defaults predicted as non-default.

4. Recall portfolio impact
00:57 - 01:12
Let's look at the recall for defaults highlighted in red in a classification report. This is an example of a report from an under-performing Logistic Regression model. Here, the proportion of true defaults predicted by our model was only 4 percent.

5. Recall portfolio impact
01:12 - 01:47
Imagine that we have 50 thousand loans in our portfolio, and they each have a total loan amount of 50 dollars. As seen in the classification report, this model has a default recall of 4 percent. So, that means we correctly predicted 4 percent of defaults, and incorrectly predicted 96 percent of defaults. If all of our true default loans defaulted right now, our estimated loss from the portfolio would be 2.4 million dollars! This loss would be something we didn't plan for, and would be unexpected.

6. Recall, precision, and accuracy
01:47 - 02:27
When it comes to metrics like recall, precision, and accuracy, it can be challenging to find an optimum number for all three as a target. Have a look at this example graph of a logistic regression model on the credit data. The blue line, which is default recall, starts out really high. This is because if we predict all loans to be a default, we definitely predict all of our defaults correctly! You can also see that when default recall is high, more often than not non-default recall is low. Initially, we have to make a determination about what scores for each are good enough in order to set a baseline for performance.

"""


"""
Thresholds and confusion matrices
You've looked at setting thresholds for defaults, but how does this impact overall performance? To do this, you can start by looking at the effects with confusion matrices.

Recall the confusion matrix as shown here:



Set different values for the threshold on probability of default, and use a confusion matrix to see how the changing values affect the model's performance.

The data frame of predictions, preds_df, as well as the model clf_logistic have been loaded in the workspace.
"""
"""
Reassign values of loan_status using a threshold of 0.5 for probability of default within preds_df.
Print the confusion matrix of the y_test data and the new loan status values.
"""

# Set the threshold for defaults to 0.5
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test,preds_df['loan_status']))

"""
[[9023  175]
 [2152  434]]
"""

# Set the threshold for defaults to 0.4
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test,preds_df['loan_status']))
"""
[[8476  722]
 [1386 1200]]
"""

"""
Based on the confusion matrices you just created, calculate the default recall for each. Using these values, answer the following: which threshold gives us the highest value for default recall?
"""
#  Answer 0.4 , Correct! The value for default recall at this threshold is actually pretty high! You can check out the non-default recalls as well to see how the threshold affected those values.


"""
How thresholds affect performance
Setting the threshold to 0.4 shows promising results for model evaluation. Now you can assess the financial impact using the default recall which is selected from the classification reporting using the function precision_recall_fscore_support().

For this, you will estimate the amount of unexpected loss using the default recall to find what proportion of defaults you did not catch with the new threshold. This will be a dollar amount which tells you how much in losses you would have if all the unfound defaults were to default all at once.

The average loan value, avg_loan_amnt has been calculated and made available in the workspace along with preds_df and y_test.
"""

"""
Reassign the loan_status values using the threshold 0.4.
Store the number of defaults in preds_df by selecting the second value from the value counts and store it as num_defaults.
Get the default recall rate from the classification matrix and store it as default_recall
Estimate the unexpected loss from the new default recall by multiplying 1 - default_recall by the average loan amount and number of default loans.
"""
# Reassign the values of loan status based on the new threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

# Store the number of loan defaults from the prediction data
num_defaults = preds_df['loan_status'].value_counts()[1]

# Store the default recall from the classification report
default_recall = precision_recall_fscore_support(y_test,preds_df['loan_status'])[1][1]

# Calculate the estimated impact of the new default recall rate
print(avg_loan_amnt * num_defaults * (1 - default_recall))


"""
Threshold selection
You know there is a trade off between metrics like default recall, non-default recall, and model accuracy. One easy way to approximate a good starting threshold value is to look at a plot of all three using matplotlib. With this graph, you can see how each of these metrics look as you change the threshold values and find the point at which the performance of all three is good enough to use for the credit data.

The threshold values thresh, default recall values def_recalls, the non-default recall values nondef_recalls and the accuracy scores accs have been loaded into the workspace. To make the plot easier to read, the array ticks for x-axis tick marks has been loaded as well.
"""
"""
Plot the graph of thresh for the x-axis then def_recalls, non-default recall values, and accuracy scores on each y-axis.
"""

plt.plot(thresh,def_recalls)
plt.plot(thresh,nondef_recalls)
plt.plot(thresh,accs)
plt.xlabel("Probability Threshold")
plt.xticks(ticks)
plt.legend(["Default Recall","Non-default Recall","Model Accuracy"])
plt.show()

"""
Have a closer look at this plot. In fact, expand the window to get a really good look. Think about the threshold values from thresh and how they affect each of these three metrics. Approximately what starting threshold value would maximize these scores evenly?
"""

#  0.275  , Yes! This is the easiest pattern to see on this graph, because it's the point where all three lines converge. This threshold would make a great starting point, but declaring all loans about 0.275 to be a default is probably not practical.







"""
Chapter 3

1. Gradient boosted trees with XGBoost
00:00 - 00:10
We've used many different ways to experiment with a logistic regression for probability of default. Now, let's have a look at gradient boosted decision trees using XGBoost.

2. Decision trees
00:10 - 00:45
So what is a decision tree? They are machine learning models which use decisions as steps in a process to eventually identify our loan status. While they produce predictions similar to logistic regressions, they are not structured the same way. Here is an example of a simple decision tree. The first box, or node, has decided to split the data into two groups. Those with an employment length above 10, and those below. Then it uses loan intent medical the same way. The results of these splits are yes and no decisions that eventually lead to a predicted loan status of default or non-default.

3. Decision trees for loan status
00:45 - 01:08
Let's have a look at a simple example of a decision tree on the loan data when what we are predicting is still defaults. Here, we have a red dot for each default, and a green dot for each non-default. The red shaded area is what our model predicted as default. While it predicted all of the defaults correctly, it predicted two non-defaults as default.

4. Decision tree impact
01:08 - 01:28
What are the consequences of this? Let's say both of these loans were worth 1500 and 1200 at the time we predicted their status. Then, maybe we decide to sell off all debt we think is likely to default for 250 per loan. As a result of the model, our loss is 2200 dollars for just two loans!

5. A forest of trees
01:28 - 02:06
XGBoost doesn't use just one decision tree though, but a large number of them in what's known as an ensemble through a method called gradient boosting. Each tree in the ensemble is like the one we just saw, and is a weak predictor. Have a look at this example. The first two boxes on the left represent two different individual models. Each of them predicts the defaults, but they also predict some non-defaults as defaults. However, when we use gradient boosting with XGBoost, we get the box on the right which combines the two weak models. In this example, the boosted model predicts all of the loans correctly.

6. Creating and training trees
02:06 - 02:25
The trees we will use are available within the xgboost package, and they train similar to logistic regression models. Here, we can see that the gradient boosted tree is created using the XGBClassifier function. Next, fit is called on the model the same as before and with the same training data.

7. Default predictions with XGBoost
02:25 - 02:48
These models predict the same way as the logistic regression do. We can use predict_proba to predict probabilities of default. The predict method gives us a value of 0 or 1 for loan status. The predict_proba method returns an array of probabilities for default and non-default. The predict method returns an array of the loan status.

8. Hyperparameters of gradient boosted trees
02:48 - 03:20
The models have parameters that are like settings that affect how a model learns. These settings are called hyperparameters. Hyperparameters cannot be learned from data; they have to be set by us. Let's look at a few of these hyperparameters. The learning rate tells the model how quickly it should learn in each step of the ensemble. The smaller the value, the more conservative it is at each step. The max depth tells the model how deep each tree can go. Keeping this value low ensures the model is not too complex.
"""

