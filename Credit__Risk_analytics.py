
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




