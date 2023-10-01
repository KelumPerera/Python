
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
print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],values=cr_loan['person_emp_length'], aggfunc='mean'))

