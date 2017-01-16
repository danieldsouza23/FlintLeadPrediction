%matplotlib inline

# Importing libraries that we'll use
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

# Assigning variables to our training and test data
train_df = pd.read_csv('./data/flint_train.csv')
test_df = pd.read_csv('./data/flint_test.csv')

# Columns that don't have numerical values. Solution? 1-hot encoding!
dummy_columns = ['Owner_Type', 'Residential_Building_Style', 'USPS_Vacancy',
                 'Building_Storeys', 'Rental', 'Use_Type', 'Prop_Class', 'Zoning', 'Future_Landuse', 'DRAFT_Zone',
                 'Housing_Condition_2012', 'Housing_Condition_2014','Hydrant_Type', 'Ward', 'PRECINCT', 'CENTRACT',
                 'Commercial_Condition_2013','CENBLOCK', 'SL_Type', 'SL_Type2', 'SL_Lead', 'Homestead']

# Columns that don't affect our prediction in any way

drop_columns = ['sample_id', 'google_add', 'parcel_id', 'Date_Submitted']

# Combining train and test data for common data manipulation
combined_df = train_df.append(test_df)

combined_df = combined_df.drop(drop_columns, axis=1)
combined_df = pd.get_dummies(combined_df, columns = dummy_columns)

# Separating the data after 1 hot encoding and dropping the dummy columns
train_df = combined_df[:len(train_df)]
test_df = combined_df[len(train_df):]

test_df = test_df.drop('Lead_(ppb)', axis=1)


Ydata_r = train_df['Lead_(ppb)']
Ydata_c = train_df['Lead_(ppb)'] > 15
Xdata = train_df.drop('Lead_(ppb)', axis=1)

# We are dealing with a regression problem, so we'll split on Ydata_r

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata_c)


#GridSearchCV(estimator,            A model like random forest, linear regression, etc.
#            param_grid,            A dictionary of parameters to try
#            n_jobs=1,              Parallelize?
#            cv=None,               How many cross validation folds?
#            verbose=0,             How much do you want the function to talk to you?
#            error_score='raise')   What should the function do if a model breaks?

rf = RandomForestClassifier()

# Now, create a dictionary of parameters to try.
param_dict={'n_estimators':[32, 64, 128, 256],
            'criterion':['gini', 'entropy'],
            'min_samples_split':[2,4,6]}

# Create the grid search classifier with our desired parameters.
grid_search_clf = GridSearchCV(rf, param_dict, n_jobs=-1, cv=3, verbose=3)

# Now we can fit and predict like any other model.
grid_search_clf.fit(Xtrain, Ytrain)

yhat_g = grid_search_clf.predict_proba(Xtest)

# Which parameters worked the best?
print(grid_search_clf.best_params_)

# Plotting the results
fig = plt.figure()
fig.set_size_inches(8,8)

fpr, tpr, _ = roc_curve(Ytest, yhat_g[:,1])
plt.plot(fpr, tpr, label= 'Grid Search (area = %0.5f)' % roc(Ytest, yhat_g[:,1]))

fpr, tpr, _ = roc_curve(Ytest, yhat2[:,1])
plt.plot(fpr, tpr, label= 'Random Forest 2(area = %0.5f)' % roc(Ytest, yhat2[:,1]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Service Line Classifiers')
plt.legend(loc="lower right")

plt.show()