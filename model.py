import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
import csv
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, make_scorer, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
# import the needed libraries first
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#---------------------------------------------------------------------------------------------------------------------------------

path = ''

train_data = pd.read_csv(path + 'train.csv',index_col='id')
test_data = pd.read_csv(path + 'test.csv',index_col='id')
event_type = pd.read_csv(path + 'event_type.csv')
log_feature = pd.read_csv(path + 'log_feature.csv')
resource_type = pd.read_csv(path + 'resource_type.csv')
severity_type = pd.read_csv(path + 'severity_type.csv')

train_data_copy = train_data.copy()
test_data_copy = test_data.copy()
event_type_copy = event_type.copy()
log_feature_copy = log_feature.copy()
resource_type_copy = resource_type.copy()
severity_type_copy = severity_type.copy()

train_data = train_data.reset_index()
test_data = test_data.reset_index()
# ---------------------------------------------------------------------------------------------------------------------
# applying lambda f'n to split all the location_id in both train and test data
# ---------------------------------------------------------------------------------
train_data['location_id'] = train_data.location.apply(lambda x: int(x.split('location ')[1]))
test_data['location_id'] = test_data.location.apply(lambda x: int(x.split('location ')[1]))

print('train', train_data.shape, 'test', test_data.shape)
# ---------------------------------------------------------------------------------------------------------------------
# applying lambda f'n to split all the event_type and get dummies for event_type
event_type['event_type'] = event_type['event_type'].map(lambda x: int(x.split(' ')[1]))
#event_type = pd.get_dummies(event_type, columns=['event_type'],dtype=np.int64) # get dummy variables for event type
event_type = event_type.groupby(event_type.id,as_index=False).sum() # compact the rows with the same id
#----------------------------------------------------------------------------------------------------------------------
# 2.1. Left join training/test set with event_type set on 'id'
X = pd.merge(left=train_data, right=event_type, how='left', left_on='id', right_on='id')
Y = pd.merge(left=test_data, right=event_type, how='left', left_on='id', right_on='id')
#-------------------------------------------------------------------------------------------------------------------------
# applying lambda f'n to split all the resource_type and get dummies for resource_type
resource_type['resource_type'] = resource_type['resource_type'].map(lambda x: int(x.split(' ')[1]))
#resource_type = pd.get_dummies(resource_type, columns=['resource_type'],dtype=np.int64)
resource_type = resource_type.groupby(resource_type.id,as_index=False).sum()
#------------------------------------------------------------------------------------------------------------------------
# 2.2. Left join training/test set with resource_type set on 'id'
X = pd.merge(left=X, right=resource_type, how='left', left_on='id', right_on='id')
Y = pd.merge(left=Y, right=resource_type, how='left', left_on='id', right_on='id')
#------------------------------------------------------------------------------------------------------------------------
# applying lambda f'n to split all the severity_type and get dummies for severity_type
severity_type['severity_type'] = severity_type['severity_type'].map(lambda x: int(x.split(' ')[1]))
#severity_type = pd.get_dummies(severity_type, columns=['severity_type'],dtype=np.int64)
severity_type = severity_type.groupby(severity_type.id,as_index=False).sum()
#----------------------------------------------------------------------------------------------------------------------
# 2.3. Left join training/test set <-> severity_type set
X = pd.merge(left=X, right=severity_type, how='left', left_on='id', right_on='id')
Y = pd.merge(left=Y, right=severity_type, how='left', left_on='id', right_on='id')
#----------------------------------------------------------------------------------------------------------------------
# applying lambda f'n to split all the log_feature and get dummies for log_feature
log_feature['log_feature'] = log_feature['log_feature'].map(lambda x: int(x.split(' ')[1]))
#log_feature = pd.get_dummies(log_feature, columns=['log_feature'],dtype=np.int64)
log_feature = log_feature.groupby(log_feature.id,as_index=False).sum()
#---------------------------------------------------------------------------------------------------------------------
# 2.4. Left join training/test set <-> log_feature set
X = pd.merge(left=X, right=log_feature, how='left', left_on='id', right_on='id')
Y = pd.merge(left=Y, right=log_feature, how='left', left_on='id', right_on='id')
#--------------------------------------------------------------------------------------------------------------------

# using groupby with 'id' for 'log_features' features and aggregating them using count,min,max,std,sum
log_vol = log_feature_copy.groupby('id')['volume'].agg(['count','min', 'mean', 'max', 'std', 'sum']).fillna(0).add_prefix('log_volume')
log_vol = log_vol.reset_index()
#-----------------------------------------------------------------------------------------------------------------------
# 2.5. Left join both X and log_vol
X = pd.merge(left=X, right=log_vol, how='left', left_on='id', right_on='id')
Y = pd.merge(left=Y, right=log_vol, how='left', left_on='id', right_on='id')
#-----------------------------------------------------------------------------------------------------------------------
#newly added feature - event_id
event_type_copy['event_id'] = event_type_copy.event_type.apply(lambda x: int(x.split('event_type ')[1]))
#-----------------------------------------------------------------------------------------------------------------------
#making a copy of X & Y
X1 = X.copy()
Y1 = Y.copy()
#-----------------------------------------------------------------------------------------------------------------------
# 2.1. Left join lg_feature copy <-> event_type set
logev_X = pd.merge(left=log_feature_copy, right=event_type_copy, how='left', left_on='id', right_on='id')
logev_Y = pd.merge(left=log_feature_copy, right=event_type_copy, how='left', left_on='id', right_on='id')

# merge log features
logev_volx = logev_X.groupby('id')['volume'].agg(['count','min', 'mean', 'max', 'std', 'sum']).fillna(0).add_prefix('event_volume')
logev_voly = logev_Y.groupby('id')['volume'].agg(['count','min', 'mean', 'max', 'std', 'sum']).fillna(0).add_prefix('event_volume')

logev_volx = logev_volx.reset_index()
logev_voly = logev_voly.reset_index()

# 2.1. Left join training/test set <-> event_type set
X1 = pd.merge(left=X1, right=logev_volx, how='inner', left_on='id', right_on='id')
Y1 = pd.merge(left=Y1, right=logev_voly, how='inner', left_on='id', right_on='id')

#newly added
severity_type_copy['sev_id'] = severity_type_copy.severity_type.apply(lambda x: int(x.split('severity_type ')[1]))

# 2.1. Left join training/test set <-> event_type set
sev_X = pd.merge(left=log_feature_copy, right=severity_type_copy, how='left', left_on='id', right_on='id')
sev_Y = pd.merge(left=log_feature_copy, right=severity_type_copy, how='left', left_on='id', right_on='id')

# merge log features
sev_volx = sev_X.groupby('id')['volume'].agg(['count','min', 'mean', 'max', 'std', 'sum']).fillna(0).add_prefix('sev_volume')
sev_voly = sev_Y.groupby('id')['volume'].agg(['count','min', 'mean', 'max', 'std', 'sum']).fillna(0).add_prefix('sev_volume')

sev_volx = sev_volx.reset_index()
sev_voly = sev_voly.reset_index()

# 2.1. Left join training/test set <-> event_type set
X1 = pd.merge(left=X1, right=sev_volx, how='inner', left_on='id', right_on='id')
Y1 = pd.merge(left=Y1, right=sev_voly, how='inner', left_on='id', right_on='id')

# Python log transform
X1.insert(len(X1.columns), 'log_volume',np.log(X1['volume']))
Y1.insert(len(Y1.columns), 'log_volume',np.log(Y1['volume']))

X1["avgvol_per_loc"]= X1.groupby(['location_id'])["log_volume"].transform('mean')
X1["maxvol_per_loc"]= X1.groupby(['location_id'])["log_volume"].transform('max')
X1["minvol_per_loc"]= X1.groupby(['location_id'])["log_volume"].transform('min')
X1["medianvol_per_loc"]= X1.groupby(['location_id'])["log_volume"].transform('median')
X1["stdvol_per_loc"] = X1.groupby(['location_id'])["log_volume"].transform('std')

Y1["avgvol_per_loc"]= Y1.groupby(['location_id'])["log_volume"].transform('mean')
Y1["maxvol_per_loc"]= Y1.groupby(['location_id'])["log_volume"].transform('max')
Y1["minvol_per_loc"]= Y1.groupby(['location_id'])["log_volume"].transform('min')
Y1["medianvol_per_loc"]= Y1.groupby(['location_id'])["log_volume"].transform('median')
Y1["stdvol_per_loc"] = Y1.groupby(['location_id'])["log_volume"].transform('std')

# order ~ time
# ---------------------------------------------------------------------------------
severity_type_order = severity_type_copy[['id']].drop_duplicates()
severity_type_order['order'] = 1. * np.arange(len(severity_type_order)) / len(severity_type_order)

X1 = pd.merge(left=X1, right=severity_type_order, how='inner', on='id')
Y1 = pd.merge(left=Y1, right=severity_type_order, how='inner', on='id')

# order ~ time
# ---------------------------------------------------------------------------------
event_type_order = event_type_copy[['id']].drop_duplicates()
event_type_order['event_order'] = 1. * np.arange(len(event_type_order)) / len(event_type_order)

X1 = pd.merge(left=X1, right=event_type_order, how='inner', on='id')
Y1 = pd.merge(left=Y1, right=event_type_order, how='inner', on='id')

# order ~ time
# ---------------------------------------------------------------------------------
log_feature_order = log_feature_copy[['id']].drop_duplicates()
log_feature_order['feature_order'] = 1. * np.arange(len(log_feature_order)) / len(log_feature_order)

X1 = pd.merge(left=X1, right=log_feature_order, how='inner', on='id')
Y1 = pd.merge(left=Y1, right=log_feature_order, how='inner', on='id')

# rank location features by ascneding and descending orders
# ---------------------------------------------------------------------------------
X1['location_rank_asc'] = X1.groupby('location_id')[['order']].rank()
X1['location_rank_desc'] = X1.groupby('location_id')[['order']].rank(ascending=False)

# ---------------------------------------------------------------------------------
Y1['location_rank_asc'] = Y1.groupby('location_id')[['order']].rank()
Y1['location_rank_desc'] = Y1.groupby('location_id')[['order']].rank(ascending=False)

location_count_tr = train_data.groupby('location_id').count()[['id']]
location_count_tr.columns = ['location_count']

location_count_te = test_data.groupby('location_id').count()[['id']]
location_count_te.columns = ['location_count']

# 2.1. Left join training/test set <-> event_type set
X1 = pd.merge(left=X1, right=location_count_tr, how='left', left_on='location_id', right_on='location_id')
Y1 = pd.merge(left=Y1, right=location_count_te, how='left', left_on='location_id', right_on='location_id')

X1['loc_rank_rel'] = 1. * X1['location_rank_asc'] / X1['location_count']
X1['loc_rank_rel'] = np.round(X1['loc_rank_rel'], 2)

Y1['loc_rank_rel'] = 1. * Y1['location_rank_asc'] / Y1['location_count']
Y1['loc_rank_rel'] = np.round(Y1['loc_rank_rel'], 2)

X_df = X1.copy()
Y_df = Y1.copy()

index = Y1['id']
Y1['id'] = index

X = X1.set_index('id')
Y = Y1.set_index('id')

y = X_df['fault_severity']

X = X.drop(['fault_severity','location'], axis=1)
Y = Y.drop(['location'], axis=1)

print(X.shape,Y.shape)

# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]
X_train,X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# create a transformed for the numerical values
numeric_transformer_std = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler',StandardScaler())])

from sklearn.compose import ColumnTransformer

preprocessor_sca_std = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_std, numeric_features),
        #('cat', categorical_transformer, categorical_features)
    ])

#We are then ready to transform!
X_train_transformed_sca1 = preprocessor_sca_std.fit_transform(X_train)
X_test_transformed_sca1 = preprocessor_sca_std.transform(X_test)
X_train_transformed_sca_copy = X_train_transformed_sca1.copy()
test_transformed_sca1 = preprocessor_sca_std.transform(Y)

X_train_sca = pd.DataFrame(X_train_transformed_sca1)
X_test_sca = pd.DataFrame(X_test_transformed_sca1)
test_sca = pd.DataFrame(test_transformed_sca1)

lgbm = lgb.LGBMClassifier()
lgbm.fit(X_train_sca,y_train)

loss2tr2 = log_loss(y_train.values, lgbm.predict_proba(X_train_sca))
print("LGBM: train log loss {:.4f}".format(loss2tr2))
print('---------------------------------------------------------------')
loss2te2 = log_loss(y_test.values, lgbm.predict_proba(X_test_sca))
print("LGBM: test log loss {:.4f}".format(loss2te2))

### Create a Pickle file using serialization
import pickle
pickle_out = open("lgbm_classifier.pkl","wb")
pickle.dump(lgbm, pickle_out)
pickle_out.close()






