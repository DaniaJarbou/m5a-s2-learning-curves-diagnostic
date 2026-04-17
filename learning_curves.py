
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split , learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


def load_data(filepath ) :
    #load data and remove unnecessary columns
    df  = pd.read_csv (filepath).copy().drop(columns=['customer_id'])
    print(f"data frame shape {df.shape}")
    return df

def prepare_data (df ,target_col ,cv=5 , random_state = 42,test_size =0.2  ):
    
    #split data into features (x)and target (y)
    x=df.drop(columns=[target_col])
    y=df[target_col]
    
    #training and testing sets
    x_train , x_test , y_train ,y_test = train_test_split(x,y,random_state=random_state,test_size=test_size)
    
    return x_train,x_test,y_train,y_test
    
def build_preprocessor ():
    #Create a preprocessor to scale numbers and encode text
   preprocessor = ColumnTransformer(
       transformers=[
           ('numeric', StandardScaler(), NUMERIC_FEATURES),
           ('categorical', OneHotEncoder(drop='first',handle_unknown='ignore'),CATEGORICAL_FEATURES)
           ]
   )
   return preprocessor


def define_models () :
    preprocessor = build_preprocessor()
     
    #create  pipelines for different models to compare  
    models = {
        'LogisticRegression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(
                C=1.0 , 
                random_state=42,
                max_iter=1000,
                class_weight= 'balanced'
            ))
        ]),
        'RodgeClassifier': Pipeline([
         ('preprocessor', preprocessor) ,
         ('model', RidgeClassifier(
             alpha=0.1,
             random_state=42,
             max_iter=1000,
             class_weight='balanced'
         ))  
        ])
    }
    return models

def diagnostic_learning_curves(models, X, y):
    #Generate and plot learning curves for each model
    for name , pipeline in models.items():
        print(f"Generating learning curve for {name}")
        
        #Calcolate training and validation scores
        train_sizes , train_scores,test_scores = learning_curve(
            estimator=pipeline,
            X=X,
            y=y, 
            train_sizes= np.linspace(0.1,1.0,5),
            cv = StratifiedKFold(5),
            scoring='f1',
            n_jobs=-1
        )
        #calcolate mean and std for the plott
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores,axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores,axis=1)
        
        #Plot the results with confidence areas (shadows)
        plt.figure(figsize=(8,5)) 
         
       #drow main lines 
        plt.plot(train_sizes, train_mean, 'o-', color="r", label='Training Score')
        plt.plot(train_sizes, test_mean, 'o-', color="g", label='Validation Score')
    
        #draw shadow areas (std)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
       
        plt.title(f"Learning Curve: {name}")
        plt.xlabel("Training set size")
        plt.ylabel("F1 Score")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(f"learning_curve_{name}.png")
        plt.show()

if __name__ == "__main__":
    #Load data 
    data_path = 'data/telecom_churn.csv'
    df = load_data(data_path)
    
   # prepare_data , use 'churned' as target col.
    X_train, X_test, y_train, y_test =prepare_data(df , "churned")
   
   #Define  models 
    models_to_test = define_models()
   
    diagnostic_learning_curves(models_to_test,X_train,y_train)