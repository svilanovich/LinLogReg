import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import numpy as np


class LinReg:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def fit(self, X, y, n, reg_type=None, reg_coef1=0.001, reg_coef2=0.001):
        
        X = np.array(X)
        y = np.array(y)

        
        self.coef_ = np.random.normal(size=X.shape[1])
        self.intercept_ = np.random.normal(1)
        
        self.n = n
        self.reg_type = reg_type
        self.reg_coef1 = reg_coef1
        self.reg_coef2 = reg_coef2

        for _ in range(n):

            y_pred = self.intercept_ + X@self.coef_ 
            error = y - y_pred
    
            w0_grad = -2 * error 
            self.intercept_ -= self.learning_rate * w0_grad.mean()

            if reg_type == 'Lasso':
                w_grad = -2 * X * error.reshape(-1, 1) + self.reg_coef1 * np.sign(self.coef_)
            elif reg_type == 'Ridge':
                w_grad = -2 * X * error.reshape(-1, 1) + 2 * self.reg_coef1 * self.coef_
            elif reg_type == 'ElasticNet':
                w_grad = -2 * X * error.reshape(-1, 1) + 2 * self.reg_coef1 * self.coef_ + self.reg_coef2 * np.sign(self.coef_)
            else:
                w_grad = -2 * X * error.reshape(-1, 1)
            
            self.coef_ -= self.learning_rate * w_grad.mean(axis=0)
            
    
    def predict(self, X): 
        X = np.array(X) 
        return self.intercept_ + X@self.coef_
    
    def score(self, X, y):
        predictions = self.intercept_ + X @ self.coef_
        return r2_score(y, predictions), mean_absolute_error(y, predictions), np.sqrt(mean_squared_error(y, predictions))


class LogReg:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y, n, reg_type=None, reg_coef1=0.001, reg_coef2=0.001):
        
        X = np.array(X)
        y = np.array(y)

        
        self.coef_ = np.random.normal(size=X.shape[1])
        self.intercept_ = np.random.normal(1)
        
        self.n = n
        self.reg_type = reg_type
        self.reg_coef1 = reg_coef1
        self.reg_coef2 = reg_coef2

        for _ in range(n):
        
            z = self.intercept_ + X @ self.coef_
            y_pred = self.sigmoid(z)
            error = y - y_pred 
    
            w0_grad = -error 
            self.intercept_ -= self.learning_rate * w0_grad.mean()

            if reg_type == 'Lasso':
                w_grad = -X * error.reshape(-1, 1) + self.reg_coef1 * np.sign(self.coef_)
            elif reg_type == 'Ridge':
                w_grad = -X * error.reshape(-1, 1) + 2 * self.reg_coef1 * self.coef_
            elif reg_type == 'ElasticNet':
                w_grad = -X * error.reshape(-1, 1) + 2 * self.reg_coef1 * self.coef_ + self.reg_coef2 * np.sign(self.coef_)
            else:            
                w_grad = -X * error.reshape(-1, 1)

            self.coef_ -= self.learning_rate * w_grad.mean(axis=0)

    
    def predict(self, X):
        X = np.array(X)
        z = self.intercept_ + X @ self.coef_
        return self.sigmoid(z)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred > 0.5)
        precision = precision_score(y, y_pred > 0.5)
        recall = recall_score(y, y_pred > 0.5)
        f1 = f1_score(y, y_pred > 0.5)
        return accuracy, precision, recall, f1  
    

uploaded_file = st.file_uploader(label='Upload your dataset', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if data is not None:  
        target_column_name = st.text_input(label='Input target column name')

        if target_column_name not in data.columns: 
            st.text(body='No such column name in provided dataset')
        else:    
            
            scaler_type = st.selectbox('Pick data scaler type', ('Standard', 'Min-max', 'Robust'))
            test_size = st.number_input(label='Input test size', value=0.2)
            learning_rate = st.number_input(label='Input learning rate', value=0.01)
            
            if test_size and scaler_type and learning_rate:

                if scaler_type == 'Standard':
                    scaler = StandardScaler()
                elif scaler_type == 'Min-max':
                    scaler = MinMaxScaler()
                elif scaler_type == 'Robust':   
                    scaler = RobustScaler()
  
  

                X = data.drop(columns=[target_column_name])
                y = data[target_column_name]


                X = pd.DataFrame(scaler.fit_transform(X))

                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                reg_type = st.selectbox('Pick regularization method', ('Lasso', 'Ridge', 'ElasticNet', 'None'))
                
                if reg_type == 'None':
                    reg_coef1 = 0.001
                    reg_coef2 = 0.001
                elif reg_type == 'ElasticNet':
                    reg_coef1 = st.number_input(label='Input alpha', format="%.3f", value=0.001)
                    reg_coef2 = st.number_input(label='Input beta', format="%.3f", value=0.001)
                else:
                    reg_coef1 = st.number_input(label='Input alpha', format="%.3f", value=0.001)   
                    reg_coef2 = 0.001 



                if y_train.nunique() > 2:
                    my_model = LinReg(learning_rate=learning_rate)
                    n_epoch = int(st.number_input(label='Input number of epochs', value=1000))
                    if n_epoch:
                        my_model.fit(X_train, y_train, n_epoch, reg_type=reg_type, reg_coef1=reg_coef1, reg_coef2=reg_coef2)
                        r2_train, mae_train, rmse_train = my_model.score(X_train, y_train)
                        r2_test, mae_test, rmse_test = my_model.score(X_test, y_test)                       
                        
                        col1, col2, col3 = st.columns(3)
                        col4, col5, col6 = st.columns(3)

                        col1.metric("**R2 train**", round(r2_train, 2))
                        col2.metric("**MAE train**", round(mae_train, 2))
                        col3.metric("**RMSE train**", round(rmse_train, 2))

                        col4.metric("**R2 test**", round(r2_test, 2))
                        col5.metric("**MAE test**", round(mae_test, 2))
                        col6.metric("**RMSE test**", round(rmse_test, 2))

                        weights = pd.DataFrame({'features': data.drop(columns=[target_column_name]).columns, 'weights': my_model.coef_})

                        st.dataframe(weights)
                else:
                    my_model = LogReg(learning_rate=learning_rate)
                    n_epoch = int(st.number_input(label='Input number of epochs', value=1000))
                    if n_epoch:
                        my_model.fit(X_train, y_train, n_epoch, reg_type=reg_type, reg_coef1=reg_coef1, reg_coef2=reg_coef2)
                        acc_train, pr_train, rec_train, f_train = my_model.score(X_train, y_train)
                        acc_test, pr_test, rec_test, f_test = my_model.score(X_test, y_test)                       
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col5, col6, col7, col8 = st.columns(4)

                        col1.metric("**Accuracy train**", round(acc_train, 2))
                        col2.metric("**Precision train**", round(pr_train, 2))
                        col3.metric("**Recall train**", round(rec_train, 2))
                        col4.metric("**F1 train**", round(f_train, 2))

                        col5.metric("**Accuracy test**", round(acc_test, 2))
                        col6.metric("**Precision test**", round(pr_test, 2))
                        col7.metric("**Recall test**", round(rec_test, 2))
                        col8.metric("**F1 test**", round(f_test, 2))

                        weights = pd.DataFrame({'features': data.drop(columns=[target_column_name]).columns, 'weights': my_model.coef_})

                        st.dataframe(weights)
