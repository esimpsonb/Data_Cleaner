from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer, StandardScaler


class Missing_Values:
    def __init__(self,df,method):
        self.df = df
        self.method = method
    
    def KNN(self,_n_neighbors=3):
        imputer = KNNImputer(n_neighbors=_n_neighbors)
        imputed_data = imputer.fit_transform(self.df)
        return pd.DataFrame(imputed_data, columns=self.df.columns)
    
    def zeros(self):
        imputer = SimpleImputer(strategy="constant")
        imputed_data = imputer.fit_transform(self.df)
        return pd.DataFrame(imputed_data, columns=self.df.columns)

    def mean(self):
        imputer = SimpleImputer(strategy="mean")
        imputed_data = imputer.fit_transform(self.df)
        return pd.DataFrame(imputed_data, columns=self.df.columns)

    def median(self):
        imputer = SimpleImputer(strategy="median")
        imputed_data = imputer.fit_transform(self.df)
        return pd.DataFrame(imputed_data, columns=self.df.columns)
    
    def mode(self):
        imputer = SimpleImputer(strategy="most_frequent")
        imputed_data = imputer.fit_transform(self.df)
        return pd.DataFrame(imputed_data, columns=self.df.columns)
    
    def processor(self):
        if self.method == "KNN": return Missing_Values.KNN()
        if self.method == "zeros": return Missing_Values.zeros()
        if self.method == "mean": return Missing_Values.mean()
        if self.method == "median": return Missing_Values.median()
        if self.method == "mode": return Missing_Values.mode()

class Encoding:
    def __init__(self,df):
        self.df = df

    def one_hot_encoding(self):
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        encoded_df = pd.get_dummies(self.df, columns=categorical_columns, prefix=categorical_columns)
        return encoded_df

    def label_encoding(self):
        encoder = preprocessing.LabelEncoder()
        encoded_df = self.df.copy()
        for column in encoded_df.columns:
            if encoded_df[column].dtype == 'object':
                encoded_df[column] = encoder.fit_transform(encoded_df[column].values)
        return encoded_df
    
    def processor(self):
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        n_values = self.df[categorical_columns].nunique()
        if (n_values <= 10).all():
            return self.one_hot_encoding()
        elif (n_values <= 20).all():
            return self.label_encoding()
        else:
            return self.df

class Outliers:
    def __init__(self,df,method):
        self.df = df
        self.method = method

    def z_score(self):
        numerical_columns = self.df.select_dtypes(include=[np.number])
        z_scores = np.abs((numerical_columns - numerical_columns.mean()) / numerical_columns.std())
        threshold = 3
        return self.df[(z_scores < threshold).all(axis=1)]
    
    def IQR(self):
        numerical_columns = self.df.select_dtypes(include=[np.number])
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        return self.df[~((numerical_columns < (Q1 - threshold * IQR)) | (numerical_columns > (Q3 + threshold * IQR))).any(axis=1)]
    
    def LOF(self):
        numerical_columns = self.df.select_dtypes(include=[np.number])
        lof = LocalOutlierFactor(n_neighbors=20)
        outlier_scores = lof.fit_predict(numerical_columns)
        self.df['Outlier'] = outlier_scores == -1
        return self.df[self.df['Outlier'] == False].drop(columns=['Outlier'])
    
    def DBSCAN(self):
        numerical_columns = self.df.select_dtypes(include=[np.number])
        eps = 3
        min_samples = 5
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(numerical_columns)
        outlier_mask = dbscan.labels_ == -1
        return self.df[~outlier_mask]
    
    def processor(self):
        if self.method == "z_score": return Outliers.z_score()
        if self.method == "IQR": return Outliers.IQR()
        if self.method == "LOF": return Outliers.LOF()
        if self.method == "DBSCAN": return Outliers.DBSCAN()

class Scale_Norm_Stand:

    def __init__(self,df):
        self.df = df
    
    def processor(self):
        numerical_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        min_max_scaler = MinMaxScaler()
        self.df[numerical_cols] = min_max_scaler.fit_transform(self.df[numerical_cols])

        normalizer = Normalizer()
        self.df[numerical_cols] = normalizer.fit_transform(self.df[numerical_cols])

        standard_scaler = StandardScaler()
        self.df[numerical_cols] = standard_scaler.fit_transform(self.df[numerical_cols])
        
        return self.df






