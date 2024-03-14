from Methods import *

class Data_Cleaner:

    def __init__(self,df,missing_method,outliers_method):
        self.df = df
        self.missing_method = missing_method
        self.outliers_method = outliers_method
    
    def convert_to_float(self):
        for column in self.df.columns:
            try:
                self.df[column] = self.df[column].astype(float)
            except ValueError:
                pass
    
    def cleaning(self):
        self.convert_to_float()
        missing_handler = Missing_Values(self.df, self.missing_method)
        self.df = missing_handler.processor()
        outliers = Outliers(self.df,self.outliers_method)
        self.df = outliers.processor()
        scale_norm_stand = Scale_Norm_Stand(self.df)
        self.df = scale_norm_stand.processor()
        encoding = Encoding(self.df)
        self.df = encoding.processor()
        return self.df