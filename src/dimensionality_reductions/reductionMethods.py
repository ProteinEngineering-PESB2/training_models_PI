import pandas as pd

class Reductions(object):

    def __init__(self,
            dataset=None,
            column_to_ignore=[]):
        
        self.dataset = dataset
        self.column_to_ignore = column_to_ignore
    
    def generateDatasetPostReduction(self, transform_values, n_components):

        header = [f"p_{i+1}" for i in range(n_components)]
        df_data = pd.DataFrame(data=transform_values, columns=header)
        return df_data