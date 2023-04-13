#Merger
import pandas as pd
xl= pd.ExcelFile('Dataset_HEAs15.xlsx')
dfs = [xl.parse(sheet_name) for sheet_name in xl.sheet_names]
