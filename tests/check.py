import pandas as pd
df = pd.read_csv("sample_data/timeseries_sample.csv")
# manually test the new function
from app.nodes.file_loader import _parse_date_columns
df2 = _parse_date_columns(df)
print(df2.dtypes)  # date should now show datetime64[ns]
exit()