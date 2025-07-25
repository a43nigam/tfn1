import pandas as pd
import numpy as np

n_rows = 100
np.random.seed(42)

dates = pd.date_range('2009-01-01', periods=n_rows, freq='10min')
data = {
    'Date Time': dates.strftime('%Y-%m-%d %H:%M:%S'),
    'T (degC)': np.random.uniform(-5, 25, size=n_rows),
}
for i in range(1, 14):
    data[f'feature{i}'] = np.random.uniform(0, 1, size=n_rows)
df = pd.DataFrame(data)
df.to_csv('jena_climate_2009_2016.csv', index=False)
print('Dummy Jena climate CSV generated: jena_climate_2009_2016.csv') 