import pandas as pd
import numpy as np

n_rows = 100
np.random.seed(42)

data = {
    'open': np.random.uniform(100, 200, size=n_rows),
    'high': np.random.uniform(100, 200, size=n_rows),
    'low': np.random.uniform(100, 200, size=n_rows),
    'close': np.random.uniform(100, 200, size=n_rows),
    'volume': np.random.randint(1000, 10000, size=n_rows),
    'Name': ['AAPL'] * n_rows,
}
df = pd.DataFrame(data)
df.to_csv('all_stocks_5yr.csv', index=False)
print('Dummy stock market CSV generated: all_stocks_5yr.csv') 