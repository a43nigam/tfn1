import pandas as pd
import numpy as np

n_rows = 100
np.random.seed(42)

data = {
    'sentence': [f'This is a test sentence {i}.' for i in range(n_rows)],
    'label': np.random.randint(0, 2, size=n_rows),
}
df = pd.DataFrame(data)
df.to_csv('SST-2_train.tsv', sep='\t', index=False)
print('Dummy GLUE SST-2 TSV generated: SST-2_train.tsv') 