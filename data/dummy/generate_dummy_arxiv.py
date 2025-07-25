import pandas as pd
import numpy as np

n_rows = 100
np.random.seed(42)

categories = [f'cat{i}' for i in range(10)]
data = {
    'abstract': [f'This is a dummy abstract {i}.' for i in range(n_rows)],
    'category': np.random.choice(categories, size=n_rows),
}
df = pd.DataFrame(data)
df.to_csv('arxiv_sample.csv', index=False)
print('Dummy arXiv CSV generated: arxiv_sample.csv') 