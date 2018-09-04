import pandas as pd
import numpy as np

clothes_types = ['coat_length_labels',
                 'collar_design_labels',
                 'lapel_design_labels',
                 'neck_design_labels',
                 'neckline_design_labels',
                 'pant_length_labels',
                 'skirt_length_labels',
                 'sleeve_length_labels',
                 ]

results = pd.DataFrame()
for cloth in clothes_types:
    cloth_result = pd.read_csv('/root/Project/src/model_2/result/%s.csv' % cloth, header=None)
    results = pd.concat([results, cloth_result], axis=0)

results.to_csv('/root/Project/src/model_2/result/submission.csv', index=None, header=None)
