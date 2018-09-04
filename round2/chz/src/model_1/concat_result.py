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

def from_scientific_to_number(values):
    values = values.split(';')
    values = ['%.8f' % float(i) for i in values]
    return ';'.join(values)

results = pd.DataFrame()
for cloth in clothes_types:
    cloth_result = pd.read_csv('/root/Project/src/model_1/result/%s.csv' % cloth)
    results = pd.concat([results, cloth_result], axis=0)

results['AttrValueProbs'] = results['AttrValueProbs'].apply(from_scientific_to_number)

results.to_csv('/root/Project/src/model_1/result/submission.csv', index=None, header=None)
