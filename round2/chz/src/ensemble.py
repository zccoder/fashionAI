import click
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

@click.command()
@click.option('--output_csv', type=str, required=True)
def main(output_csv):
	result_1 = pd.read_csv('/root/Project/src/model_1/result/submission.csv', names=['path', 'type', 'prob']).sort_values(by='path').reset_index(drop=True)
	result_2 = pd.read_csv('/root/Project/src/model_2/result/submission.csv', names=['path', 'type', 'prob']).sort_values(by='path').reset_index(drop=True)

	result_1['prob'] = result_1['prob'].apply(lambda x: np.array([float(i) for i in x.split(';')]))
	result_2['prob'] = result_2['prob'].apply(lambda x: np.array([float(i) for i in x.split(';')]))

	rows = result_1.shape[0]
	probs = [0] * rows

	clothes_types = ['collar_design_labels', 'lapel_design_labels', 'neck_design_labels', 'neckline_design_labels', 
				     'coat_length_labels', 'pant_length_labels', 'skirt_length_labels', 'sleeve_length_labels']

	label_count = {'coat_length_labels': 8, 'collar_design_labels': 5, 'lapel_design_labels': 5, 'neck_design_labels': 5, 
				   'neckline_design_labels': 10, 'pant_length_labels': 6, 'skirt_length_labels': 6, 'sleeve_length_labels': 9}

	ensembled = pd.DataFrame()

	for cloth in clothes_types:
		cloth_result_1 = result_1[result_1['type'] == cloth].reset_index(drop=True)
		cloth_result_2 = result_2[result_2['type'] == cloth].reset_index(drop=True)
		result_1_array = np.ones((cloth_result_1.shape[0], label_count[cloth]))
		result_2_array = np.ones((cloth_result_2.shape[0], label_count[cloth]))
		for i in range(cloth_result_1.shape[0]):
		    result_1_array[i] = cloth_result_1['prob'][i]
		for i in range(cloth_result_2.shape[0]):
		    result_2_array[i] = cloth_result_2['prob'][i]

		temp = cloth_result_1[['path', 'type', 'prob']].reset_index(drop=True)

		prob = np.log(np.array([result_1_array, result_2_array]) + 1e-6)
		prob = np.mean(prob, axis=0)
		prob -= np.max(prob, axis=1, keepdims=True)
		prob = np.exp(prob)
		prob /= np.sum(prob, axis=1, keepdims=True)
		
		for i in range(temp.shape[0]):
			temp['prob'][i] = prob[i]

		ensembled = pd.concat([ensembled, temp], axis=0)
		

	ensembled['prob'] = ensembled['prob'].apply(lambda x: ';'.join('%.8f' % output for output in x))

	ensembled.to_csv(output_csv, header=None, index=None)

if __name__ == '__main__':
	main()
