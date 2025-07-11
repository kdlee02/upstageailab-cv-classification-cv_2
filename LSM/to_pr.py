import pandas as pd

df = pd.read_csv('experiments/lsm_convnext_small_size_380_batch_32_os3-4-7-14x7_18classes/predictions.csv')

df[df['target']==17] = 7

df.to_csv('experiments/lsm_convnext_small_size_380_batch_32_os3-4-7-14x7_18classes/predictions_to_class.csv',index=False)