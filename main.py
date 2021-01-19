from engine import *


adc_path = './data/sample/ADC'
t2wi_path = './data/sample/T2WI'
label_path = './data/sample/label.txt'

# train(adc_path, t2wi_path, label_path)  # to train the CNN network uncomment this, please go to the function to read the instruction
# test(adc_path, t2wi_path, label_path)   # to evaluate the CNN network uncomment this, please go to the function to read the instruction
# predict(adc_path, t2wi_path, label_path, mode='train') # to get the merged CNN features and prediction on train data
# predict(adc_path, t2wi_path, label_path, mode='test')  # to get the merged CNN features and prediction on test data
# svm_stage_1(adc_path, t2wi_path, label_path, adc_path, t2wi_path, label_path)                # run only the svm stage-1
# svm_stage_2_combine_1(adc_path, t2wi_path, label_path, adc_path, t2wi_path, label_path)      # run svm stage-2 with combine-1
# svm_stage_2_combine_2(adc_path, t2wi_path, label_path, adc_path, t2wi_path, label_path)    # run svm stage-2 with combine-2
svm_stage_2_combine_3(label_path, label_path)                   # run svm stage-2 with combine-3
