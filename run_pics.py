import os

files = ['pics_exp3_feat_dims_thesis.py',
         'pics_exp3_memory_ratio_thesis.py',
         'pics_exp4_memory_thesis.py',
         'pics_exp4_stack_time_thesis.py',
         'pics_exp5_time_contrast_thesis.py',
         'pics_exp7_paras_acc_thesis.py']

for file in files:
    cmd = 'python ' + file
    print(cmd)
    os.system(cmd)