import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

file_path='/home/output/model_output.csv'
# file_path='/home/output/model_output.csv'
correct = 0
equal_length = 0
missed = 0
wrong = 0


df=pd.read_csv(file_path)
for i in range(len(df)):
    if len(str(df['FileName'][i])) == len(str(df['OCR_prediction'][i])):
        equal_length += 1
        if str(df['FileName'][i]) == str(df['OCR_prediction'][i]):
            correct += 1
        else:
            wrong += 1
    else:
        if len(str(df['FileName'][i])) > len(str(df['OCR_prediction'][i])):
            missed += 1
        else:
            wrong += 1


# total_files = len(os.listdir("/home/images"))
total_files = len(os.listdir("/home/images"))
print(f"correct: {correct}")
print(f"equal_length: {equal_length}")
print(f"missed: {missed}")
print(f"wrong: {wrong}")


val=round(((correct*100) / total_files), 2)
plt.bar(['Correct OCR', 'Incorrect OCR', 'Missed OCR'],[correct, wrong, missed])
plt.xlabel('Results')
plt.ylabel('No of Samples')
plt.text(1-.08, 1200, 'Success  = {val} %'.format(val=val))
plt.savefig("/home/output/results.jpeg")


# tag_numbers = []
# for full_file_name in os.listdir("/home/images/"):
#     name = full_file_name.split(".j")[0]
#     another_split_list = name.split('_')
#     tag_number_gt = another_split_list[-2]
#     tag_numbers.append(tag_number_gt)
    # print(name)
    # break
# print(tag_numbers)
# print(len(tag_numbers))
# plt.imsave(os.path.join(path2outdir, update_name), temp_image)