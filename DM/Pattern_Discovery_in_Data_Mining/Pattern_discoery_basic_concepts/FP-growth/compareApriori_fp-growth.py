# @Time : 2018/11/9 下午12:48 
# @Author : Kaishun Zhang 
# @File : compareApriori_fp-growth.py 
# @Function: compare result of apriori and fp-growth
data_apriori = open('../Apriori/result2.txt').read().split('\n')
data_fp_growth = open('result-fp-growth').read().split('\n')
dic_apriori = dict()
dic_fp_growth = dict()
for data in data_apriori:
    dat = data.split(':')[1].split(';');
    dat = sorted(dat)
    dic_apriori[';'.join(dat)] = data.split(':')[0]

for data in data_fp_growth:
    dat = data.split('; ')[0].split(';')
    dat = sorted(dat)
    dic_fp_growth[';'.join(dat)] = data.split('; ')[1]


for f_key,f_value in dic_fp_growth.items():
    if not f_key in dic_apriori.keys():
        print(f_key, ' -> ',f_value);
    else:
        if f_value != dic_apriori[f_key]:
            print(f_key, '--- > ',f_value)