import json
import numpy

f1 = open('extract_qs_gen_gpt4.json', 'r') # yentinglin breeze
# f2 = open('extract_qs_gen_followup_yentinglin.json', 'r')
data1 = json.load(f1)
# data2 = json.load(f2)

arr = numpy.zeros(23)
for i in range(len(data1)):
    # print(data1[i]['line number'])
    arr[int(data1[i]['line number']) // 50] += 1

print(arr)
print(sum(arr))
print(len(data1))
