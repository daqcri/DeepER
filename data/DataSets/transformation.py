import os
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', required= True)
parser.add_argument('-label', required=False, default=1) #put another number if you have a train file with labels (zero or one)
opt = parser.parse_args()

dataset = opt.dataset
label = opt.label

path = "%s" %(dataset)


train_file_path = os.path.join(path, "train.csv")
ltable_file_path = os.path.join(path, "tableA.csv")
rtable_file_path = os.path.join(path, "tableB.csv")



def readcsv(filename):
    with open(filename,newline= "\n", encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        return list(reader)

def csvToList(file):
    temp = []
    for i in range (len (file)-1):
        temp.insert(i,file[i+1][:1])
    return temp

def makefilecsv_nolabel (ltable,rtable,train):
    l=csvToList(ltable)
    r=csvToList(rtable)
    train_temp=[]
    for i in range (len(train)-1):
        train_temp.insert(i,(train[i+1][0] ,train[i+1][1]))
    train_def = "\"id_ltable\",\"id_rtable\"\n"
    for j in range(len(train_temp)):
        print(train_temp[j])
        train_def += (str(l[int((train_temp[j])[0])]) + "," + str(r[int((train_temp[j])[1])]) +"\n")
    train_def = train_def.replace("['","")
    train_def = train_def.replace("']","")
    return train_def

def makefilecsv_label (ltable,rtable,train):
    l=csvToList(ltable)
    r=csvToList(rtable)
    train_temp=[]
    for i in range (len(train)-1):
        if ((train[i+1][2])=='1'):
            train_temp.insert(i,(train[i+1][0],train[i+1][1]))
    train_def = "\"id_ltable\",\"id_rtable\"\n"
    for touple in train_temp:
        train_def += (str(l[int(touple[0])]) + "," + str(r[int(touple[1])]) +"\n")
    train_def = train_def.replace("['","")
    train_def = train_def.replace("']","")
    return train_def




if (label==1):
    train = makefilecsv_nolabel(readcsv(ltable_file_path),readcsv(rtable_file_path),readcsv(train_file_path))
else:
    train = makefilecsv_label(readcsv(ltable_file_path),readcsv(rtable_file_path),readcsv(train_file_path))


with open((path + "/%s_perfectMapping.csv" % (dataset)),"w+", encoding="utf-8") as f:
    f.write(train)