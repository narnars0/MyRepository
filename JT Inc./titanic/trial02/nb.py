import csv as csv
import numpy as np
from sklearn.naive_bayes import GaussianNB

csv_file_object = csv.reader(open('train.csv', 'rb'))
header = csv_file_object.next()
data=[]
count_row = 0

for row in csv_file_object:
	if row[4] == "female":
		row[4] = 10.0
	else:
		row[4] = 0.0

	if row[9] == "":
		row[9] = 0.0
	else:
		row[9] = float(row[9])

	if row[5] == "":
		row[5] = 40.0
	else:
		row[5] = float(row[5])

	if row[10] == "":
		row[10] = 0.0
	else:
		row[10] = 10.0

	if row[6] == "":
		row[6] = 0.0
	else:
		row[6] = float(row[6])

	if row[2] == "":
		row[2] = 0.0
	else:
		row[2] = float(row[2])

	data.append(row)
	count_row += 1

features = []
data = np.array(data)

features = data[:, [2, 4, 9]].astype(np.float)
output = data[:, 1]

clf = GaussianNB()
clf.fit(features, output)

test_file = open("test.csv", "rb")
test_file_object = csv.reader(test_file)
header = test_file_object.next()

prediction_file = open("naivebayes.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])

for row in test_file_object:
	if row[3] == "female":
		row[3] = 10.0
	else:
		row[3] = 0.0

	if row[8] == "":
		row[8] = 0.0
	else:
		row[8] = float(row[8])

	if row[4] == "":
		row[4] = 40.0
	else:
		row[4] = float(row[4])

	if row[9] == "":
		row[9] = 0.0
	else:
		row[9] = 10.0

	if row[5] == "":
		row[5] = 0.0
	else:
		row[5] = float(row[5])

	if row[1] == "":
		row[1] = 0.0
	else:
		row[1] = float(row[1])

	prediction_file_object.writerow([row[0], clf.predict([[row[1], row[3], row[8]]])[0]])

test_file.close()
prediction_file.close()

