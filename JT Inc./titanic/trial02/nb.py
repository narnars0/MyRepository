import csv as csv
import numpy as np
from sklearn.naive_bayes import GaussianNB

csv_file_object = csv.reader(open('train.csv', 'rb'))
header = csv_file_object.next()
data=[]
count_row = 0

for row in csv_file_object:
	if row[4] == "female":
		row[4] = 1.0
	else:
		row[4] = 0.0

	if row[9] == "":
		row[9] = 0.0
	else:
		row[9] = float(row[9])

	if row[5] == "":
		row[5] = 0.0
	else:
		row[5] = float(row[5])

	data.append(row)
	count_row += 1

features = []
data = np.array(data)

features = data[:, [4, 9]].astype(np.float)
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
		row[3] = 1.0
	else:
		row[3] = 0.0

	if row[8] == "":
		row[8] = 0.0
	else:
		row[8] = float(row[8])

	prediction_file_object.writerow([row[0], clf.predict([[row[3], row[8]]])[0]])

test_file.close()
prediction_file.close()

