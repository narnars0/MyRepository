import csv as csv
import numpy as np
from sklearn.naive_bayes import GaussianNB

csv_file_object = csv.reader(open('train.csv', 'rb'))
header = csv_file_object.next()
data=[]

for row in csv_file_object:
	data.append(row)

data = np.array(data)

women_only_stats = data[:, 4] == "female"
men_only_stats = data[:, 4] == "male"

women_onboard = women_only_stats.astype(np.float)
men_onboard = men_only_stats.astype(np.float)

id_stats = data[:, 0].astype(np.float)
fare_stats = data[:, 9].astype(np.float)

id_stats = id_stats.tolist()
fare_stats = fare_stats.tolist()
women_only_stats = women_onboard.tolist()

features = []

number_passengers = np.size(data[:, 1].astype(np.float))

for i in range(0, number_passengers):
	features.append([fare_stats[i], women_only_stats[i]])

output = data[:, 1]

features = np.array(features)
output = np.array(output)

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
		prediction_file_object.writerow([row[0], clf.predict([[1.0, float(row[8])]])[0]])
	else:
		if row[8] != "":
			prediction_file_object.writerow([row[0], clf.predict([[0.0, float(row[8])]])[0]])
		else:
			prediction_file_object.writerow([row[0], clf.predict([[0.0, 0.0]])[0]])

test_file.close()
prediction_file.close()

