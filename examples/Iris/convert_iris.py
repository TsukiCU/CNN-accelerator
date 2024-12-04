import csv

iris_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

input_file = "iris.csv"
output_file = "iris.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    reader = csv.reader(infile)
    writer = outfile.write
    for row in reader:
        if not row:
            continue
        features = row[:4]
        label = iris_mapping[row[4]]
        writer(",".join(features) + f",{label}\n")