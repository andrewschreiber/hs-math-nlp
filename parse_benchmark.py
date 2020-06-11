import os
import sys
import glob

path = sys.argv[1]

results = {}
counter = 0
total_correct = 0
for filename in glob.glob(os.path.join(path, "*.txt")):
    counter += 1
    module = filename.split("/")[-1].split(".")[0]
    with open(os.path.join(os.getcwd(), filename), "r") as f:
        correct = int(f.read().split("\n")[0])
        results[module] = correct
        total_correct += correct

performance = total_correct / (counter * 10000)
print(results)
print(performance)
