#!/usr/bin/env python3

# converts the trajectory output of InfiniTAM into TUM format

import csv
import sys

def read_data(path):
    data = []
    with open(path, newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        column_names = next(csvreader)[1:]
        for row in csvreader:
            data.append(row)
    return data

def write_TUM(data, path_out):
    count = 0
    with open(path_out, "w") as f:
        for row in data:
            f.write("{:07.2f}".format(count / 30.0))
            for x in row[:7]:
                f.write(" {:.8f}".format(float(x)))
            f.write("\n")
            count += 1

def main():
    path_in = sys.argv[1]
    path_out = sys.argv[2]

    data = read_data(path_in)
    write_TUM(data, path_out)

if __name__ == "__main__":
    main()
