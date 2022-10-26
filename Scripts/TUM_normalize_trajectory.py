#!/usr/bin/env python3

# transforms the input trajectory (TUM format), s.t. the first pose is at position (0, 0, 0) with unit quaternion (0, 0, 0, 1)

import pyquaternion

import numpy as np
import csv
import sys

def read_data(path):
    data = []
    with open(path, newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        column_names = next(csvreader)[1:]
        for row in csvreader:
            if row[0] == "#":
                continue
            data.append([float(x) for x in row])
    return data

def write_data(data, path_out):
    count = 0
    with open(path_out, "w") as f:
        for row in data:
            f.write("{:07.2f}".format(count / 30.0))
            translation = row[0]
            rotation = row[1]
            f.write(" {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
                translation[0], translation[1], translation[2],
                rotation[1], rotation[2], rotation[3], rotation[0]))
            f.write("\n")
            count += 1

def corrected_ICL(data):
    # Transform, so initial pose is origin
    initial_rotation = pyquaternion.Quaternion(data[0][7], data[0][4],data[0][5],data[0][6])
    T_initial = np.matrix(initial_rotation.transformation_matrix)
    T_initial[:,3] = np.matrix([data[0][1], data[0][2], data[0][3], 1]).T
    corrected = []
    for row in data:
        M = np.matrix(pyquaternion.Quaternion(row[7], row[4],row[5],row[6]).transformation_matrix)
        M[:,3] = np.matrix([row[1], row[2], row[3], 1]).T
        # Transform, so initial pose is origin
        M0 = T_initial.I * M
        rotation = pyquaternion.Quaternion(matrix=M0.A)
        translation = M0.A[:3,3]

        ## Special treatment, because ICL has negative fy (focal length)
        R = pyquaternion.Quaternion(matrix=np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]]))
        rotation = R * rotation * R.inverse
        translation[1] *= -1

        corrected.append([translation, rotation])
    return corrected

def corrected_TUM(data):
    initial_rotation = pyquaternion.Quaternion(data[0][7], data[0][4],data[0][5],data[0][6])
    T_initial = np.matrix(initial_rotation.transformation_matrix)
    T_initial[:,3] = np.matrix([data[0][1], data[0][2], data[0][3], 1]).T
    corrected = []
    for row in data:
        M = np.matrix(pyquaternion.Quaternion(row[7], row[4],row[5],row[6]).transformation_matrix)
        M[:,3] = np.matrix([row[1], row[2], row[3], 1]).T
        # Transform, so initial pose is origin
        M0 = T_initial.I * M
        rotation = pyquaternion.Quaternion(matrix=M0.A)
        translation = M0.A[:3,3]
        corrected.append([translation, rotation])
    return corrected

def main():
    path_in = sys.argv[1]
    path_out = sys.argv[2]

    data = read_data(path_in)

    if len(sys.argv) >=4 and sys.argv[3] == "ICL":
        corrected_data = corrected_ICL(data)
    else:
        corrected_data = corrected_TUM(data)

    write_data(corrected_data, path_out)

if __name__ == "__main__":
    main()
