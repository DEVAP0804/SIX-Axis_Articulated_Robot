from sympy import symbols, cos, sin, pi, simplify, pprint, tan, expand_trig, sqrt, trigsimp, atan2
from sympy.matrices import Matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import numpy as np  # Scientific computing library
import math



def get_angles(px1, py1, pz1, roll, pitch, yaw):

    theta = [((math.pi * roll) / 180), ((math.pi * pitch) / 180), ((math.pi * yaw) / 180)]
    a1 = 0.1012
    a2 = 0.362
    a3 = 0.04621
    a4 = 0.0
    a5 = 0.0
    a6 = 0.0
    a7 = 0.0
    d1 = 0.3975
    d2 = 0.0
    d3 = 0.0
    d4 = 0.33254
    d5 = 0.0
    d6 = 0.04602
    d7 = 0.0

    th1 = 0.0
    th2 = -90
    th3 = 0.0
    th4 = 0.0
    th5 = 0.0
    th6 = 180
    th7 = 0.0

    alp1 = -90
    alp2 = 0.0
    alp3 = -90
    alp4 = 90
    alp5 = -90
    alp6 = 0.0
    alp7 = 0.0

    q = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    q_4 = q_5 = q_6 = 0.0

    d_h_table = np.array([[np.deg2rad(th1), np.deg2rad(alp1), a1, d1],
                          [np.deg2rad(th2), np.deg2rad(alp2), a2, d2],
                          [np.deg2rad(th3), np.deg2rad(alp3), a3, d3],
                          [np.deg2rad(th4), np.deg2rad(alp4), a4, d4],
                          [np.deg2rad(th5), np.deg2rad(alp5), a5, d5],
                          [np.deg2rad(th6), np.deg2rad(alp6), a6, d6],
                          [np.deg2rad(th7), np.deg2rad(alp7), a7, d7]])

    i = 0
    homgen_0_1 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])

    # Homogeneous transformation matrix from frame 1 to frame 2
    i = 1
    homgen_1_2 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])

    # Homogeneous transformation matrix from frame 2 to frame 3
    i = 2
    homgen_2_3 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])

    # Homogeneous transformation matrix from frame 3 to frame 4
    i = 3
    homgen_3_4 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])

    # Homogeneous transformation matrix from frame 4 to frame 5
    i = 4
    homgen_4_5 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])

    # Homogeneous transformation matrix from frame 5 to frame 6
    i = 5
    homgen_5_6 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])

    homgen_0_6_fk = homgen_0_1 @ homgen_1_2 @ homgen_2_3 @ homgen_3_4 @ homgen_4_5 @ homgen_5_6

    # Print the homogeneous transformation matrices
    # Print the homogeneous transformation matrices
    # print("Homogeneous Matrix Frame 0 to Frame 1:")
    # print(homgen_0_1)
    # print()
    # print("Homogeneous Matrix Frame 1 to Frame 2:")
    # print(homgen_1_2)
    # print()
    # print("Homogeneous Matrix Frame 2 to Frame 3:")
    # print(homgen_2_3)
    # print()
    # print("Homogeneous Matrix Frame 3 to Frame 4:")
    # print(homgen_3_4)
    # print()
    # print("Homogeneous Matrix Frame 4 to Frame 5:")
    # print(homgen_4_5)
    # print()
    # print("Homogeneous Matrix Frame 5 to Frame 6:")
    # print(homgen_5_6)
    # print()
    # print("Homogeneous Matrix Frame 0 to Frame 6 FK:")
    # print(homgen_0_6_fk)
    # print()
    # print("DH TABLE:")
    # print(d_h_table)

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    # INVERSE KINEMATICS STARTING
    homgen_pxyz = np.array([[R[0][0], R[0][1], R[0][2], px1],
                            [R[1][0], R[1][1], R[1][2], py1],
                            [R[2][0], R[2][1], R[2][2], pz1],
                            [0, 0, 0, 1.0]])
    print("NEW POSITIONSSSSSSSSSSSSSSSSSSSSSSSSS", homgen_pxyz)

    homgen_0_6 = homgen_pxyz
    print("NEW POSITIONSSSSSSSSSSSSSSSSSSSSSSSSS for IK", homgen_0_6)
    # Joint Angle Calculation Theta 1:
    L6 = d6
    L2 = a2
    L3 = d4
    # A2 = a3
    Px1 = ((homgen_0_6[0][3]) - (L6 * homgen_0_6[0][2]))
    Py1 = ((homgen_0_6[1][3]) - (L6 * homgen_0_6[1][2]))
    Pz1 = ((homgen_0_6[2][3]) - (L6 * homgen_0_6[2][2]))
    # print("L6 =========", L6)
    # print("px =========", homgen_0_6[0][3])
    # print("wcx =========", (L6*homgen_0_6[0][2]))
    # print("py =========", homgen_0_6[1][3])
    # print("wcy =========", (L6*homgen_0_6[1][2]))
    # print("pz =========", homgen_0_6[2][3])
    # print("wcz =========", (L6*homgen_0_6[2][2]))
    # print("Px1, Py1, Pz1 =", Px1,Py1,Pz1)
    homgen_0_6_pxyz = np.array([[Px1, 0, 0, 0],
                                [Py1, 0, 0, 0],
                                [Pz1, 0, 0, 0],
                                [1, 0, 0, 0]])

    # print(homgen_0_6_pxyz)
    # print("L6, px, wcx = ",L6,(homgen_0_6[0][3]), (L6*homgen_0_6[0][2]))
    # print("L6, py, wcy = ",L6,(homgen_0_6[0][3]), (L6*homgen_0_6[0][2]))

    theta1_1 = abs(round(((180 * math.atan2((Py1), (Px1))) / math.pi), 1))
    # print("Theta1_1 ======== ", (theta1_1))
    theta1_2 = 180 + theta1_1
    # print("Theta1_2 ======== ", (theta1_2))
    q[0][0] = q[0][1] = q[0][2] = q[0][3] = theta1_1
    q[0][4] = q[0][5] = q[0][6] = q[0][7] = theta1_2

    # Joint Angle Calculation Theta 2:

    th1_1 = theta1_1

    d_h_table = np.array([[np.deg2rad(th1_1), np.deg2rad(alp1), a1, d1],
                          [np.deg2rad(th2), np.deg2rad(alp2), a2, d2],
                          [np.deg2rad(th3), np.deg2rad(alp3), a3, d3],
                          [np.deg2rad(th4), np.deg2rad(alp4), a4, d4],
                          [np.deg2rad(th5), np.deg2rad(alp5), a5, d5],
                          [np.deg2rad(th6), np.deg2rad(alp6), a6, d6],
                          [np.deg2rad(th7), np.deg2rad(alp7), a7, d7]])
    i = 0
    homgen_0_1_t2_11 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                  np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                  d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                                 [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                  -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                  d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                                 [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                                 [0, 0, 0, 1]])
    th1_2 = theta1_2

    d_h_table = np.array([[np.deg2rad(th1_2), np.deg2rad(alp1), a1, d1],
                          [np.deg2rad(th2), np.deg2rad(alp2), a2, d2],
                          [np.deg2rad(th3), np.deg2rad(alp3), a3, d3],
                          [np.deg2rad(th4), np.deg2rad(alp4), a4, d4],
                          [np.deg2rad(th5), np.deg2rad(alp5), a5, d5],
                          [np.deg2rad(th6), np.deg2rad(alp6), a6, d6],
                          [np.deg2rad(th7), np.deg2rad(alp7), a7, d7]])
    i = 0

    homgen_0_1_t2_12 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                  np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                  d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                                 [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                  -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                  d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                                 [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                                 [0, 0, 0, 1]])

    # invhom_theta1_1 = hom_theta1_1.transpose()
    # print("inverse hom_trans for theta1_1 = ", invhom_theta1_1)
    # homgen_0_6_t2 = homgen_0_1 @ homgen_0_6_pxyz
    # homgen_4_6 = homgen_4_5 @ homgen_5_6
    # homgen_6_4 = (np.linalg.inv(homgen_4_6))
    # print("Theta 11   INV MATRIX  PM Matrix & M MAtrix")
    homgen_1_0 = (np.linalg.inv(homgen_0_1_t2_11))
    hom_t14_t21 = homgen_1_0 @ homgen_0_6_pxyz
    # print("INV MATRIX",homgen_1_0)
    # print("PM MATRIX",homgen_0_6_pxyz)
    # print("Matrix 4-1 for theta2 21",hom_t14_t21)

    # print("Theta 12   INV MATRIX  PM Matrix & M MAtrix")
    homgen_1_0 = (np.linalg.inv(homgen_0_1_t2_12))
    hom_t14_t22 = homgen_1_0 @ homgen_0_6_pxyz
    # print("INV MATRIX",homgen_1_0)
    # print("PM MATRIX",homgen_0_6_pxyz)
    # print("Matrix 4-1 for theta2 22",hom_t14_t22)
    # L4 = math.sqrt((math.pow(A2,2))+(math.pow(L3,2)))

    r = math.sqrt((math.pow(hom_t14_t21[0][0], 2)) + (math.pow(hom_t14_t21[1][0], 2)))
    print("L2 L3 r=", L2, L3, r)
    beta = math.atan2(-hom_t14_t21[1][0], hom_t14_t21[0][0])
    num = ((math.pow(L2, 2)) + (math.pow(r, 2)) - (math.pow(L3, 2)))
    den = (2 * r * L2)
    div = num / den
    if div <= 1.00 and div >= -1.0:
        gamma = math.acos(div)
    gamma = math.acos(div)
    print("gamma=", gamma)
    # print("DIV 21 = ===============", div)
    # gamma = math.acos(div)
    theta2_11 = 90 - ((beta * 180) / math.pi) - ((gamma * 180) / math.pi)
    theta2_12 = 90 - ((beta * 180) / math.pi) + ((gamma * 180) / math.pi)

    r = math.sqrt((math.pow(hom_t14_t22[0][0], 2)) + (math.pow(hom_t14_t22[1][0], 2)))
    beta1 = math.atan2(-hom_t14_t22[1][0], hom_t14_t22[0][0])
    num = ((math.pow(L2, 2)) + (math.pow(r, 2)) - (math.pow(L3, 2)))
    den = (2 * r * L2)
    div = num / den
    if div <= 1.00 and div >= -1.0:
        gamma1 = math.acos(div)
    # print("DIV 22 = ===============", div)
    gamma1 = math.acos(div)
    theta2_21 = 90 - ((beta1 * 180) / math.pi) - ((gamma1 * 180) / math.pi)
    theta2_22 = 90 - ((beta1 * 180) / math.pi) + ((gamma1 * 180) / math.pi)
    q[1][0] = q[1][1] = ((theta2_11))
    q[1][2] = q[1][3] = ((theta2_12))
    q[1][4] = q[1][5] = ((theta2_21))
    q[1][6] = q[1][7] = ((theta2_22))
    # print(" THETA2_11  ========== ", theta2_11)
    # print(" THETA2_12  ========== ", theta2_12)
    # print(" THETA2_21  ========== ", theta2_21)
    # print(" THETA2_22  ========== ", theta2_22)
    # Joint Angle Calculation Theta 3:
    homgen_0_4 = homgen_0_1 @ homgen_1_2 @ homgen_2_3 @ homgen_3_4
    # print("Homogeneous Matrix Frame 1 to Frame 3: Theta 11")
    homgen_1_0 = (np.linalg.inv(homgen_0_1_t2_11))
    hom_t14_t31 = homgen_1_0 @ homgen_0_6_pxyz
    # print("Matrix 4-1 for theta 11",hom_t14_t21)
    # print("Homogeneous Matrix Frame 1 to Frame 3: Theta 12")
    homgen_1_0 = (np.linalg.inv(homgen_0_1_t2_12))
    hom_t14_t32 = homgen_1_0 @ homgen_0_6_pxyz
    # print("Matrix 4-1 for theta 12",hom_t14_t22)
    r = math.sqrt((math.pow(hom_t14_t31[0][0], 2)) + (math.pow(hom_t14_t31[1][0], 2)))
    num = ((math.pow(L2, 2)) + (math.pow(L3, 2)) - (math.pow(r, 2)))
    den = (2 * L2 * L3)
    div = num / den
    phi = math.acos(div)

    # print("----------THETA 3 11 PHI------------")
    num = ((math.pow(L2, 2)) + (math.pow(L3, 2)) - (math.pow(r, 2)))
    den = (2 * L2 * L3)
    div = num / den
    eta = math.acos(div)

    # print("-----------THETA 3 12 ETA------------")
    # print(" PHI  = ", ((phi*180)/math.pi))
    # print(" ETA  = ", ((eta*180)/math.pi))

    theta3_11 = 90 - ((eta * 180) / math.pi)
    theta3_12 = ((eta * 180) / math.pi) - (3 * 90)

    # print(" theta3_11    =========== ", theta3_11)
    # print(" theta3_12    =========== ", theta3_12)

    r = math.sqrt((math.pow(hom_t14_t32[0][0], 2)) + (math.pow(hom_t14_t32[1][0], 2)))
    num = ((math.pow(L2, 2)) + (math.pow(L3, 2)) - (math.pow(r, 2)))
    den = (2 * L2 * L3)
    div = num / den
    phi = math.acos(div)

    # print("------------THETA 3 21 PHI-----------")
    num = ((math.pow(L2, 2)) + (math.pow(L3, 2)) - (math.pow(r, 2)))
    den = (2 * L2 * L3)
    div = num / den
    eta = math.acos(div)

    # print("------------THETA 3 22 ETA------------")
    # print(" PHI  = ", ((phi*180)/math.pi))
    # print(" ETA  = ", ((eta*180)/math.pi))

    theta3_21 = 90 - ((eta * 180) / math.pi)
    theta3_22 = ((eta * 180) / math.pi) - (3 * 90)

    # print(" theta3_21    =========== ", theta3_21)
    # print(" theta3_22    =========== ", theta3_22)

    q[2][0] = q[2][1] = ((theta3_11))
    q[2][2] = q[2][3] = ((theta3_12))
    q[2][4] = q[2][5] = ((theta3_21))
    q[2][6] = q[2][7] = ((theta3_22))
    for x in range(3):
        print(q[x][0], q[x][1], q[x][2], q[x][3], q[x][4], q[x][5], q[x][6], q[x][7])

    # Joint Angle Calculation Theta 456:
    th = np.array([[q[0][0], q[0][1], q[0][2], q[0][3], q[0][4], q[0][5], q[0][6], q[0][7]],
                   [(-90.0 + q[1][0]), (-90.0 + q[1][1]), (-90.0 + q[1][2]), (-90.0 + q[1][3]), (-90.0 + q[1][4]),
                    (-90.0 + q[1][5]), (-90.0 + q[1][6]), (-90.0 + q[1][7])],
                   [q[2][0], q[2][1], q[2][2], q[2][3], q[2][4], q[2][5], q[2][6], q[2][7]]])
    s = 1
    for l in range(8):

        d_h_table = np.array([[np.deg2rad(th[0][l]), np.deg2rad(alp1), a1, d1],
                              [np.deg2rad(th[1][l]), np.deg2rad(alp2), a2, d2],
                              [np.deg2rad(th[2][l]), np.deg2rad(alp3), a3, d3],
                              [np.deg2rad(th4), np.deg2rad(alp4), a4, d4],
                              [np.deg2rad(th5), np.deg2rad(alp5), a5, d5],
                              [np.deg2rad(th6), np.deg2rad(alp6), a6, d6],
                              [np.deg2rad(th7), np.deg2rad(alp7), a7, d7]])

        i = 0
        homgen_0_1 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                               [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                               [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                               [0, 0, 0, 1]])

        # Homogeneous transformation matrix from frame 1 to frame 2
        i = 1
        homgen_1_2 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                               [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                               [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                               [0, 0, 0, 1]])

        # Homogeneous transformation matrix from frame 2 to frame 3
        i = 2
        homgen_2_3 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                               [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                               [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                               [0, 0, 0, 1]])

        # Homogeneous transformation matrix from frame 3 to frame 4
        i = 3
        homgen_3_4 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                               [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                               [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                               [0, 0, 0, 1]])

        # Homogeneous transformation matrix from frame 4 to frame 5
        i = 4
        homgen_4_5 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                               [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                               [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                               [0, 0, 0, 1]])

        # Homogeneous transformation matrix from frame 5 to frame 6
        i = 5
        homgen_5_6 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                               [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                                d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                               [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                               [0, 0, 0, 1]])

        # homgen_0_3 = homgen_0_1 @ homgen_1_2 @ homgen_2_3
        homgen_1_0 = np.linalg.inv(homgen_0_1)
        homgen_2_1 = np.linalg.inv(homgen_1_2)
        homgen_3_2 = np.linalg.inv(homgen_2_3)
        homgen_3_6 = homgen_3_2 @ homgen_2_1 @ homgen_1_0 @ homgen_0_6_fk @ homgen_pxyz

        # print("Homogeneous Matrix Frame 3 to Frame 6:")
        # print("")
        # print("")
        # print("HT = l",l)
        # print(homgen_1_0)
        # print("")
        thresh = 1e-12
        t22 = homgen_3_6[2][2]
        t22 = t22 - 1.0
        at22 = math.fabs(t22)
        # print("t22 = ",t22)
        # print("at22 = ",at22)
        # print("S = ", s)
        if at22 > thresh:
            if s == 1:  # wrist up
                q_4 = math.atan2(homgen_3_6[1][2], homgen_3_6[0][2])
                q_6 = math.atan2(-homgen_3_6[2][1], homgen_3_6[2][0])
                s = 0
            # print("UP THETA 4 = RAD = ",q_4)
            # print("UP THETA 5 = RAD = ",q_5)
            # print("UP THETA 6 = RAD = ",q_6)
            else:  # wrist down
                q_4 = (math.atan2(homgen_3_6[1][2], homgen_3_6[0][2])) + math.pi
                q_6 = (math.atan2(-homgen_3_6[2][1], homgen_3_6[2][0])) + math.pi
                s = 1
            # print("DOWN THETA 4 = RAD = ",q_4)
            # print("DOWN THETA 5 = RAD = ",q_5)
            # print("DOWN THETA 6 = RAD = ",q_6)
            if (math.fabs(math.cos(q_6 + q_4))) > thresh:
                cq5 = (((-homgen_3_6[0][0] - homgen_3_6[1][1]) / (math.cos(q_6 + q_4))) - 1)
            # print("CQ5=========",cq5)

            if (math.fabs(math.sin(q_6 + q_4))) > thresh:
                cq5 = (((homgen_3_6[0][1] - homgen_3_6[1][0]) / (math.sin(q_6 + q_4))) - 1)
            # print("CQ5=========",cq5)

            if (math.fabs(math.sin(q_6))) > thresh:
                sq5 = homgen_3_6[2][1] / math.sin(q_6)
            # print("SQ5=========",sq5)

            if (math.fabs(math.cos(q_6))) > thresh:
                sq5 = -homgen_3_6[2][0] / math.cos(q_6)
            # print("SQ5=========",sq5)

            q_5 = math.atan2(sq5, cq5)
        # print("Q5= ", q_5)

        else:
            if s == 1:
                s = 0
                q_4 = 0
                q_5 = 0
                q_6 = math.atan2((homgen_3_6[0][1] - homgen_3_6[1][0]), (- homgen_3_6[0][0] - homgen_3_6[1][1]))
            # print("UP THETA 4 = RAD = ",q_4)
            # print("UP THETA 5 = RAD = ",q_5)
            # print("UP THETA 6 = RAD = ",q_6)

            else:
                q_4 = -math.pi
                q_5 = 0
                q_6 = (math.atan2((homgen_3_6[0][1] - homgen_3_6[1][0]),
                                  (- homgen_3_6[0][0] - homgen_3_6[1][1]))) + math.pi
                s = 1
        # print("DOWN THETA 4 = RAD = ",q_4)
        # print("DOWN THETA 5 = RAD = ",q_5)
        # print("DOWN THETA 6 = RAD = ",q_6)
        q4 = q_4 = math.atan2(math.sin(q_4), math.cos(q_4))
        q5 = q_5 = math.atan2(math.sin(q_5), math.cos(q_5))
        q6 = q_6 = math.atan2(math.sin(q_6), math.cos(q_6))
        q[3][l] = ((q4 * 180) / math.pi)
        q[4][l] = ((q5 * 180) / math.pi)
        q[5][l] = ((q6 * 180) / math.pi)

    # print("Theta  4 =",l,((q[3][l])))
    # print("Theta  5 =",l,((q[4][l])))
    # print("Theta  6 =",l,((q[5][l])))

    # print("theta 1          theta 2          theta 3           theta 4           theta 5")
    for m in range(8):
        print(" ", round(q[0][m], 4), round(q[1][m], 4), round(q[2][m], 4), round(q[3][m], 4), round(q[4][m], 4),
              round(q[5][m], 4))

    sum_jnt = 0
    mn = 0
    q1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rob_dis_jnt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(8):
        for k in range(6):
            sum_jnt += math.fabs(q[k][i] - q1[k])
        rob_dis_jnt[i] = sum_jnt
    mx = rob_dis_jnt[0]
    mn = rob_dis_jnt[0]

    for i in range(8):
        if rob_dis_jnt[i] > mx:
            mx = rob_dis_jnt[i]
            solution_mx = i
        if rob_dis_jnt[i] < mn:
            mn = rob_dis_jnt[i]
            solution_mx = i

    for i in range(8):
        if rob_dis_jnt[i] == mn:
            solution = i
    print("FINAL SOLUTION")
    for k in range(6):
        print(q[k][solution])
        if k == 0:
            j1 = q[k][solution]
        elif k == 1:
            j2 = q[k][solution]
        elif k == 2:
            j3 = q[k][solution]
        elif k == 3:
            j4 = q[k][solution]
        elif k == 4:
            j5 = q[k][solution]
        elif k == 5:
            j6 = q[k][solution]

    return j1, j2, j3, j4, j5, j6


def forward_kin(q1, q2, q3, q4, q5, q6):
    X = []
    Y = []
    Z = []

    X.append(0)
    Y.append(0)
    Z.append(0)

    a1 = 0.1012  # Length of link 1
    a2 = 0.362  # Length of link 2
    a3 = 0.04621  # Length of link 3
    a4 = 0.0
    a5 = 0.0
    a6 = 0.0
    a7 = 0.0
    # Initialize values for the displacements
    d1 = 0.3975  # Displacement of link 1
    d2 = 0.0  # Displacement of link 2
    d3 = 0.0  # Displacement of link 3
    d4 = 0.33254
    d5 = 0.0
    d6 = 0.04602
    d7 = 0.0

    th1 = (q1 + 0.0)
    th2 = (q2 - 90)
    th3 = (q3 + 0.0)
    th4 = (q4 + 0.0)
    th5 = (q5 + 0.0)
    th6 = (q6 + 180)
    th7 = 0.0

    alp1 = -90
    alp2 = 0.0
    alp3 = -90
    alp4 = 90
    alp5 = -90
    alp6 = 0.0
    alp7 = 0.0

    d_h_table = np.array([[np.deg2rad(th1), np.deg2rad(alp1), a1, d1],
                          [np.deg2rad(th2), np.deg2rad(alp2), a2, d2],
                          [np.deg2rad(th3), np.deg2rad(alp3), a3, d3],
                          [np.deg2rad(th4), np.deg2rad(alp4), a4, d4],
                          [np.deg2rad(th5), np.deg2rad(alp5), a5, d5],
                          [np.deg2rad(th6), np.deg2rad(alp6), a6, d6],
                          [np.deg2rad(th7), np.deg2rad(alp7), a7, d7]])
    # Homogeneous transformation matrix from frame 0 to frame 1
    i = 0
    homgen_0_1 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])

    # Homogeneous transformation matrix from frame 1 to frame 2
    i = 1
    homgen_1_2 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])

    # Homogeneous transformation matrix from frame 2 to frame 3
    i = 2
    homgen_2_3 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])

    # Homogeneous transformation matrix from frame 3 to frame 4
    i = 3
    homgen_3_4 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])

    # Homogeneous transformation matrix from frame 4 to frame 5
    i = 4
    homgen_4_5 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])

    # Homogeneous transformation matrix from frame 5 to frame 6
    i = 5
    homgen_5_6 = np.array([[np.cos(d_h_table[i, 0]), -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.cos(d_h_table[i, 0])],
                           [np.sin(d_h_table[i, 0]), np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                            -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                            d_h_table[i, 2] * np.sin(d_h_table[i, 0])],
                           [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
                           [0, 0, 0, 1]])
    homgen_0_2 = homgen_0_1 @ homgen_1_2
    homgen_0_3 = homgen_0_1 @ homgen_1_2 @ homgen_2_3
    homgen_0_4 = homgen_0_1 @ homgen_1_2 @ homgen_2_3 @ homgen_3_4
    homgen_0_5 = homgen_0_1 @ homgen_1_2 @ homgen_2_3 @ homgen_3_4 @ homgen_4_5
    homgen_0_6 = homgen_0_1 @ homgen_1_2 @ homgen_2_3 @ homgen_3_4 @ homgen_4_5 @ homgen_5_6

    # Print the homogeneous transformation matrices
    # Print the homogeneous transformation matrices
    # print("Homogeneous Matrix Frame 0 to Frame 1:")
    # print(homgen_0_1)
    # print()
    px, py, pz = homgen_0_1[0, 3], homgen_0_1[1, 3], homgen_0_1[2, 3]
    X.append(px)
    Y.append(py)
    Z.append(pz)
    # print("Homogeneous Matrix Frame 0 to Frame 2:")
    # print(homgen_0_2)
    # print()
    px, py, pz = homgen_0_2[0, 3], homgen_0_2[1, 3], homgen_0_2[2, 3]
    X.append(px)
    Y.append(py)
    Z.append(pz)
    # print("Homogeneous Matrix Frame 0 to Frame 3:")
    # print(homgen_0_3)
    # print()
    px, py, pz = homgen_0_3[0, 3], homgen_0_3[1, 3], homgen_0_3[2, 3]
    X.append(px)
    Y.append(py)
    Z.append(pz)
    # print("Homogeneous Matrix Frame 0 to Frame 4:")
    # print(homgen_0_4)
    # print()
    px, py, pz = homgen_0_4[0, 3], homgen_0_4[1, 3], homgen_0_4[2, 3]
    X.append(px)
    Y.append(py)
    Z.append(pz)
    # print("Homogeneous Matrix Frame 0 to Frame 5:")
    # print(homgen_0_5)
    px, py, pz = homgen_0_5[0, 3], homgen_0_5[1, 3], homgen_0_5[2, 3]
    X.append(px)
    Y.append(py)
    Z.append(pz)
    # print("Homogeneous Matrix Frame 0 to Frame 6:")
    # print(homgen_0_6)
    px, py, pz = homgen_0_6[0, 3], homgen_0_6[1, 3], homgen_0_6[2, 3]
    X.append(px)
    Y.append(py)
    Z.append(pz)
    return X, Y, Z


def create_plot():
    fig = plt.figure()
    fig.set_size_inches(15, 15)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.set_xlim3d([-4, 4])
    ax.set_ylim3d([-4, 4])
    ax.set_zlim3d([0, 4])
    ax.set_autoscale_on(True)
    return fig, ax


def update_plot(X, Y, Z, fig, ax):
    X = np.reshape(X, (1, 7))
    Y = np.reshape(Y, (1, 7))
    Z = np.reshape(Z, (1, 7))
    ax.cla()
    ax.plot_wireframe(X, Y, Z, linewidth=10)
    plt.draw()
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.set_xlim3d([-2.2, 2.5])
    ax.set_ylim3d([-2.2, 2.5])
    ax.set_zlim3d([0, 2.5])
    ax.set_autoscale_on(False)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()


def main():
    Px1, Py1, Pz1 = 0.49792, 1.3673, 3
    # px, py, pz = (0.78+1.142+0.2), 0.0, (0.32+1.075+0.2)
    roll, pitch, yaw = 0, 0, 0
    # roll, pitch, yaw = 0.366, -0.078, 2.561
    q1, q2, q3, q4, q5, q6 = get_angles(Px1, Py1, Pz1, roll, pitch, yaw)
    print("q1 : ", q1)
    print("q2 : ", q2)
    print("q3 : ", q3)
    print("q4 : ", q4)
    print("q5 : ", q5)
    print("q6 : ", q6)
    create_plot()


if __name__