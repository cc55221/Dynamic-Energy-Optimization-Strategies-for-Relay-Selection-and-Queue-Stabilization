"""
Evaluate the impact of V_1 on our algorithm.

Obtain the relay decision result by adjusting V_1.

      Author: Che Chen (cc5551@foxmail.com)
"""
import math
import random
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

# Color
color_list = ['#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF',
              '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF',
              '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00',
              '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C',
              '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9',
              '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00',
              '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B',
              '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF']
# Mark
line_mark = ['.', ',', 'o', 'v', '^', '<', '>',
             '1', '2', '3', '4', 's', 'p', '*',
             'h', 'H', '+', 'x', 'D', 'd', '|', '_']
# Style
line_style = ['-', '--',
              '-.', ':']
# ==========================================================================
# Remark
# Local computing model
# E_user[t] = kapa * (C_0) ** 2 * m * A[t]

# RN computing model
# tau_1 = w * np.log2(1 + (p_rn_tx[t] * (h) ** 2) / ((1 + d) * sigma ** 2))
# E_rn_tran[t] = p_rn_tx[t] * A_[t] / tau_1
# D_rn_cal[t] = tau * C_rn_cal[t]
# E_rn_cal[t] = kapa * (C_rn_cal[t]) ** 3 * tau
# E_rn[t] = E_rn_tran[t] + E_rn_cal[t]

# RS computing model
# tau = w * np.log2(1 + (p_rs_tx[t] * (h) ** 2) / ((1 + d) * sigma ** 2))
# E_rs_tran[t] = p_rs_tx[t] * A_[t] / tau
# D_rs_cal[t] = tau * C_rs_cal[t]
# E_rs_cal[t] = kapa * (C_rs_cal[t]) ** 3 * tau
# E_rs[t] = E_rs_tran[t] + E_rs_cal[t]

# Queue model
# Q_rn[t + 1] = max(Q_rn[t] - D_rn_cal[t] - D_rn_to_rs[t], 0) + A[t]
# Q_rs[t + 1] = min(max(Q_rs[t] - D_rs_cal[t], 0), D_rn_to_rs[t])

# Performance index
# E_total[t] = E_user[t] + E_rn[t] + E_rs[t]
# Q_total[t] = Q_rn[t] + Q_rs[t]

# Parameters required by Lyapunov
# C = (1/2) ((D_rn_cal_max - D_rn_to_rs_max) ** 2 + D_rs_cal_max ** 2 + D_rn_to_rs_max ** 2 + A_max ** 2)
# Question P_2
# P_2[t] = -(D_rn_cal[t] + D_rn_to_rs[t]) * Q_rn[t] - (D_rs_cal[t] - D_rn_to_rs[t]) * Q_rs[t] + V * E_total[t]
# Optimal RN transmission power p_rs_tx[t]
# TP_1[t] = -(Q_rn[t] = Q_rs[t]) * D_rn_to_rs[t] + V * E_rs_tx[t]
# Optimal RN calculation frequency
# TP_2[t] = -Q_rs[t] * D_rs_cal[t] + V * E_rn_cal[t]
# Optimal RS calculation frequency
# TP_3[t] = -Q_rs[t] * D_rn_cal[t] + V * E_rs[t]

T = 550       # Timeline
# V = 0.9
kappa_user = 1e-7   # 10 ** (-7)
kapa = 1e-8   # 10 ** (-8)
tau = 0.02    # ms
m = 100
g_user = 1000  # The period required for the user to calculate 1bit
g_MEC = 100    # The period required for MEC server to calculate 1bit
w = 1
sigma = 10 ** (-13)
d = 500
h = 0.0005
C_0 = 1000
p_0 = 50
p_rn_tx_max = 100
C_rn_cal_max = 10000
C_rs_cal_max = 10000

h_k_min, h_k_max = 0.0002, 0.008   # modify
d_k_min, d_k_max = 50, 2000        # modify

A_min, A_max = 1.4, 1.6
num_RN = 5
num_user = 5
use_ratio = 0.8
V = 1
average = 50


# Given range [min_value, max_value], generate num_server random integers
def ra_sample(min_value, max_value, num_server):
    begin = min_value
    end = max_value
    need_count = num_server
    sample_list = random.sample(range(begin, end), need_count)
    return sample_list


# Given range [min_value, max_value], generate num_server random decimals
def ra_sample_float(min_value, max_value, num_server):
    sample_list = []
    for i in range(num_server):
        rad_num = random.uniform(min_value, max_value)
        sample_list.append(round(rad_num, 5))
    return sample_list


def selection(mode_x):
    A_1 = []
    for i in range(0, num_user):
        A_1.append([])
    A_2 = []
    for i in range(0, num_RN):
        A_2.append([])

    Q_rn = []
    Q_rs = []
    E_total = []
    Q_total = []
    for j in range(0, num_RN):
        Q_rn.append([5])
        Q_rs.append([0])
        E_total.append([])
        Q_total.append([])

    h_k_list = []
    d_k_list = []
    # h_k_list = ra_sample_float(h_k_min, h_k_max, num_RN)
    # d_k_list = ra_sample(d_k_min, d_k_max, num_RN)
    for i in range(0, num_user):
        h_k_list.append(ra_sample_float(h_k_min, h_k_max, num_RN))
        d_k_list.append(ra_sample(d_k_min, d_k_max, num_RN))

    h_d_k_list = ra_sample_float(h_k_min, h_k_max, num_RN)
    d_d_k_list = ra_sample(d_k_min, d_k_max, num_RN)

    E_channel = []
    for i in range(0, num_user):
        E_channel.append([])

    for t in range(0, T):
        # V = 20
        # MEC_can_use_list = ra_sample_float(0, 1, 2)
        # E_channel = []
        E_judge = []
        E_judge_1 = []
        E_judge_2 = []
        for i in range(0, num_user):
            # E_channel.append([])
            E_judge.append([])
            E_judge_1.append([])
            E_judge_2.append([])
        E_channel_d = []
        for j in range(0, num_RN):
            A_2[j].append(0)

# =============================================================================================

        for i in range(0, num_user):
            # A_1[i].append(random.uniform(A_min, A_max))  # User i generates random tasks
            A_1[i].append(1.5)  # User i generates random tasks
            # Stage 1
            # User i selects a relay and updates A_2 according to A_1
            for j in range(0, num_RN):
                E_user_tran = p_0 * A_1[i][t] / (w * tau * np.log2(1 + (p_0 * (h_k_list[i][j]) ** 2) / ((1 + d_k_list[i][j]) * sigma ** 2)))

                E_judge_1[i].append(10 * E_user_tran + 1 * Q_rn[j][t])
                E_judge_2[i].append(30 * E_user_tran + 1 * Q_rn[j][t])
                E_judge[i].append(100 * E_user_tran + 1 * Q_rn[j][t])
                # E_channel[i].append(E_user_tran)
            # opt_select_E = min(E_channel[i])
            # num_RN_optimal = E_channel[i].index(opt_select_E)    # Optimal energy consumption selection
            if mode_x == 1:
                # num_RN_optimal = random.randint(0, num_RN - 1)          # Random selection
                # print(num_RN_optimal)
                opt_select_judge = min(E_judge_1[i])
                num_RN_optimal = E_judge_1[i].index(opt_select_judge)     # queue only
            elif mode_x == 2:
                # opt_select_D = min(d_k_list[i])
                # num_RN_optimal = d_k_list[i].index(opt_select_D)        # Shortest path selection
                # print(num_RN_optimal)
                opt_select_judge = min(E_judge_2[i])
                num_RN_optimal = E_judge_2[i].index(opt_select_judge)     # E_user only
            else:
                opt_select_judge = min(E_judge[i])
                num_RN_optimal = E_judge[i].index(opt_select_judge)     # Energy consumption and queue weight selection
                # print(num_RN_optimal)
            E_channel[i].append(p_0 * A_1[i][t] / (w * tau * np.log2(
                1 + (p_0 * (h_k_list[i][num_RN_optimal]) ** 2) / ((1 + d_k_list[i][num_RN_optimal]) * sigma ** 2))))
            # print(E_channel)

            # Task A_2 arrives randomly, update RN
            for j in range(0, num_RN):
                if j == num_RN_optimal:
                    # If there is a user choice, then A[t] enters a valid value
                    A_2[j][t] = A_2[j][t] + A_1[i][t]
                else:
                    # If there is no user choice, A[t] enters 0
                    A_2[j][t] = A_2[j][t]  # i.i.d  random.uniform()

# =============================================================================================

        # Stage 2
        # Traverse RN according to A_2 and update the queue of RN_j in the list
        for j in range(0, num_RN):
            # Obtain the optimal policy
            # Obtain the optimal value of the calculated frequency and transmission power
            p_rn_tx_1 = w * (Q_rn[j][t] - Q_rs[j][t]) / (V * np.log(2)) - (1 + d_d_k_list[j]) * sigma ** 2 / (h_d_k_list[j] ** 2)
            p_rn_tx = max(min(p_rn_tx_1, p_rn_tx_max), 0)
            C_rn_cal_1 = (Q_rn[j][t] / (3 * V * kapa * g_MEC)) ** 0.5
            C_rn_cal = min(C_rn_cal_1, C_rn_cal_max)
            C_rs_cal_1 = (Q_rs[j][t] / (3 * V * kapa * g_MEC)) ** 0.5
            C_rs_cal = min(C_rs_cal_1, C_rs_cal_max)

            # According to the optimal value, calculate the energy consumption and delay
            E_rn_tran = p_rn_tx * tau
            E_rn_cal = kapa * (C_rn_cal) ** 3 * tau
            E_rs_cal = kapa * (C_rs_cal) ** 3 * tau
            E_total[j].append(E_rn_cal + E_rn_tran + E_rs_cal)

            # Update the queue and move on to the next moment
            D_rn_cal = tau * C_rn_cal / g_MEC
            D_rs_cal = tau * C_rs_cal / g_MEC
            D_rn_to_rs = w * tau * np.log2(1 + (p_rn_tx * (h_d_k_list[j]) ** 2) / ((1 + d_d_k_list[j]) * sigma ** 2))

            Q_rn[j].append(max(Q_rn[j][t] - D_rn_cal - D_rn_to_rs, 0) + A_2[j][t])
            Q_rs[j].append(max(Q_rs[j][t] - D_rs_cal, 0) + D_rn_to_rs)
            Q_total[j].append(Q_rn[j][t] + Q_rs[j][t])

    Q_temp_t = []
    for y in range(0, T):
        x_0 = 0
        for x in range(0, num_RN):
            x_0 = x_0 + Q_total[x][y]
        Q_temp_t.append(x_0)
    Q = Q_temp_t

    E_local_temp_t = []
    for y in range(0, T):
        x_0 = 0
        for x in range(0, num_user):
            x_0 = x_0 + E_channel[x][y]
        E_local_temp_t.append(x_0/num_user)
    E_local = E_local_temp_t

    E_RN_temp_t = []
    for y in range(0, T):
        x_0 = 0
        for x in range(0, num_RN):
            x_0 = x_0 + E_total[x][y]
        E_RN_temp_t.append(x_0)
    E_RN = E_RN_temp_t

    E_1 = np.array(E_local)
    E_2 = np.array(E_RN)
    E = list(E_1 + E_2)

    # E = list(E_local + E_RN)
    return (E_local)


def selection_average(mode_x):
    v_1_matrix = []
    for average_time in range(0, average):
        v_1_sequence = selection(mode_x)
        v_1_matrix.append(v_1_sequence)
        print(average_time)
    return v_1_matrix


v_1 = list(np.mean(selection_average(1), axis=0))
value_stage_Q = OrderedDict()
for i in range(100, T, 100):
    value_stage_Q[str(i)] = v_1[i]
d_time_Q = np.array([int(x) for x in value_stage_Q.keys()])
e_consu_Q = value_stage_Q.values()

v_2 = list(np.mean(selection_average(2), axis=0))
value_stage_E = OrderedDict()
for i in range(100, T, 100):
    value_stage_E[str(i)] = v_2[i]
d_time_E = np.array([int(x) for x in value_stage_E.keys()])
e_consu_E = value_stage_E.values()

v_3 = list(np.mean(selection_average(3), axis=0))
value_stage_D = OrderedDict()
for i in range(100, T, 100):
    value_stage_D[str(i)] = v_3[i]
d_time_D = np.array([int(x) for x in value_stage_D.keys()])
e_consu_D = value_stage_D.values()

# plt init
plt.figure(figsize=(5, 3))
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel('时间/(s)')
# plt.ylabel("总缓冲队列/(kbits)")
plt.ylabel('用户总能耗/(mW)')
# plt.xlim([2, 20])
# 设置坐标轴刻度
my_x_ticks = np.arange(0, T, 50)
plt.xticks(my_x_ticks)
plt.xlim(0, 500)
plt.plot(d_time_Q, e_consu_Q, color='black', alpha=1.0,
          linestyle='--',
         label=r'$V_{1}$=10 $bits · W^{-1}$', markersize=5, marker=line_mark[4], linewidth=1, clip_on = False)  # marker=line_mark[12],
plt.plot(d_time_E, e_consu_E, color='black', alpha=1.0,
          linestyle='--',
         label=r'$V_{1}$=30 $bits · W^{-1}$', markersize=5, marker=line_mark[2], linewidth=1, clip_on = False)  # marker=line_mark[12],
plt.plot(d_time_D, e_consu_D, color='black', alpha=1.0,
          linestyle='--',
         label=r'$V_{1}$=100 $bits · W^{-1}$', markersize=7, marker=line_mark[13], linewidth=1, clip_on = False)
plt.legend(fontsize=8)
# plt.grid(linestyle='-.')     # Add grid
plt.savefig("Figure_result_1.png", dpi=500, bbox_inches='tight')    # Solve the problem of unclear and incomplete pictures
plt.show()

# width = 20
# # plt.figure(figsize=(5, 3))
# fig, ax = plt.subplots()
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.bar(d_time_Q, e_consu_Q, width, # edgecolor='black',
#         color='white', alpha=1.0, hatch='///', label=r'$V_{1}$=10 $bits · W^{-1}$', edgecolor='black')
# plt.bar(d_time_E + width, e_consu_E, width, # edgecolor='black',
#         color='white', alpha=1.0, hatch='...', label=r'$V_{1}$=30 $bits · W^{-1}$', edgecolor='black')
# plt.bar(d_time_D + width + width, e_consu_D, width, # edgecolor='black',
#         color='white', alpha=1.00, hatch='---', label=r'$V_{1}$=100 $bits · W^{-1}$', edgecolor='black')
# ax.set_xticks(d_time_E + width)      # 将坐标设置在指定位置
# ax.set_xticklabels(d_time_D)           # 将横坐标替换成
# plt.ylabel('用户平均能耗/(mW)')
# plt.xlabel("时间/(s)")
# plt.ylim(54.7, 55.3)
# plt.legend(fontsize=9)
# plt.grid(linestyle='-.')     # Add grid
# plt.savefig("Figure_result_1.png", dpi=500, bbox_inches='tight')    # Solve the problem of unclear and incomplete pictures
# plt.show()
