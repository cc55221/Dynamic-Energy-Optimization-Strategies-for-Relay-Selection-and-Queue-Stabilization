import math
import random
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

# ==========================绘图设置======================================
# 设置线条的颜色
color_list = ['#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF',
              '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF',
              '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00',
              '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C',
              '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9',
              '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00',
              '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B',
              '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF']
# 线条标志
line_mark = ['.', ',', 'o', 'v', '^', '<', '>',
             '1', '2', '3', '4', 's', 'p', '*',
             'h', 'H', '+', 'x', 'D', 'd', '|', '_']
# 线条类型
line_style = ['-', '--',
              '-.', ':']
# ==========================================================================


# 本地计算模型
# E_user[t] = kapa * (C_0) ** 2 * m * A[t]

# RN计算模型
# tau_1 = w * np.log2(1 + (p_rn_tx[t] * (h) ** 2) / ((1 + d) * sigma ** 2))
# E_rn_tran[t] = p_rn_tx[t] * A_[t] / tau_1
# D_rn_cal[t] = tau * C_rn_cal[t]
# E_rn_cal[t] = kapa * (C_rn_cal[t]) ** 3 * tau
# E_rn[t] = E_rn_tran[t] + E_rn_cal[t]

# RS计算模型
# tau = w * np.log2(1 + (p_rs_tx[t] * (h) ** 2) / ((1 + d) * sigma ** 2))
# E_rs_tran[t] = p_rs_tx[t] * A_[t] / tau
# D_rs_cal[t] = tau * C_rs_cal[t]
# E_rs_cal[t] = kapa * (C_rs_cal[t]) ** 3 * tau
# E_rs[t] = E_rs_tran[t] + E_rs_cal[t]

# Q队列模型
# Q_rn[t + 1] = max(Q_rn[t] - D_rn_cal[t] - D_rn_to_rs[t], 0) + A[t]
# Q_rs[t + 1] = min(max(Q_rs[t] - D_rs_cal[t], 0), D_rn_to_rs[t])

# 性能指标
# E_total[t] = E_user[t] + E_rn[t] + E_rs[t]
# Q_total[t] = Q_rn[t] + Q_rs[t]

# Lyapunov所需要的参数
# C = (1/2) ((D_rn_cal_max - D_rn_to_rs_max) ** 2 + D_rs_cal_max ** 2 + D_rn_to_rs_max ** 2 + A_max ** 2)
# 问题P_2
# P_2[t] = -(D_rn_cal[t] + D_rn_to_rs[t]) * Q_rn[t] - (D_rs_cal[t] - D_rn_to_rs[t]) * Q_rs[t] + V * E_total[t]
# 最优 RN 卸载功率 p_rs_tx[t]
# TP_1[t] = -(Q_rn[t] = Q_rs[t]) * D_rn_to_rs[t] + V * E_rs_tx[t]
# 最优 RN 计算频率
# TP_2[t] = -Q_rs[t] * D_rs_cal[t] + V * E_rn_cal[t]
# 最优 RS 计算频率
# TP_3[t] = -Q_rs[t] * D_rn_cal[t] + V * E_rs[t]

def ra_sample(min_value, max_value, num_server):
    begin = min_value
    end = max_value
    need_count = num_server
    sample_list = random.sample(range(begin, end), need_count)
    return sample_list


def ra_sample_float(min_value, max_value, num_server):
    sample_list = []
    for i in range(num_server):
        rad_num = random.uniform(min_value, max_value)
        sample_list.append(round(rad_num, 5))
    return sample_list


T = 2000
# V = 0.9
kappa_user = 1e-7
kapa = 1e-8   # 10 ** (-28)
tau = 0.02
m = 100
g_user = 1000  # 用户计算1bit所需要的周期
g_MEC = 100    # MEC计算1bit所需要的周期
w = 1
sigma = 10 ** (-13)
d = 500
h = 0.0005
C_0 = 1000
p_0 = 100
p_rn_tx_max = 100
C_rn_cal_max = 10000
C_rs_cal_max = 10000

h_k_min, h_k_max = 0.0002, 0.0008
d_k_min, d_k_max = 200, 800

A_min, A_max = 1, 2
num_RN = 7
# num_RN_max = 15
# num_user = 15
num_user_max = 16
use_ratio = 0.8
average = 1
# V = 10


def multi_user(num_user, V):
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

    # Time slot begin
    for t in range(0, T):
        # V = 10
        # MEC_can_use_list = ra_sample_float(0, 1, 2)
        # E_channel = []
        E_judge = []
        for i in range(0, num_user):
            # E_channel.append([])
            E_judge.append([])
        E_channel_d = []
        Selection_matrix = []        # choose matrix
        # a = 0                        # 当不同用户到达同一RN时，用来累加
        for j in range(0, num_RN):
            A_2[j].append(0)

# =============================================================================================

        for i in range(0, num_user):
            A_1[i].append(random.randint(A_min, A_max))  # 用户 i 产生随机任务
            # A_1[i].append(2.5)  # 用户 i 产生随机任务
            # 阶段一：用户 i 的 relay 选择，根据 A_1来更新 A_2
            for j in range(0, num_RN):
                # E_judge = E_user_tran  # + Q_total[j][t]  之后再进行添加队列作为判断条件
                E_user_tran = p_0 * A_1[i][t] / (w * tau * np.log2(1 + (p_0 * (h_k_list[i][j]) ** 2) / ((1 + d_k_list[i][j]) * sigma ** 2)))
                E_judge[i].append(5 * E_user_tran + 1 * Q_rn[j][t])
                # E_channel[i].append(E_user_tran)

            # opt_select_E = min(E_channel[i])
            # num_RN_optimal = E_channel[i].index(opt_select_E)    # 最优能耗选择
            # num_RN_optimal = random.randint(0, 1)                  # 随机选择
            # opt_select_D = min(d_k_list[i])
            # num_RN_optimal = d_k_list[i].index(opt_select_D)       # 最短路径选择
            opt_select_judge = min(E_judge[i])
            num_RN_optimal = E_judge[i].index(opt_select_judge)      # 能耗和队列加权选择
            E_channel[i].append(p_0 * A_1[i][t] / (w * tau * np.log2(1 + (p_0 * (h_k_list[i][num_RN_optimal]) ** 2) / ((1 + d_k_list[i][num_RN_optimal]) * sigma ** 2))))
            # print(E_channel)

            # 更新 RN 到达的随机任务 A_2
            for j in range(0, num_RN):
                # Step 1: 任务随机到达
                if j == num_RN_optimal:
                    A_2[j][t] = A_2[j][t] + A_1[i][t]    # 有用户选择，则 A[t]输入有效值
                else:
                    A_2[j][t] = A_2[j][t]  # i.i.d  random.uniform()    # 没有用户选择，则 A[t]输入 0

# =============================================================================================

        # 阶段二：根据 A_2来遍历 RN，更新列表中 RN_j 的队列
        for j in range(0, num_RN):
            # Step 2: 获取最优策略，得到计算频率和传输功率最优值
            p_rn_tx_1 = w * (Q_rn[j][t] - Q_rs[j][t]) / (V * np.log(2)) - (1 + d_d_k_list[j]) * sigma ** 2 / (h_d_k_list[j] ** 2)
            p_rn_tx = max(min(p_rn_tx_1, p_rn_tx_max), 0)
            C_rn_cal_1 = (Q_rn[j][t] / (3 * V * kapa * g_MEC)) ** 0.5
            C_rn_cal = min(C_rn_cal_1, C_rn_cal_max)
            C_rs_cal_1 = (Q_rs[j][t] / (3 * V * kapa * g_MEC)) ** 0.5
            C_rs_cal = min(C_rs_cal_1, C_rs_cal_max)

            # Step 3: 根据最优值，计算出能耗和延迟
            E_rn_tran = p_rn_tx * tau
            E_rn_cal = kapa * (C_rn_cal) ** 3 * tau
            E_rs_cal = kapa * (C_rs_cal) ** 3 * tau
            E_total[j].append(E_rn_cal + E_rn_tran + E_rs_cal)

            # Step 4： 更新队列，进入下一个时刻
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
        E_local_temp_t.append(x_0)
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

    return (Q)# + Q_total[1])
    # return (E_total[0] + E_total[1] + E_channel[0] + E_channel[1])


def multi_user_user(V):
    multi_user_user_sequence = []
    for num in range(1, num_user_max):
        print(np.mean(multi_user(num, V)))
        multi_user_user_sequence.append(np.mean(multi_user(num, V)))
    return multi_user_user_sequence


def multi_user_average(V):
    multi_user_matrix = []
    for i in range(0, average):
        multi_user_sequence = multi_user_user(V)
        multi_user_matrix.append(multi_user_sequence)
        print(i)
    return multi_user_matrix


v_1 = list(np.mean(multi_user_average(1), axis=0))
value_stage_Q = OrderedDict()
for i in range(2, num_user_max - 1):
    value_stage_Q[str(i)] = v_1[i]
d_time_Q = np.array([int(x) + 1 for x in value_stage_Q.keys()])
e_consu_Q = value_stage_Q.values()

v_2 = list(np.mean(multi_user_average(10), axis=0))
value_stage_E = OrderedDict()
for i in range(2, num_user_max - 1):
    value_stage_E[str(i)] = v_2[i]
d_time_E = np.array([int(x) + 1 for x in value_stage_E.keys()])
e_consu_E = value_stage_E.values()

v_3 = list(np.mean(multi_user_average(50), axis=0))
value_stage_D = OrderedDict()
for i in range(2, num_user_max - 1):
    value_stage_D[str(i)] = v_3[i]
d_time_D = np.array([int(x) + 1 for x in value_stage_D.keys()])
e_consu_D = value_stage_D.values()


# plt init
plt.figure(figsize=(5, 3))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('用户数量/(个)')
# plt.ylabel("总能耗/(mW)")
plt.ylabel("总缓冲队列/(kbits)")
my_x_ticks = np.arange(0, num_user_max, 3)
plt.xticks(my_x_ticks)
plt.xlim(3, 15)
plt.plot(d_time_Q, e_consu_Q, color='black', alpha=1.0,
          linestyle='-',
         label=r'$V_{2}$=1 $bits^2 · W^{-1}$', markersize=5, marker=line_mark[4], linewidth=1, clip_on = False)
plt.plot(d_time_E, e_consu_E, color='black', alpha=1.0,
          linestyle='-',
         label=r'$V_{2}$=10 $bits^2 · W^{-1}$', markersize=5, marker=line_mark[2], linewidth=1, clip_on = False)
plt.plot(d_time_D, e_consu_D, color='black', alpha=1.0,
          linestyle='-',
         label=r'$V_{2}$=50 $bits^2 · W^{-1}$', markersize=7, marker=line_mark[13], linewidth=1, clip_on = False)
plt.legend(fontsize=8)
# plt.grid(linestyle='-.')     # Add grid
plt.savefig("Figure_result_1.png", dpi=500, bbox_inches='tight')
plt.show()
