"""
Since this part only considers the relationship between task buffer constraints
and energy consumption of relay and remote,
and does not involve the user selection stage.
Thus, it is sufficient to extract only one set of RN and RS for analysis.

Obtain the relay decision result by adjusting V_2

      Author: Che Chen (cc5551@foxmail.com)
"""
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

T = 600       # Timeline
# V = 0.9
kapa = 1e-8   # 10 ** (-8)
tau = 0.02
m = 100
m_user = 1000  # The period required for the user to calculate 1bit
m_MEC = 300    # The period required for MEC server to calculate 1bit
w = 1
sigma = 10 ** (-13)
d = 500
h = 0.0005
p_0 = 100
p_rn_tx_max = 100
C_rn_cal_max = 10000
C_rs_cal_max = 10000

A_min, A_max = 2.3, 2.7


def trade_off_v2(V):
    A = []
    Q_rn = [5]
    Q_rs = [0]
    p_rn_tx_1 = []
    p_rn_tx = []
    C_rn_cal_1 = []
    C_rn_cal = []
    C_rs_cal_1 = []
    C_rs_cal = []

    E_rn = []
    E_rs = []
    E_total = []
    Q_total = []

    D_rn_cal = []
    D_rs_cal = []
    D_rn_to_rs = []
    for t in range(0, T):
        # V = 10
        # Step 1
        # Tasks arrive randomly
        A.append(random.uniform(A_min, A_max))  # i.i.d  random.uniform()
        # A.append(1.5)  # i.i.d  random.uniform()

        # Step 2
        # Get the optimal strategy
        # Obtain the optimal value of the calculated frequency and transmission power
        # RN min 1
        p_rn_tx_1.append(w * (Q_rn[t] - Q_rs[t]) / (V * np.log(2)) - (1 + d) * sigma ** 2 / (h ** 2))
        # Obtain the optimal p_rn_tx[t], avoid appearing less than 0
        p_rn_tx.append(round(max(min(p_rn_tx_1[t], p_rn_tx_max), 0), 1))
        # print(p_rn_tx[t])

        C_rn_cal_1.append((Q_rn[t] / (3 * V * kapa * m)) ** 0.5)
        # Obtain the best C_rn_cal[t]
        C_rn_cal.append(round(min(C_rn_cal_1[t], C_rn_cal_max), 1))
        # print('C_rn_cal', C_rn_cal[t])

        C_rs_cal_1.append((Q_rs[t] / (3 * V * kapa * m)) ** 0.5)
        # Obtain the best C_rs_cal[t]
        C_rs_cal.append(round(min(C_rs_cal_1[t], C_rs_cal_max), 1))
        # print(C_rs_cal[t])

        # Step 3
        # According to the optimal value, calculate the energy consumption and delay
        tau_1 = w * np.log2(1 + (p_rn_tx[t] * (h) ** 2) / ((1 + d) * sigma ** 2))
        E_rn_tran = p_rn_tx[t] * tau
        E_rn_cal = kapa * (C_rn_cal[t]) ** 3 * tau
        E_rn.append(E_rn_cal + E_rn_tran)
        # print(E_rn_cal, E_rn_tran, E_rn)

        E_rs_cal = kapa * (C_rs_cal[t]) ** 3 * tau
        E_rs.append(E_rs_cal)
        # print(E_rs_cal, E_rs)

        E_total.append(E_rn[t] + E_rs[t])

        # Step 4
        # Update the queue and move on to the next moment
        D_rn_cal.append(tau * C_rn_cal[t] / m)
        D_rs_cal.append(tau * C_rs_cal[t] / m)
        D_rn_to_rs.append(w * tau * np.log2(1 + (p_rn_tx[t] * (h) ** 2) / ((1 + d) * sigma ** 2)))
        # print(A[t], E_total[t], E_rn[t], E_rs[t], p_rn_tx[t], D_rn_to_rs[t])

        Q_rn.append(max(Q_rn[t] - D_rn_cal[t] - D_rn_to_rs[t], 0) + A[t])
        Q_rs.append(max(Q_rs[t] - D_rs_cal[t], 0) + D_rn_to_rs[t])
        Q_total.append(Q_rn[t] + Q_rs[t])
        # print(D_rn_cal[t], D_rn_to_rs[t], D_rs_cal[t])
        print(p_rn_tx[t], C_rn_cal[t], C_rs_cal[t])
        print(Q_total[t], Q_rn[t], Q_rs[t])
    return E_total


v_1 = trade_off_v2(1)
value_stage_Q = OrderedDict()
for i in range(0, T, 20):
    value_stage_Q[str(i)] = v_1[i]

d_time_Q = np.array([int(x) for x in value_stage_Q.keys()])
e_consu_Q = value_stage_Q.values()

# v_2 = trade_off_v2(30)
# value_stage_E = OrderedDict()
# for i in range(0, T, 2000):
#     value_stage_E[str(i)] = v_2[i]
# d_time_E = np.array([int(x) for x in value_stage_E.keys()])
# e_consu_E = value_stage_E.values()
#
# v_3 = trade_off_v2(50)
# value_stage_D = OrderedDict()
# for i in range(0, T, 2000):
#     value_stage_D[str(i)] = v_3[i]
# d_time_D = np.array([int(x) for x in value_stage_D.keys()])
# e_consu_D = value_stage_D.values()

# plt init
plt.figure(figsize=(5, 3))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('时间/(s)')
# plt.ylabel("一组中继对能耗/(mW)")
plt.ylabel("一组中继对缓冲队列/(kbits)")
# plt.xlim([0, 30000])
# Set the axis scale
my_x_ticks = np.arange(0, T, 20)
plt.xticks(my_x_ticks)
plt.plot(d_time_Q, e_consu_Q, color='red', alpha=1.0,
          linestyle='-',
         label=r'$V_{2}$=10 $bits^2 · W^{-1}$', markersize=5, marker=line_mark[5], linewidth=1, clip_on = False)
# plt.plot(d_time_E, e_consu_E, color='green', alpha=1.0,
#           linestyle='-',
#          label=r'$V_{2}$=30 $bits^2 · W^{-1}$', markersize=5, marker=line_mark[2], linewidth=1, clip_on = False)
# plt.plot(d_time_D, e_consu_D, color='blue', alpha=1.0,
#           linestyle='-',
#          label=r'$V_{2}$=50 $bits^2 · W^{-1}$', markersize=7, marker=line_mark[13], linewidth=1, clip_on = False)
plt.legend(fontsize=8)
# plt.grid(linestyle='-.')     # Add grid
plt.savefig("Figure_result_1.png", dpi=500, bbox_inches='tight')    # Solve the problem of unclear and incomplete pictures
plt.show()
