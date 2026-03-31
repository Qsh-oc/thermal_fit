#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电机热路参数辨识 - 改进版（3 节点模型）
考虑机壳温度动态变化

模型结构:
T_amb ── R_amb ── T_case ── R_1 ── T_1 ── R_2 ── T_coil
                     │           │           │
                    C_case      C_1         C_2
                     │           │           │
                    GND         GND         GND
                                  │
                                J_loss

待辨识参数：R_amb, R_1, R_2, C_case, C_1, C_2 (6 个参数)
"""

import numpy as np
import scipy.io
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("电机热路参数辨识 - 3 节点改进模型")
print("考虑机壳温度动态变化")
print("="*70)

# ============ 1. 加载数据 ============
mat_path = Path(__file__).parent / 'temp_data.mat'
print(f"\n加载数据：{mat_path}")
data = scipy.io.loadmat(str(mat_path))

def extract_data(con, J_loss):
    """提取数据并降采样"""
    if con.ndim == 2 and con.shape == (1, 1):
        con = con[0, 0]
    
    time = con['time'].flatten()
    case = con['case'].flatten()
    coilF = con['coilF'].flatten()
    coilB = con['coilB'].flatten()
    coilM = con['coilM'].flatten()
    T_coil = (coilF + coilB + coilM) / 3
    
    # 降采样：每 20 个点取 1 个（保留更多动态信息）
    step = 20
    return {
        'time': time[::step],
        'T_case': case[::step],
        'T_coil': T_coil[::step],
        'J_loss': J_loss,
        'T_amb': 14  # 环境温度
    }

data1 = extract_data(data['con1'], 652)
data2 = extract_data(data['con2'], 452)

print(f"工况 1: {len(data1['time'])} 个数据点，J_loss = {data1['J_loss']}W")
print(f"工况 2: {len(data2['time'])} 个数据点，J_loss = {data2['J_loss']}W")

# ============ 2. 稳态估算 ============
print("\n" + "="*70)
print("稳态参数估算")
print("="*70)

idx1 = slice(int(len(data1['time']) * 0.9), None)
idx2 = slice(int(len(data2['time']) * 0.9), None)

T_coil_ss1 = np.mean(data1['T_coil'][idx1])
T_case_ss1 = np.mean(data1['T_case'][idx1])
T_coil_ss2 = np.mean(data2['T_coil'][idx2])
T_case_ss2 = np.mean(data2['T_case'][idx2])

# 总热阻（从损耗到线圈）
R_total1 = (T_coil_ss1 - T_case_ss1) / data1['J_loss']
R_total2 = (T_coil_ss2 - T_case_ss2) / data2['J_loss']
R_total_coil = (R_total1 + R_total2) / 2

# 机壳到环境的热阻
R_case_amb1 = (T_case_ss1 - data1['T_amb']) / data1['J_loss']
R_case_amb2 = (T_case_ss2 - data2['T_amb']) / data2['J_loss']
R_case_amb = (R_case_amb1 + R_case_amb2) / 2

print(f"工况 1 稳态:")
print(f"  T_coil = {T_coil_ss1:.2f}°C, T_case = {T_case_ss1:.2f}°C, T_amb = {data1['T_amb']}°C")
print(f"  ΔT_coil_case = {T_coil_ss1 - T_case_ss1:.2f}°C, ΔT_case_amb = {T_case_ss1 - data1['T_amb']:.2f}°C")
print(f"工况 2 稳态:")
print(f"  T_coil = {T_coil_ss2:.2f}°C, T_case = {T_case_ss2:.2f}°C, T_amb = {data2['T_amb']}°C")
print(f"  ΔT_coil_case = {T_coil_ss2 - T_case_ss2:.2f}°C, ΔT_case_amb = {T_case_ss2 - data2['T_amb']:.2f}°C")
print(f"\n估算热阻:")
print(f"  R_coil_case (线圈→机壳) = {R_total_coil:.6f} °C/W")
print(f"  R_case_amb  (机壳→环境)  = {R_case_amb:.6f} °C/W")

# ============ 3. 3 节点热路模型 ============

def thermal_ode_3node(t, x, params, T_amb, J_loss):
    """
    3 节点热路 ODE 方程
    
    状态：x = [T_case, T_1, T_coil]
    参数：params = [R_amb, R_1, R_2, C_case, C_1, C_2]
    
    热平衡方程:
    C_case * dT_case/dt = (T_amb - T_case)/R_amb + (T_1 - T_case)/R_1
    C_1 * dT_1/dt = (T_case - T_1)/R_1 + (T_coil - T_1)/R_2
    C_2 * dT_coil/dt = (T_1 - T_coil)/R_2 + J_loss
    """
    R_amb, R_1, R_2, C_case, C_1, C_2 = params
    
    T_case, T_1, T_coil = x
    
    dT_case = ((T_amb - T_case) / R_amb + (T_1 - T_case) / R_1) / C_case
    dT_1 = ((T_case - T_1) / R_1 + (T_coil - T_1) / R_2) / C_1
    dT_coil = ((T_1 - T_coil) / R_2 + J_loss) / C_2
    
    return [dT_case, dT_1, dT_coil]


def simulate_3node(params, t_exp, T_amb, J_loss, x0):
    """仿真 3 节点模型"""
    
    def ode_func(t, x):
        return thermal_ode_3node(t, x, params, T_amb, J_loss)
    
    sol = solve_ivp(ode_func, [t_exp[0], t_exp[-1]], x0, 
                    t_eval=t_exp, method='RK45', rtol=1e-6, atol=1e-6)
    
    if not sol.success:
        raise ValueError(f"ODE 求解失败：{sol.message}")
    
    return sol.t, sol.y.T


def objective_3node(params):
    """目标函数：最小化两个工况的预测误差"""
    # 初始状态：[T_case, T_1, T_coil] = [T_amb, T_amb, T_amb]
    x0_1 = [data1['T_amb'], data1['T_amb'], data1['T_amb']]
    x0_2 = [data2['T_amb'], data2['T_amb'], data2['T_amb']]
    
    try:
        _, T_sim1 = simulate_3node(params, data1['time'], data1['T_amb'], data1['J_loss'], x0_1)
        _, T_sim2 = simulate_3node(params, data2['time'], data2['T_amb'], data2['J_loss'], x0_2)
        
        # T_sim[:, 0] = T_case, T_sim[:, 1] = T_1, T_sim[:, 2] = T_coil
        T_case_sim1 = T_sim1[:, 0]
        T_coil_sim1 = T_sim1[:, 2]
        T_case_sim2 = T_sim2[:, 0]
        T_coil_sim2 = T_sim2[:, 2]
        
        # 同时拟合机壳温度和线圈温度
        error_case1 = T_case_sim1 - data1['T_case']
        error_coil1 = T_coil_sim1 - data1['T_coil']
        error_case2 = T_case_sim2 - data2['T_case']
        error_coil2 = T_coil_sim2 - data2['T_coil']
        
        # 加权误差平方和（线圈温度权重更高）
        obj = (np.sum(error_case1**2) + np.sum(error_case2**2)) * 0.3 + \
              (np.sum(error_coil1**2) + np.sum(error_coil2**2)) * 0.7
        
        return obj
    except Exception as e:
        return 1e10


# ============ 4. 参数辨识 ============
print("\n" + "="*70)
print("参数辨识（差分进化全局优化）")
print("="*70)

# 参数边界
# [R_amb, R_1, R_2, C_case, C_1, C_2]
bounds = [
    (0.01, 0.5),     # R_amb: 机壳→环境热阻
    (0.001, 0.2),    # R_1: 机壳→中间节点
    (0.001, 0.2),    # R_2: 中间节点→线圈
    (100, 5000),     # C_case: 机壳热容
    (50, 2000),      # C_1: 中间节点热容
    (50, 2000),      # C_2: 线圈热容
]

print("参数范围:")
print(f"  R_amb  (机壳→环境):  0.01 - 0.5 °C/W")
print(f"  R_1    (机壳→中间):  0.001 - 0.2 °C/W")
print(f"  R_2    (中间→线圈):  0.001 - 0.2 °C/W")
print(f"  C_case (机壳热容):   100 - 5000 J/°C")
print(f"  C_1    (中间热容):   50 - 2000 J/°C")
print(f"  C_2    (线圈热容):   50 - 2000 J/°C")
print("\n开始全局优化（可能需要 5-10 分钟）...")

result = differential_evolution(
    objective_3node, bounds,
    seed=42,
    maxiter=500,
    tol=1e-7,
    workers=1,
    updating='deferred',
    polish=True,
    disp=True,
    popsize=25
)

R_amb, R_1, R_2, C_case, C_1, C_2 = result.x

print(f"\n{'='*70}")
print("辨识结果")
print(f"{'='*70}")
print(f"R_amb   = {R_amb:.8f} °C/W  (机壳→环境)")
print(f"R_1     = {R_1:.8f} °C/W    (机壳→中间节点)")
print(f"R_2     = {R_2:.8f} °C/W    (中间节点→线圈)")
print(f"C_case  = {C_case:.4f} J/°C  (机壳热容)")
print(f"C_1     = {C_1:.4f} J/°C     (中间节点热容)")
print(f"C_2     = {C_2:.4f} J/°C     (线圈热容)")
print(f"R_total = {R_amb + R_1 + R_2:.8f} °C/W")

# ============ 5. 模型验证 ============
print("\n" + "="*70)
print("模型验证")
print("="*70)

x0_1 = [data1['T_amb'], data1['T_amb'], data1['T_amb']]
x0_2 = [data2['T_amb'], data2['T_amb'], data2['T_amb']]

_, T_sim1 = simulate_3node(result.x, data1['time'], data1['T_amb'], data1['J_loss'], x0_1)
_, T_sim2 = simulate_3node(result.x, data2['time'], data2['T_amb'], data2['J_loss'], x0_2)

# 提取预测值
T_case_sim1 = T_sim1[:, 0]
T_coil_sim1 = T_sim1[:, 2]
T_case_sim2 = T_sim2[:, 0]
T_coil_sim2 = T_sim2[:, 2]

# 计算误差
error_case1 = T_case_sim1 - data1['T_case']
error_coil1 = T_coil_sim1 - data1['T_coil']
error_case2 = T_case_sim2 - data2['T_case']
error_coil2 = T_coil_sim2 - data2['T_coil']

def calc_metrics(error):
    return {
        'RMSE': np.sqrt(np.mean(error**2)),
        'MAE': np.mean(np.abs(error)),
        'MaxError': np.max(np.abs(error))
    }

metrics_case1 = calc_metrics(error_case1)
metrics_coil1 = calc_metrics(error_coil1)
metrics_case2 = calc_metrics(error_case2)
metrics_coil2 = calc_metrics(error_coil2)

print("工况 1 (652W):")
print(f"  机壳温度：RMSE = {metrics_case1['RMSE']:.4f}°C, MAE = {metrics_case1['MAE']:.4f}°C, Max = {metrics_case1['MaxError']:.4f}°C")
print(f"  线圈温度：RMSE = {metrics_coil1['RMSE']:.4f}°C, MAE = {metrics_coil1['MAE']:.4f}°C, Max = {metrics_coil1['MaxError']:.4f}°C")
print("工况 2 (452W):")
print(f"  机壳温度：RMSE = {metrics_case2['RMSE']:.4f}°C, MAE = {metrics_case2['MAE']:.4f}°C, Max = {metrics_case2['MaxError']:.4f}°C")
print(f"  线圈温度：RMSE = {metrics_coil2['RMSE']:.4f}°C, MAE = {metrics_coil2['MAE']:.4f}°C, Max = {metrics_coil2['MaxError']:.4f}°C")

# 精度判定（以线圈温度为准）
max_coil_error = max(metrics_coil1['MaxError'], metrics_coil2['MaxError'])
if max_coil_error <= 5:
    print(f"\n✅ 精度要求满足！线圈最大误差 {max_coil_error:.4f}°C < ±5°C")
    accuracy_pass = True
else:
    print(f"\n⚠️  精度要求未完全满足，线圈最大误差：{max_coil_error:.4f}°C > ±5°C")
    accuracy_pass = False

# ============ 6. 可视化 ============
print("\n" + "="*70)
print("生成图表")
print("="*70)

fig = plt.figure(figsize=(16, 12))
fig.suptitle('电机热路参数辨识结果（3 节点模型）- 绕组温度预测精度验证', fontsize=16, fontweight='bold')

# 工况 1: 机壳温度对比
ax = plt.subplot(3, 2, 1)
ax.plot(data1['time'], data1['T_case'], 'bo', markersize=3, alpha=0.5, label='实验值')
ax.plot(data1['time'], T_case_sim1, 'r-', linewidth=2, label='预测值')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 1 (J_loss = {data1['J_loss']}W) - 机壳温度")
ax.legend()
ax.grid(True, alpha=0.3)

# 工况 1: 线圈温度对比
ax = plt.subplot(3, 2, 2)
ax.plot(data1['time'], data1['T_coil'], 'bo', markersize=3, alpha=0.5, label='实验值')
ax.plot(data1['time'], T_coil_sim1, 'r-', linewidth=2, label='预测值')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 1 (J_loss = {data1['J_loss']}W) - 绕组温度")
ax.legend()
ax.grid(True, alpha=0.3)

# 工况 2: 机壳温度对比
ax = plt.subplot(3, 2, 3)
ax.plot(data2['time'], data2['T_case'], 'bo', markersize=3, alpha=0.5, label='实验值')
ax.plot(data2['time'], T_case_sim2, 'r-', linewidth=2, label='预测值')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 2 (J_loss = {data2['J_loss']}W) - 机壳温度")
ax.legend()
ax.grid(True, alpha=0.3)

# 工况 2: 线圈温度对比
ax = plt.subplot(3, 2, 4)
ax.plot(data2['time'], data2['T_coil'], 'bo', markersize=3, alpha=0.5, label='实验值')
ax.plot(data2['time'], T_coil_sim2, 'r-', linewidth=2, label='预测值')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 2 (J_loss = {data2['J_loss']}W) - 绕组温度")
ax.legend()
ax.grid(True, alpha=0.3)

# 工况 1: 误差曲线
ax = plt.subplot(3, 2, 5)
ax.plot(data1['time'], error_coil1, 'g-', linewidth=1.5, label='线圈误差')
ax.plot(data1['time'], error_case1, 'b--', linewidth=1.5, label='机壳误差', alpha=0.7)
ax.axhline(y=5, color='r', linestyle='--', linewidth=2, label='+5°C 边界')
ax.axhline(y=-5, color='r', linestyle='--', linewidth=2)
ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
ax.fill_between(data1['time'], -5, 5, alpha=0.2, color='green')
ax.set_xlabel('时间 (s)')
ax.set_ylabel('预测误差 (°C)')
ax.set_title(f"工况 1 误差 (线圈 RMSE={metrics_coil1['RMSE']:.3f}°C, Max={metrics_coil1['MaxError']:.3f}°C)")
ax.legend()
ax.grid(True, alpha=0.3)

# 工况 2: 误差曲线
ax = plt.subplot(3, 2, 6)
ax.plot(data2['time'], error_coil2, 'g-', linewidth=1.5, label='线圈误差')
ax.plot(data2['time'], error_case2, 'b--', linewidth=1.5, label='机壳误差', alpha=0.7)
ax.axhline(y=5, color='r', linestyle='--', linewidth=2)
ax.axhline(y=-5, color='r', linestyle='--', linewidth=2)
ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
ax.fill_between(data2['time'], -5, 5, alpha=0.2, color='green')
ax.set_xlabel('时间 (s)')
ax.set_ylabel('预测误差 (°C)')
ax.set_title(f"工况 2 误差 (线圈 RMSE={metrics_coil2['RMSE']:.3f}°C, Max={metrics_coil2['MaxError']:.3f}°C)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
result_path = Path(__file__).parent / 'parameter_identification_3node.png'
plt.savefig(result_path, dpi=150, bbox_inches='tight')
print(f"结果图已保存：{result_path}")
plt.close()

# ============ 7. 保存结果 ============
results = {
    'model': '3-node',
    'parameters': {
        'R_amb': float(R_amb),
        'R_1': float(R_1),
        'R_2': float(R_2),
        'C_case': float(C_case),
        'C_1': float(C_1),
        'C_2': float(C_2),
        'R_total': float(R_amb + R_1 + R_2)
    },
    'accuracy': {
        'case': {
            'RMSE1': float(metrics_case1['RMSE']),
            'RMSE2': float(metrics_case2['RMSE']),
            'MaxError1': float(metrics_case1['MaxError']),
            'MaxError2': float(metrics_case2['MaxError'])
        },
        'coil': {
            'RMSE1': float(metrics_coil1['RMSE']),
            'RMSE2': float(metrics_coil2['RMSE']),
            'MAE1': float(metrics_coil1['MAE']),
            'MAE2': float(metrics_coil2['MAE']),
            'MaxError1': float(metrics_coil1['MaxError']),
            'MaxError2': float(metrics_coil2['MaxError'])
        },
        'accuracy_pass': accuracy_pass
    },
    'conditions': {
        'J_loss1': data1['J_loss'],
        'J_loss2': data2['J_loss'],
        'T_amb': 14
    }
}

output_path = Path(__file__).parent / 'identified_parameters_3node.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"参数已保存：{output_path}")

# 打印报告
print("\n" + "━"*70)
print("辨识报告 - 3 节点模型")
print("━"*70)
print("热阻参数:")
print(f"  R_amb   = {R_amb:.8f} °C/W  (机壳→环境)")
print(f"  R_1     = {R_1:.8f} °C/W    (机壳→中间节点)")
print(f"  R_2     = {R_2:.8f} °C/W    (中间节点→线圈)")
print("热容参数:")
print(f"  C_case  = {C_case:.4f} J/°C  (机壳热容)")
print(f"  C_1     = {C_1:.4f} J/°C     (中间节点热容)")
print(f"  C_2     = {C_2:.4f} J/°C     (线圈热容)")
print("━"*70)
print("模型精度（线圈温度）:")
print(f"  工况 1 (652W): RMSE = {metrics_coil1['RMSE']:.4f}°C, 最大误差 = {metrics_coil1['MaxError']:.4f}°C")
print(f"  工况 2 (452W): RMSE = {metrics_coil2['RMSE']:.4f}°C, 最大误差 = {metrics_coil2['MaxError']:.4f}°C")
print("━"*70)
print("稳态温升预测:")
print(f"  652W: ΔT_coil = {(R_1 + R_2) * 652:.2f} °C, ΔT_case = {R_amb * 652:.2f} °C")
print(f"  452W: ΔT_coil = {(R_1 + R_2) * 452:.2f} °C, ΔT_case = {R_amb * 452:.2f} °C")
print("━"*70)
print("\n辨识完成！")
