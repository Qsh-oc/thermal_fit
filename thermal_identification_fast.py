#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电机热路参数辨识 - 快速版本（实测机壳温度边界条件）
使用 L-BFGS-B 局部优化，从稳态估算值开始
"""

import numpy as np
import scipy.io
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("电机热路参数辨识 - 快速版本")
print("使用实测机壳温度作为边界条件")
print("="*70)

# ============ 1. 加载数据 ============
mat_path = Path(__file__).parent / 'temp_data.mat'
print(f"\n加载数据：{mat_path}")
data = scipy.io.loadmat(str(mat_path))

def extract_data(con, J_loss):
    if con.ndim == 2 and con.shape == (1, 1):
        con = con[0, 0]
    
    time = con['time'].flatten()
    case = con['case'].flatten()
    coilF = con['coilF'].flatten()
    coilB = con['coilB'].flatten()
    coilM = con['coilM'].flatten()
    T_coil = (coilF + coilB + coilM) / 3
    
    # 降采样：每 30 个点取 1 个
    step = 30
    return {
        'time': time[::step],
        'T_case': case[::step],
        'T_coil': T_coil[::step],
        'J_loss': J_loss,
    }

data1 = extract_data(data['con1'], 652)
data2 = extract_data(data['con2'], 452)

print(f"工况 1: {len(data1['time'])} 个数据点，J_loss = {data1['J_loss']}W")
print(f"工况 2: {len(data2['time'])} 个数据点，J_loss = {data2['J_loss']}W")

# ============ 2. 稳态估算 ============
print("\n" + "="*70)
print("稳态参数估算")
print("="*70)

idx1 = slice(int(len(data1['time']) * 0.95), None)
idx2 = slice(int(len(data2['time']) * 0.95), None)

T_coil_ss1 = np.mean(data1['T_coil'][idx1])
T_case_ss1 = np.mean(data1['T_case'][idx1])
T_coil_ss2 = np.mean(data2['T_coil'][idx2])
T_case_ss2 = np.mean(data2['T_case'][idx2])

R_total1 = (T_coil_ss1 - T_case_ss1) / data1['J_loss']
R_total2 = (T_coil_ss2 - T_case_ss2) / data2['J_loss']
R_total = (R_total1 + R_total2) / 2

print(f"工况 1 稳态：T_coil = {T_coil_ss1:.2f}°C, T_case = {T_case_ss1:.2f}°C, ΔT = {T_coil_ss1 - T_case_ss1:.2f}°C")
print(f"工况 2 稳态：T_coil = {T_coil_ss2:.2f}°C, T_case = {T_case_ss2:.2f}°C, ΔT = {T_coil_ss2 - T_case_ss2:.2f}°C")
print(f"估算总热阻 R_total = {R_total:.6f} °C/W")

# ============ 3. 热路模型 ============

def thermal_ode(t, x, params, T_case_interp, J_loss):
    R_1, R_2, C_1, C_2 = params
    T_1, T_coil = x
    T_case = T_case_interp(t)
    
    dT_1 = ((T_case - T_1) / R_1 + (T_coil - T_1) / R_2) / C_1
    dT_coil = ((T_1 - T_coil) / R_2 + J_loss) / C_2
    
    return [dT_1, dT_coil]


def simulate(params, t_exp, T_case_exp, J_loss, x0):
    T_case_interp = lambda t: np.interp(t, t_exp, T_case_exp, 
                                         left=T_case_exp[0], 
                                         right=T_case_exp[-1])
    
    def ode_func(t, x):
        return thermal_ode(t, x, params, T_case_interp, J_loss)
    
    sol = solve_ivp(ode_func, [t_exp[0], t_exp[-1]], x0, 
                    t_eval=t_exp, method='RK45', rtol=1e-6, atol=1e-6)
    return sol.t, sol.y.T


def objective(params):
    x0_1 = [data1['T_case'][0], data1['T_coil'][0]]
    x0_2 = [data2['T_case'][0], data2['T_coil'][0]]
    
    try:
        _, T_sim1 = simulate(params, data1['time'], data1['T_case'], data1['J_loss'], x0_1)
        _, T_sim2 = simulate(params, data2['time'], data2['T_case'], data2['J_loss'], x0_2)
        
        error1 = T_sim1[:, 1] - data1['T_coil']
        error2 = T_sim2[:, 1] - data2['T_coil']
        
        return np.sum(error1**2) + np.sum(error2**2)
    except:
        return 1e10


# ============ 4. 参数辨识（局部优化） ============
print("\n" + "="*70)
print("参数辨识（L-BFGS-B 局部优化）")
print("="*70)

# 初始猜测：基于稳态估算
R_1_init = R_total * 0.4
R_2_init = R_total * 0.6
C_1_init = 300
C_2_init = 150
params0 = [R_1_init, R_2_init, C_1_init, C_2_init]

bounds = [
    (0.0001, 0.1),
    (0.0001, 0.1),
    (50, 1000),
    (50, 1000),
]

print(f"初始参数：R_1={R_1_init:.6f}, R_2={R_2_init:.6f}, C_1={C_1_init:.2f}, C_2={C_2_init:.2f}")
print("开始优化...")

result = minimize(objective, params0, method='L-BFGS-B', bounds=bounds,
                 options={'maxiter': 200, 'ftol': 1e-10})

R_1, R_2, C_1, C_2 = result.x

print(f"\n{'='*70}")
print("辨识结果")
print(f"{'='*70}")
print(f"R_1     = {R_1:.8f} °C/W")
print(f"R_2     = {R_2:.8f} °C/W")
print(f"C_1     = {C_1:.4f} J/°C")
print(f"C_2     = {C_2:.4f} J/°C")
print(f"R_total = {R_1 + R_2:.8f} °C/W")

# ============ 5. 模型验证 ============
print("\n" + "="*70)
print("模型验证")
print("="*70)

x0_1 = [data1['T_case'][0], data1['T_coil'][0]]
x0_2 = [data2['T_case'][0], data2['T_coil'][0]]

_, T_sim1 = simulate(result.x, data1['time'], data1['T_case'], data1['J_loss'], x0_1)
_, T_sim2 = simulate(result.x, data2['time'], data2['T_case'], data2['J_loss'], x0_2)

error1 = T_sim1[:, 1] - data1['T_coil']
error2 = T_sim2[:, 1] - data2['T_coil']

RMSE1 = np.sqrt(np.mean(error1**2))
RMSE2 = np.sqrt(np.mean(error2**2))
MAE1 = np.mean(np.abs(error1))
MAE2 = np.mean(np.abs(error2))
MaxError1 = np.max(np.abs(error1))
MaxError2 = np.max(np.abs(error2))

print(f"工况 1: RMSE = {RMSE1:.4f}°C, MAE = {MAE1:.4f}°C, MaxError = {MaxError1:.4f}°C")
print(f"工况 2: RMSE = {RMSE2:.4f}°C, MAE = {MAE2:.4f}°C, MaxError = {MaxError2:.4f}°C")

max_error = max(MaxError1, MaxError2)
if max_error <= 5:
    print(f"\n✅ 精度要求满足！最大误差 {max_error:.4f}°C < ±5°C")
    accuracy_pass = True
else:
    print(f"\n⚠️  精度要求未完全满足，最大误差：{max_error:.4f}°C > ±5°C")
    accuracy_pass = False

# ============ 6. 可视化 ============
print("\n" + "="*70)
print("生成图表")
print("="*70)

fig = plt.figure(figsize=(14, 10))
fig.suptitle('电机热路参数辨识结果 - 实测机壳温度边界条件', fontsize=16, fontweight='bold')

ax = plt.subplot(3, 2, 1)
ax.plot(data1['time'], data1['T_case'], 'b-', linewidth=1.5, label='T_case 实测')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 1 (J_loss = {data1['J_loss']}W) - 机壳温度（实测输入）")
ax.legend()
ax.grid(True, alpha=0.3)

ax = plt.subplot(3, 2, 2)
ax.plot(data1['time'], data1['T_coil'], 'bo', markersize=3, alpha=0.5, label='实验值')
ax.plot(data1['time'], T_sim1[:, 1], 'r-', linewidth=2, label='预测值')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 1 - 绕组温度预测")
ax.legend()
ax.grid(True, alpha=0.3)

ax = plt.subplot(3, 2, 3)
ax.plot(data2['time'], data2['T_case'], 'b-', linewidth=1.5, label='T_case 实测')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 2 (J_loss = {data2['J_loss']}W) - 机壳温度（实测输入）")
ax.legend()
ax.grid(True, alpha=0.3)

ax = plt.subplot(3, 2, 4)
ax.plot(data2['time'], data2['T_coil'], 'bo', markersize=3, alpha=0.5, label='实验值')
ax.plot(data2['time'], T_sim2[:, 1], 'r-', linewidth=2, label='预测值')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 2 - 绕组温度预测")
ax.legend()
ax.grid(True, alpha=0.3)

ax = plt.subplot(3, 2, 5)
ax.plot(data1['time'], error1, 'g-', linewidth=1.5, label='预测误差')
ax.axhline(y=5, color='r', linestyle='--', linewidth=2, label='+5°C 边界')
ax.axhline(y=-5, color='r', linestyle='--', linewidth=2)
ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
ax.fill_between(data1['time'], -5, 5, alpha=0.2, color='green')
ax.set_xlabel('时间 (s)')
ax.set_ylabel('预测误差 (°C)')
ax.set_title(f"工况 1 误差 (RMSE={RMSE1:.3f}°C, Max={MaxError1:.3f}°C)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = plt.subplot(3, 2, 6)
ax.plot(data2['time'], error2, 'g-', linewidth=1.5, label='预测误差')
ax.axhline(y=5, color='r', linestyle='--', linewidth=2)
ax.axhline(y=-5, color='r', linestyle='--', linewidth=2)
ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
ax.fill_between(data2['time'], -5, 5, alpha=0.2, color='green')
ax.set_xlabel('时间 (s)')
ax.set_ylabel('预测误差 (°C)')
ax.set_title(f"工况 2 误差 (RMSE={RMSE2:.3f}°C, Max={MaxError2:.3f}°C)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
result_path = Path(__file__).parent / 'parameter_identification_measured.png'
plt.savefig(result_path, dpi=150, bbox_inches='tight')
print(f"结果图已保存：{result_path}")
plt.close()

# ============ 7. 保存结果 ============
results = {
    'model': '2-node-measured-case',
    'description': '使用实测机壳温度作为边界条件',
    'parameters': {
        'R_1': float(R_1),
        'R_2': float(R_2),
        'C_1': float(C_1),
        'C_2': float(C_2),
        'R_total': float(R_1 + R_2)
    },
    'accuracy': {
        'RMSE1': float(RMSE1),
        'RMSE2': float(RMSE2),
        'MAE1': float(MAE1),
        'MAE2': float(MAE2),
        'MaxError1': float(MaxError1),
        'MaxError2': float(MaxError2),
        'accuracy_pass': accuracy_pass
    },
    'conditions': {
        'J_loss1': data1['J_loss'],
        'J_loss2': data2['J_loss']
    }
}

output_path = Path(__file__).parent / 'identified_parameters_measured.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"参数已保存：{output_path}")

# 打印报告
print("\n" + "━"*70)
print("辨识报告")
print("━"*70)
print(f"R_1     = {R_1:.8f} °C/W")
print(f"R_2     = {R_2:.8f} °C/W")
print(f"C_1     = {C_1:.4f} J/°C")
print(f"C_2     = {C_2:.4f} J/°C")
print("━"*70)
print(f"工况 1 ({data1['J_loss']}W): RMSE = {RMSE1:.4f}°C, 最大误差 = {MaxError1:.4f}°C")
print(f"工况 2 ({data2['J_loss']}W): RMSE = {RMSE2:.4f}°C, 最大误差 = {MaxError2:.4f}°C")
print("━"*70)
print(f"稳态温升预测（相对于机壳）:")
print(f"  {data1['J_loss']}W: ΔT = {(R_1 + R_2) * data1['J_loss']:.2f} °C")
print(f"  {data2['J_loss']}W: ΔT = {(R_1 + R_2) * data2['J_loss']:.2f} °C")
print("━"*70)
print("\n辨识完成！")
