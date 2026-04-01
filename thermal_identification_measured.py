#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电机热路参数辨识 - 实测机壳温度边界条件版本

利用实时测量的机壳温度 T_case(t) 作为边界条件
预测绕组温升

模型结构:
T_case(t) ── R_1 ── T_1 ── R_2 ── T_coil
             │           │           │
            C_1         C_2         J_loss

待辨识参数：R_1, R_2, C_1, C_2 (4 个参数)
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
print("电机热路参数辨识 - 实测机壳温度边界条件")
print("利用 T_case(t) 实测值作为输入预测绕组温升")
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
    
    # 降采样：每 10 个点取 1 个（保留更多动态信息）
    step = 10
    return {
        'time': time[::step],
        'T_case': case[::step],
        'T_coil': T_coil[::step],
        'J_loss': J_loss,
        'T_amb': 14
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

# 线圈到机壳的热阻
R_coil_case1 = (T_coil_ss1 - T_case_ss1) / data1['J_loss']
R_coil_case2 = (T_coil_ss2 - T_case_ss2) / data2['J_loss']
R_coil_case = (R_coil_case1 + R_coil_case2) / 2

print(f"工况 1 稳态：T_coil = {T_coil_ss1:.2f}°C, T_case = {T_case_ss1:.2f}°C, ΔT = {T_coil_ss1 - T_case_ss1:.2f}°C")
print(f"工况 2 稳态：T_coil = {T_coil_ss2:.2f}°C, T_case = {T_case_ss2:.2f}°C, ΔT = {T_coil_ss2 - T_case_ss2:.2f}°C")
print(f"估算线圈→机壳热阻 R_coil_case = {R_coil_case:.6f} °C/W")

# ============ 3. 热路模型（实测 T_case 边界） ============

def thermal_ode_measured(t, x, params, T_case_interp, J_loss):
    """
    2 节点热路 ODE 方程（实测机壳温度作为边界条件）
    
    状态：x = [T_1, T_coil]
    参数：params = [R_1, R_2, C_1, C_2]
    
    热平衡方程:
    C_1 * dT_1/dt = (T_case(t) - T_1)/R_1 + (T_coil - T_1)/R_2
    C_2 * dT_coil/dt = (T_1 - T_coil)/R_2 + J_loss
    
    T_case(t) 是实测值（通过插值获取）
    """
    R_1, R_2, C_1, C_2 = params
    
    T_1, T_coil = x
    T_case = T_case_interp(t)
    
    dT_1 = ((T_case - T_1) / R_1 + (T_coil - T_1) / R_2) / C_1
    dT_coil = ((T_1 - T_coil) / R_2 + J_loss) / C_2
    
    return [dT_1, dT_coil]


def simulate_measured(params, t_exp, T_case_exp, J_loss, x0):
    """仿真（使用实测机壳温度）"""
    
    # 创建机壳温度插值函数
    T_case_interp = lambda t: np.interp(t, t_exp, T_case_exp, 
                                         left=T_case_exp[0], 
                                         right=T_case_exp[-1])
    
    def ode_func(t, x):
        return thermal_ode_measured(t, x, params, T_case_interp, J_loss)
    
    sol = solve_ivp(ode_func, [t_exp[0], t_exp[-1]], x0, 
                    t_eval=t_exp, method='RK45', rtol=1e-8, atol=1e-8)
    
    if not sol.success:
        raise ValueError(f"ODE 求解失败：{sol.message}")
    
    return sol.t, sol.y.T


def objective_measured(params):
    """目标函数：最小化两个工况的线圈温度预测误差"""
    # 初始状态：[T_1, T_coil] 从实验数据的初始值开始
    x0_1 = [data1['T_case'][0], data1['T_coil'][0]]
    x0_2 = [data2['T_case'][0], data2['T_coil'][0]]
    
    try:
        _, T_sim1 = simulate_measured(params, data1['time'], data1['T_case'], 
                                       data1['J_loss'], x0_1)
        _, T_sim2 = simulate_measured(params, data2['time'], data2['T_case'], 
                                       data2['J_loss'], x0_2)
        
        # T_sim[:, 0] = T_1, T_sim[:, 1] = T_coil
        T_coil_sim1 = T_sim1[:, 1]
        T_coil_sim2 = T_sim2[:, 1]
        
        error1 = T_coil_sim1 - data1['T_coil']
        error2 = T_coil_sim2 - data2['T_coil']
        
        # 误差平方和
        obj = np.sum(error1**2) + np.sum(error2**2)
        
        return obj
    except Exception as e:
        return 1e10


# ============ 4. 参数辨识 ============
print("\n" + "="*70)
print("参数辨识（差分进化全局优化）")
print("="*70)

# 参数边界
# [R_1, R_2, C_1, C_2]
bounds = [
    (0.0001, 0.2),    # R_1: 机壳→中间节点
    (0.0001, 0.2),    # R_2: 中间节点→线圈
    (10, 2000),       # C_1: 中间节点热容
    (10, 2000),       # C_2: 线圈热容
]

print("参数范围:")
print(f"  R_1  (机壳→中间):  0.0001 - 0.2 °C/W")
print(f"  R_2  (中间→线圈):  0.0001 - 0.2 °C/W")
print(f"  C_1  (中间热容):   10 - 2000 J/°C")
print(f"  C_2  (线圈热容):   10 - 2000 J/°C")
print("\n开始全局优化...")

result = differential_evolution(
    objective_measured, bounds,
    seed=42,
    maxiter=500,
    tol=1e-8,
    workers=1,
    updating='deferred',
    polish=True,
    disp=True,
    popsize=20
)

R_1, R_2, C_1, C_2 = result.x

print(f"\n{'='*70}")
print("辨识结果")
print(f"{'='*70}")
print(f"R_1     = {R_1:.8f} °C/W  (机壳→中间节点)")
print(f"R_2     = {R_2:.8f} °C/W  (中间节点→线圈)")
print(f"C_1     = {C_1:.4f} J/°C   (中间节点热容)")
print(f"C_2     = {C_2:.4f} J/°C   (线圈热容)")
print(f"R_total = {R_1 + R_2:.8f} °C/W")

# ============ 5. 模型验证 ============
print("\n" + "="*70)
print("模型验证")
print("="*70)

x0_1 = [data1['T_case'][0], data1['T_coil'][0]]
x0_2 = [data2['T_case'][0], data2['T_coil'][0]]

_, T_sim1 = simulate_measured(result.x, data1['time'], data1['T_case'], 
                               data1['J_loss'], x0_1)
_, T_sim2 = simulate_measured(result.x, data2['time'], data2['T_case'], 
                               data2['J_loss'], x0_2)

T_coil_sim1 = T_sim1[:, 1]
T_coil_sim2 = T_sim2[:, 1]

error1 = T_coil_sim1 - data1['T_coil']
error2 = T_coil_sim2 - data2['T_coil']

def calc_metrics(error):
    return {
        'RMSE': np.sqrt(np.mean(error**2)),
        'MAE': np.mean(np.abs(error)),
        'MaxError': np.max(np.abs(error))
    }

metrics1 = calc_metrics(error1)
metrics2 = calc_metrics(error2)

print(f"工况 1 (652W): RMSE = {metrics1['RMSE']:.4f}°C, MAE = {metrics1['MAE']:.4f}°C, MaxError = {metrics1['MaxError']:.4f}°C")
print(f"工况 2 (452W): RMSE = {metrics2['RMSE']:.4f}°C, MAE = {metrics2['MAE']:.4f}°C, MaxError = {metrics2['MaxError']:.4f}°C")

max_error = max(metrics1['MaxError'], metrics2['MaxError'])
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

# 工况 1: 机壳温度
ax = plt.subplot(3, 2, 1)
ax.plot(data1['time'], data1['T_case'], 'b-', linewidth=1.5, label='T_case 实测')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 1 (J_loss = {data1['J_loss']}W) - 机壳温度（实测输入）")
ax.legend()
ax.grid(True, alpha=0.3)

# 工况 1: 线圈温度对比
ax = plt.subplot(3, 2, 2)
ax.plot(data1['time'], data1['T_coil'], 'bo', markersize=3, alpha=0.5, label='实验值')
ax.plot(data1['time'], T_coil_sim1, 'r-', linewidth=2, label='预测值')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 1 - 绕组温度预测")
ax.legend()
ax.grid(True, alpha=0.3)

# 工况 2: 机壳温度
ax = plt.subplot(3, 2, 3)
ax.plot(data2['time'], data2['T_case'], 'b-', linewidth=1.5, label='T_case 实测')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 2 (J_loss = {data2['J_loss']}W) - 机壳温度（实测输入）")
ax.legend()
ax.grid(True, alpha=0.3)

# 工况 2: 线圈温度对比
ax = plt.subplot(3, 2, 4)
ax.plot(data2['time'], data2['T_coil'], 'bo', markersize=3, alpha=0.5, label='实验值')
ax.plot(data2['time'], T_coil_sim2, 'r-', linewidth=2, label='预测值')
ax.set_ylabel('温度 (°C)')
ax.set_title(f"工况 2 - 绕组温度预测")
ax.legend()
ax.grid(True, alpha=0.3)

# 工况 1: 误差曲线
ax = plt.subplot(3, 2, 5)
ax.plot(data1['time'], error1, 'g-', linewidth=1.5, label='预测误差')
ax.axhline(y=5, color='r', linestyle='--', linewidth=2, label='+5°C 边界')
ax.axhline(y=-5, color='r', linestyle='--', linewidth=2)
ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
ax.fill_between(data1['time'], -5, 5, alpha=0.2, color='green')
ax.set_xlabel('时间 (s)')
ax.set_ylabel('预测误差 (°C)')
ax.set_title(f"工况 1 误差 (RMSE={metrics1['RMSE']:.3f}°C, Max={metrics1['MaxError']:.3f}°C)")
ax.legend()
ax.grid(True, alpha=0.3)

# 工况 2: 误差曲线
ax = plt.subplot(3, 2, 6)
ax.plot(data2['time'], error2, 'g-', linewidth=1.5, label='预测误差')
ax.axhline(y=5, color='r', linestyle='--', linewidth=2)
ax.axhline(y=-5, color='r', linestyle='--', linewidth=2)
ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
ax.fill_between(data2['time'], -5, 5, alpha=0.2, color='green')
ax.set_xlabel('时间 (s)')
ax.set_ylabel('预测误差 (°C)')
ax.set_title(f"工况 2 误差 (RMSE={metrics2['RMSE']:.3f}°C, Max={metrics2['MaxError']:.3f}°C)")
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
        'RMSE1': float(metrics1['RMSE']),
        'RMSE2': float(metrics2['RMSE']),
        'MAE1': float(metrics1['MAE']),
        'MAE2': float(metrics2['MAE']),
        'MaxError1': float(metrics1['MaxError']),
        'MaxError2': float(metrics2['MaxError']),
        'accuracy_pass': accuracy_pass
    },
    'conditions': {
        'J_loss1': data1['J_loss'],
        'J_loss2': data2['J_loss'],
        'T_amb': 14
    }
}

output_path = Path(__file__).parent / 'identified_parameters_measured.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"参数已保存：{output_path}")

# 打印报告
print("\n" + "━"*70)
print("辨识报告 - 实测机壳温度边界条件")
print("━"*70)
print("热阻参数:")
print(f"  R_1     = {R_1:.8f} °C/W  (机壳→中间节点)")
print(f"  R_2     = {R_2:.8f} °C/W  (中间节点→线圈)")
print("热容参数:")
print(f"  C_1     = {C_1:.4f} J/°C   (中间节点热容)")
print(f"  C_2     = {C_2:.4f} J/°C   (线圈热容)")
print("━"*70)
print("模型精度（绕组温度）:")
print(f"  工况 1 (652W): RMSE = {metrics1['RMSE']:.4f}°C, 最大误差 = {metrics1['MaxError']:.4f}°C")
print(f"  工况 2 (452W): RMSE = {metrics2['RMSE']:.4f}°C, 最大误差 = {metrics2['MaxError']:.4f}°C")
print("━"*70)
print(f"稳态温升预测（相对于机壳）:")
print(f"  652W: ΔT_coil_case = {(R_1 + R_2) * 652:.2f} °C")
print(f"  452W: ΔT_coil_case = {(R_1 + R_2) * 452:.2f} °C")
print("━"*70)
print("\n辨识完成！")

# ============ 8. 创建预测工具 ============
print("\n生成实时预测工具...")

predict_code = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绕组温度实时预测工具
使用实测机壳温度作为边界条件

辨识参数:
R_1 = {R_1:.8f} °C/W
R_2 = {R_2:.8f} °C/W
C_1 = {C_1:.4f} J/°C
C_2 = {C_2:.4f} J/°C
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 辨识参数
PARAMS = {{
    'R_1': {R_1},
    'R_2': {R_2},
    'C_1': {C_1},
    'C_2': {C_2}
}}

def predict_winding_temp(T_case_time, T_case_values, J_loss, T_init=None):
    """
    预测绕组温度
    
    Args:
        T_case_time: 机壳温度时间数组 (s)
        T_case_values: 机壳温度实测值 (°C)
        J_loss: 损耗功率 (W)
        T_init: 初始温度 [T_1, T_coil]，默认从机壳温度开始
    
    Returns:
        t, T_coil: 时间和绕组温度预测值
    """
    R_1, R_2, C_1, C_2 = PARAMS['R_1'], PARAMS['R_2'], PARAMS['C_1'], PARAMS['C_2']
    
    # 创建机壳温度插值函数
    T_case_interp = lambda t: np.interp(t, T_case_time, T_case_values,
                                         left=T_case_values[0],
                                         right=T_case_values[-1])
    
    def ode_func(t, x):
        T_1, T_coil = x
        T_case = T_case_interp(t)
        
        dT_1 = ((T_case - T_1) / R_1 + (T_coil - T_1) / R_2) / C_1
        dT_coil = ((T_1 - T_coil) / R_2 + J_loss) / C_2
        
        return [dT_1, dT_coil]
    
    # 初始状态
    if T_init is None:
        T_init = [T_case_values[0], T_case_values[0]]
    
    # 求解
    sol = solve_ivp(ode_func, [T_case_time[0], T_case_time[-1]], T_init,
                    t_eval=T_case_time, method='RK45', rtol=1e-6, atol=1e-6)
    
    return sol.t, sol.y[1]


if __name__ == "__main__":
    # 示例：使用实验数据验证
    import scipy.io
    
    mat_path = Path(__file__).parent / 'temp_data.mat'
    data = scipy.io.loadmat(str(mat_path))
    
    # 加载工况 1
    con1 = data['con1'][0, 0]
    time = con1['time'].flatten()[::10]
    T_case = con1['case'].flatten()[::10]
    T_coil_exp = ((con1['coilF'] + con1['coilB'] + con1['coilM']) / 3).flatten()[::10]
    
    # 预测
    t_pred, T_coil_pred = predict_winding_temp(time, T_case, J_loss=652)
    
    # 计算误差
    error = T_coil_pred - T_coil_exp
    RMSE = np.sqrt(np.mean(error**2))
    MaxError = np.max(np.abs(error))
    
    print(f"验证结果:")
    print(f"  RMSE = {{RMSE:.4f}}°C")
    print(f"  MaxError = {{MaxError:.4f}}°C")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, T_coil_exp, 'bo', markersize=3, alpha=0.5, label='实验值')
    ax.plot(t_pred, T_coil_pred, 'r-', linewidth=2, label='预测值')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('绕组温度 (°C)')
    ax.set_title('绕组温度预测验证 (J_loss = 652W)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'prediction_verify.png', dpi=150)
    print("验证图已保存：prediction_verify.png")
    plt.show()
'''

predict_path = Path(__file__).parent / 'winding_temp_predict.py'
with open(predict_path, 'w', encoding='utf-8') as f:
    f.write(predict_code)
print(f"预测工具已创建：{predict_path}")

print("\n" + "="*70)
print("全部完成！")
print("="*70)
