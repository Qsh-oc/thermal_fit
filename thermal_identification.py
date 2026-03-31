#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电机热路参数辨识程序
高爆发永磁力矩电机 - 绕组温度预测

功能：
1. 加载温升实验数据 (temp_data.mat)
2. 辨识热路参数 (R1, R2, C1, C2)
3. 模型验证与精度评估
4. 可视化预测结果

要求：温升预测误差 < ±5°C
"""

import numpy as np
import scipy.io
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
import json


# ==================== 热路模型 ====================

def thermal_ode(t, x, params, T_case_interp, J_loss):
    """
    热路 ODE 方程
    
    状态：x = [T_1, T_coil]
    参数：params = [R1, R2, C1, C2]
    
    热平衡方程:
    C_1 * dT_1/dt = (T_case - T_1)/R_1 + (T_coil - T_1)/R_2
    C_2 * dT_coil/dt = (T_1 - T_coil)/R_2 + J_loss
    """
    R1, R2, C1, C2 = params
    
    T_1, T_coil = x
    T_case = T_case_interp(t)
    
    dT_1 = ((T_case - T_1) / R1 + (T_coil - T_1) / R2) / C1
    dT_coil = ((T_1 - T_coil) / R2 + J_loss) / C2
    
    return [dT_1, dT_coil]


def simulate_thermal(params, t_exp, T_case_exp, J_loss, x0):
    """
    仿真热路模型
    
    Args:
        params: [R1, R2, C1, C2]
        t_exp: 实验时间数组
        T_case_exp: 实验机壳温度数组
        J_loss: 损耗功率 (W)
        x0: 初始状态 [T_1, T_coil]
    
    Returns:
        T_sim: 仿真温度 [n_points, 2]
    """
    # 创建机壳温度插值函数
    T_case_interp = lambda t: np.interp(t, t_exp, T_case_exp)
    
    # 定义 ODE 函数
    def ode_func(t, x):
        return thermal_ode(t, x, params, T_case_interp, J_loss)
    
    # 求解
    sol = solve_ivp(ode_func, [t_exp[0], t_exp[-1]], x0, 
                    t_eval=t_exp, method='RK45',
                    rtol=1e-6, atol=1e-6)
    
    return sol.t, sol.y.T


def calc_objective(params, time1, T_case1, J_loss1, T_coil1,
                   time2, T_case2, J_loss2, T_coil2):
    """
    目标函数：最小化两个工况的预测误差平方和
    """
    x0_1 = [T_case1[0], T_coil1[0]]
    x0_2 = [T_case2[0], T_coil2[0]]
    
    try:
        _, T_sim1 = simulate_thermal(params, time1, T_case1, J_loss1, x0_1)
        _, T_sim2 = simulate_thermal(params, time2, T_case2, J_loss2, x0_2)
        
        error1 = T_sim1[:, 1] - T_coil1
        error2 = T_sim2[:, 1] - T_coil2
        
        obj = np.sum(error1**2) + np.sum(error2**2)
        return obj
    except Exception as e:
        return 1e10  # 返回大值避免无效参数


# ==================== 数据加载 ====================

def load_experimental_data(mat_path):
    """
    加载实验数据
    
    Returns:
        data1, data2: 两个工况的数据字典
    """
    data = scipy.io.loadmat(mat_path)
    
    # 工况 1 (J_loss = 652W)
    con1 = data['con1'][0, 0]
    data1 = {
        'time': con1['time'].flatten(),
        'T_case': con1['case'].flatten(),
        'coilF': con1['coilF'].flatten(),
        'coilB': con1['coilB'].flatten(),
        'coilM': con1['coilM'].flatten(),
        'J_loss': 652
    }
    data1['T_coil'] = (data1['coilF'] + data1['coilB'] + data1['coilM']) / 3
    
    # 工况 2 (J_loss = 452W)
    con2 = data['con2'][0, 0]
    data2 = {
        'time': con2['time'].flatten(),
        'T_case': con2['case'].flatten(),
        'coilF': con2['coilF'].flatten(),
        'coilB': con2['coilB'].flatten(),
        'coilM': con2['coilM'].flatten(),
        'J_loss': 452
    }
    data2['T_coil'] = (data2['coilF'] + data2['coilB'] + data2['coilM']) / 3
    
    return data1, data2


def estimate_steady_state_params(data1, data2):
    """
    从稳态数据估算总热阻
    """
    # 取最后 10% 数据作为稳态
    idx1 = slice(int(len(data1['time']) * 0.9), None)
    idx2 = slice(int(len(data2['time']) * 0.9), None)
    
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
    
    return R_total


# ==================== 参数辨识 ====================

def identify_parameters(data1, data2):
    """
    辨识热路参数
    """
    print("\n=== 开始参数辨识 ===")
    
    # 稳态估算
    R_total = estimate_steady_state_params(data1, data2)
    
    # 初始猜测
    R1_init = R_total * 0.6
    R2_init = R_total * 0.4
    C1_init = 150
    C2_init = 80
    params0 = [R1_init, R2_init, C1_init, C2_init]
    
    # 参数边界
    bounds = [
        (0.001, 10),      # R1
        (0.001, 10),      # R2
        (1, 10000),       # C1
        (1, 10000),       # C2
    ]
    
    # 目标函数
    objective = lambda p: calc_objective(
        p, data1['time'], data1['T_case'], data1['J_loss'], data1['T_coil'],
           data2['time'], data2['T_case'], data2['J_loss'], data2['T_coil']
    )
    
    # 优化
    print(f"初始参数：R1={R1_init:.4f}, R2={R2_init:.4f}, C1={C1_init:.2f}, C2={C2_init:.2f}")
    
    result = minimize(objective, params0, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 1000, 'ftol': 1e-6})
    
    params_opt = result.x
    R1, R2, C1, C2 = params_opt
    
    print(f"\n=== 辨识结果 ===")
    print(f"R1 = {R1:.6f} °C/W")
    print(f"R2 = {R2:.6f} °C/W")
    print(f"C1 = {C1:.2f} J/°C")
    print(f"C2 = {C2:.2f} J/°C")
    print(f"R_total = {R1 + R2:.6f} °C/W")
    
    return params_opt


# ==================== 模型验证 ====================

def validate_model(params, data1, data2):
    """
    模型验证
    """
    print("\n=== 模型验证 ===")
    
    x0_1 = [data1['T_case'][0], data1['T_coil'][0]]
    x0_2 = [data2['T_case'][0], data2['T_coil'][0]]
    
    _, T_sim1 = simulate_thermal(params, data1['time'], data1['T_case'], data1['J_loss'], x0_1)
    _, T_sim2 = simulate_thermal(params, data2['time'], data2['T_case'], data2['J_loss'], x0_2)
    
    error1 = T_sim1[:, 1] - data1['T_coil']
    error2 = T_sim2[:, 1] - data2['T_coil']
    
    RMSE1 = np.sqrt(np.mean(error1**2))
    RMSE2 = np.sqrt(np.mean(error2**2))
    MAE1 = np.mean(np.abs(error1))
    MAE2 = np.mean(np.abs(error2))
    MaxError1 = np.max(np.abs(error1))
    MaxError2 = np.max(np.abs(error2))
    
    print(f"工况 1 验证:")
    print(f"  RMSE = {RMSE1:.3f} °C, MAE = {MAE1:.3f} °C, MaxError = {MaxError1:.3f} °C")
    print(f"工况 2 验证:")
    print(f"  RMSE = {RMSE2:.3f} °C, MAE = {MAE2:.3f} °C, MaxError = {MaxError2:.3f} °C")
    
    if MaxError1 <= 5 and MaxError2 <= 5:
        print("\n✅ 精度要求满足！最大误差 < ±5°C")
    else:
        print(f"\n⚠️  精度要求未完全满足，最大误差：{max(MaxError1, MaxError2):.2f}°C")
    
    return {
        'RMSE1': RMSE1, 'RMSE2': RMSE2,
        'MAE1': MAE1, 'MAE2': MAE2,
        'MaxError1': MaxError1, 'MaxError2': MaxError2,
        'T_sim1': T_sim1, 'T_sim2': T_sim2,
        'error1': error1, 'error2': error2
    }


# ==================== 可视化 ====================

def plot_results(params, data1, data2, validation):
    """
    可视化辨识结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 子图 1: 工况 1 温度对比
    ax = axes[0, 0]
    ax.plot(data1['time'], data1['T_coil'], 'bo-', linewidth=1.5, label='实验值', markersize=3)
    ax.plot(data1['time'], validation['T_sim1'][:, 1], 'r--', linewidth=2, label='预测值')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title(f"工况 1 (J_loss = {data1['J_loss']}W) - 绕组温度")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 子图 2: 工况 2 温度对比
    ax = axes[0, 1]
    ax.plot(data2['time'], data2['T_coil'], 'bo-', linewidth=1.5, label='实验值', markersize=3)
    ax.plot(data2['time'], validation['T_sim2'][:, 1], 'r--', linewidth=2, label='预测值')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title(f"工况 2 (J_loss = {data2['J_loss']}W) - 绕组温度")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 子图 3: 工况 1 误差曲线
    ax = axes[1, 0]
    ax.plot(data1['time'], validation['error1'], 'g-', linewidth=1.5, label='预测误差')
    ax.axhline(y=5, color='r', linestyle='--', linewidth=1.5, label='+5°C 边界')
    ax.axhline(y=-5, color='r', linestyle='--', linewidth=1.5, label='-5°C 边界')
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('预测误差 (°C)')
    ax.set_title(f"工况 1 预测误差 (RMSE = {validation['RMSE1']:.2f}°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    error_range = max(abs(validation['error1'].min()), abs(validation['error1'].max()))
    ax.set_ylim([-max(10, error_range*1.1), max(10, error_range*1.1)])
    
    # 子图 4: 工况 2 误差曲线
    ax = axes[1, 1]
    ax.plot(data2['time'], validation['error2'], 'g-', linewidth=1.5, label='预测误差')
    ax.axhline(y=5, color='r', linestyle='--', linewidth=1.5)
    ax.axhline(y=-5, color='r', linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('预测误差 (°C)')
    ax.set_title(f"工况 2 预测误差 (RMSE = {validation['RMSE2']:.2f}°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    error_range = max(abs(validation['error2'].min()), abs(validation['error2'].max()))
    ax.set_ylim([-max(10, error_range*1.1), max(10, error_range*1.1)])
    
    fig.suptitle('电机热路参数辨识结果 - 绕组温度预测精度验证', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    output_path = Path(__file__).parent / 'parameter_identification_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n结果图已保存：{output_path}")
    
    return fig


# ==================== 保存结果 ====================

def save_results(params, validation, data1, data2):
    """
    保存辨识结果
    """
    R1, R2, C1, C2 = params
    
    results = {
        'parameters': {
            'R1': float(R1),
            'R2': float(R2),
            'C1': float(C1),
            'C2': float(C2),
            'R_total': float(R1 + R2)
        },
        'accuracy': {
            'RMSE1': float(validation['RMSE1']),
            'RMSE2': float(validation['RMSE2']),
            'MAE1': float(validation['MAE1']),
            'MAE2': float(validation['MAE2']),
            'MaxError1': float(validation['MaxError1']),
            'MaxError2': float(validation['MaxError2'])
        },
        'conditions': {
            'J_loss1': data1['J_loss'],
            'J_loss2': data2['J_loss']
        }
    }
    
    # 保存为 JSON
    output_path = Path(__file__).parent / 'identified_parameters.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"辨识参数已保存：{output_path}")
    
    # 打印报告
    print("\n" + "="*50)
    print("辨识报告")
    print("="*50)
    print(f"辨识参数:")
    print(f"  R1 = {R1:.6f} °C/W  (机壳→中间节点热阻)")
    print(f"  R2 = {R2:.6f} °C/W  (中间节点→线圈热阻)")
    print(f"  C1 = {C1:.2f} J/°C   (中间节点热容)")
    print(f"  C2 = {C2:.2f} J/°C   (线圈热容)")
    print("="*50)
    print(f"模型精度:")
    print(f"  工况 1 ({data1['J_loss']}W): RMSE = {validation['RMSE1']:.3f}°C, 最大误差 = {validation['MaxError1']:.3f}°C")
    print(f"  工况 2 ({data2['J_loss']}W): RMSE = {validation['RMSE2']:.3f}°C, 最大误差 = {validation['MaxError2']:.3f}°C")
    print("="*50)
    print(f"稳态特性:")
    print(f"  总热阻 R_total = {R1 + R2:.6f} °C/W")
    print(f"  {data1['J_loss']}W 稳态温升预测 = {(R1 + R2) * data1['J_loss']:.2f} °C")
    print(f"  {data2['J_loss']}W 稳态温升预测 = {(R1 + R2) * data2['J_loss']:.2f} °C")
    print("="*50)
    
    return results


# ==================== 主程序 ====================

def main():
    """
    主程序入口
    """
    print("="*50)
    print("电机热路参数辨识程序")
    print("高爆发永磁力矩电机 - 绕组温度预测")
    print("="*50)
    
    # 查找数据文件
    data_dir = Path(__file__).parent
    mat_files = list(data_dir.glob('*.mat'))
    
    if not mat_files:
        print("错误：未找到 .mat 数据文件")
        print("请将 temp_data.mat 放在同一目录下")
        return
    
    # 使用第一个 mat 文件（或指定文件名）
    mat_path = None
    for f in mat_files:
        if 'temp' in f.name.lower() or 'experimental' in f.name.lower():
            mat_path = f
            break
    if mat_path is None:
        mat_path = mat_files[0]
    
    print(f"\n使用数据文件：{mat_path}")
    
    # 加载数据
    print("\n=== 加载实验数据 ===")
    try:
        data1, data2 = load_experimental_data(str(mat_path))
        print(f"工况 1: {len(data1['time'])} 个数据点，J_loss = {data1['J_loss']}W")
        print(f"工况 2: {len(data2['time'])} 个数据点，J_loss = {data2['J_loss']}W")
    except Exception as e:
        print(f"数据加载失败：{e}")
        print("请确保 mat 文件包含 con1 和 con2 结构体")
        return
    
    # 参数辨识
    params = identify_parameters(data1, data2)
    
    # 模型验证
    validation = validate_model(params, data1, data2)
    
    # 可视化
    plot_results(params, data1, data2, validation)
    
    # 保存结果
    save_results(params, validation, data1, data2)
    
    print("\n辨识完成！")


if __name__ == "__main__":
    main()
