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

热路模型:
T_case ── R_1 ── T_1 ── R_2 ── T_coil
                  │           │
                 C_1         C_2
"""

import numpy as np
import scipy.io
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')


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
        t_sim, T_sim: 仿真时间和温度 [n_points, 2]
    """
    # 创建机壳温度插值函数
    T_case_interp = lambda t: np.interp(t, t_exp, T_case_exp, left=T_case_exp[0], right=T_case_exp[-1])
    
    # 定义 ODE 函数
    def ode_func(t, x):
        return thermal_ode(t, x, params, T_case_interp, J_loss)
    
    # 求解
    sol = solve_ivp(ode_func, [t_exp[0], t_exp[-1]], x0, 
                    t_eval=t_exp, method='RK45',
                    rtol=1e-8, atol=1e-8)
    
    if not sol.success:
        raise ValueError(f"ODE 求解失败：{sol.message}")
    
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
        
        # 加权误差平方和
        obj = np.sum(error1**2) + np.sum(error2**2)
        return obj
    except Exception:
        return 1e10  # 返回大值避免无效参数


# ==================== 数据加载 ====================

def load_experimental_data(mat_path):
    """
    加载实验数据
    
    支持两种数据格式：
    1. 嵌套结构：data['con1'][0,0]['time']
    2. 直接数组：data['con1']['time'][0,0]
    
    Returns:
        data1, data2: 两个工况的数据字典
    """
    data = scipy.io.loadmat(mat_path)
    
    # 环境温度
    T_amb = 14  # °C (测试时室温)
    
    def extract_field(con, field_name):
        """提取字段并展平为 1D 数组"""
        val = con[field_name]
        # 处理嵌套结构
        if val.ndim == 2 and val.shape[0] == 1:
            val = val[0, 0]
        return val.flatten()
    
    # 工况 1 (J_loss = 652W)
    con1 = data['con1']
    # 检查数据结构类型
    if con1.ndim == 2 and con1.shape == (1, 1):
        con1 = con1[0, 0]
    
    data1 = {
        'time': extract_field(con1, 'time'),
        'T_case': extract_field(con1, 'case'),
        'coilF': extract_field(con1, 'coilF'),
        'coilB': extract_field(con1, 'coilB'),
        'coilM': extract_field(con1, 'coilM'),
        'J_loss': 652,
        'T_amb': T_amb
    }
    data1['T_coil'] = (data1['coilF'] + data1['coilB'] + data1['coilM']) / 3
    
    # 工况 2 (J_loss = 452W)
    con2 = data['con2']
    if con2.ndim == 2 and con2.shape == (1, 1):
        con2 = con2[0, 0]
    
    data2 = {
        'time': extract_field(con2, 'time'),
        'T_case': extract_field(con2, 'case'),
        'coilF': extract_field(con2, 'coilF'),
        'coilB': extract_field(con2, 'coilB'),
        'coilM': extract_field(con2, 'coilM'),
        'J_loss': 452,
        'T_amb': T_amb
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

def identify_parameters(data1, data2, method='hybrid'):
    """
    辨识热路参数
    
    Args:
        data1, data2: 两个工况的数据
        method: 辨识方法 ('hybrid' | 'global' | 'local')
            - hybrid: 全局搜索 + 局部优化 (推荐)
            - global: 仅全局优化 (差分进化)
            - local: 仅局部优化 (L-BFGS-B)
    """
    print("\n" + "="*60)
    print("开始参数辨识")
    print("="*60)
    
    # 稳态估算
    R_total = estimate_steady_state_params(data1, data2)
    
    # 初始猜测
    R1_init = R_total * 0.6
    R2_init = R_total * 0.4
    C1_init = 150
    C2_init = 80
    
    # 参数边界
    bounds = [
        (0.0001, 5),      # R1
        (0.0001, 5),      # R2
        (10, 5000),       # C1
        (10, 5000),       # C2
    ]
    
    # 目标函数
    objective = lambda p: calc_objective(
        p, data1['time'], data1['T_case'], data1['J_loss'], data1['T_coil'],
           data2['time'], data2['T_case'], data2['J_loss'], data2['T_coil']
    )
    
    if method == 'hybrid':
        # 第一阶段：全局搜索（差分进化）
        print("\n[阶段 1] 全局搜索 (差分进化)...")
        result_global = differential_evolution(
            objective, bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-6,
            workers=1,
            updating='deferred',
            polish=False,
            disp=True
        )
        params_global = result_global.x
        print(f"全局搜索结果：{params_global}")
        
        # 第二阶段：局部优化
        print("\n[阶段 2] 局部优化 (L-BFGS-B)...")
        result_local = minimize(
            objective, params_global, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        params_opt = result_local.x
        
    elif method == 'global':
        print("\n[单阶段] 全局优化 (差分进化)...")
        result = differential_evolution(
            objective, bounds,
            seed=42,
            maxiter=500,
            tol=1e-8,
            workers=1,
            disp=True
        )
        params_opt = result.x
        
    else:  # local
        print("\n[单阶段] 局部优化 (L-BFGS-B)...")
        params0 = [R1_init, R2_init, C1_init, C2_init]
        result = minimize(
            objective, params0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        params_opt = result.x
    
    R1, R2, C1, C2 = params_opt
    
    print(f"\n{'='*60}")
    print("辨识结果")
    print(f"{'='*60}")
    print(f"R1      = {R1:.8f} °C/W")
    print(f"R2      = {R2:.8f} °C/W")
    print(f"C1      = {C1:.4f} J/°C")
    print(f"C2      = {C2:.4f} J/°C")
    print(f"R_total = {R1 + R2:.8f} °C/W")
    print(f"{'='*60}")
    
    return params_opt


# ==================== 模型验证 ====================

def validate_model(params, data1, data2):
    """
    模型验证
    """
    print("\n" + "="*60)
    print("模型验证")
    print("="*60)
    
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
    MAPE1 = np.mean(np.abs(error1 / data1['T_coil'])) * 100
    MAPE2 = np.mean(np.abs(error2 / data2['T_coil'])) * 100
    
    print(f"工况 1 (J_loss = {data1['J_loss']}W):")
    print(f"  RMSE   = {RMSE1:.4f} °C")
    print(f"  MAE    = {MAE1:.4f} °C")
    print(f"  MAPE   = {MAPE1:.4f} %")
    print(f"  MaxErr = {MaxError1:.4f} °C")
    print(f"工况 2 (J_loss = {data2['J_loss']}W):")
    print(f"  RMSE   = {RMSE2:.4f} °C")
    print(f"  MAE    = {MAE2:.4f} °C")
    print(f"  MAPE   = {MAPE2:.4f} %")
    print(f"  MaxErr = {MaxError2:.4f} °C")
    
    # 精度判定
    max_error = max(MaxError1, MaxError2)
    if max_error <= 5:
        print(f"\n✅ 精度要求满足！最大误差 {max_error:.4f}°C < ±5°C")
        accuracy_pass = True
    else:
        print(f"\n⚠️  精度要求未完全满足，最大误差：{max_error:.4f}°C > ±5°C")
        accuracy_pass = False
    
    return {
        'RMSE1': RMSE1, 'RMSE2': RMSE2,
        'MAE1': MAE1, 'MAE2': MAE2,
        'MAPE1': MAPE1, 'MAPE2': MAPE2,
        'MaxError1': MaxError1, 'MaxError2': MaxError2,
        'T_sim1': T_sim1, 'T_sim2': T_sim2,
        'error1': error1, 'error2': error2,
        'accuracy_pass': accuracy_pass
    }


# ==================== 可视化 ====================

def plot_results(params, data1, data2, validation, save_path=None):
    """
    可视化辨识结果
    """
    R1, R2, C1, C2 = params
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('电机热路参数辨识结果 - 绕组温度预测精度验证', fontsize=16, fontweight='bold')
    
    # 子图 1: 工况 1 温度对比
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(data1['time'], data1['T_coil'], 'bo', markersize=4, alpha=0.7, label='实验值')
    ax1.plot(data1['time'], validation['T_sim1'][:, 1], 'r-', linewidth=2, label='预测值')
    ax1.set_xlabel('时间 (s)', fontsize=11)
    ax1.set_ylabel('温度 (°C)', fontsize=11)
    ax1.set_title(f"工况 1 (J_loss = {data1['J_loss']}W)", fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 子图 2: 工况 2 温度对比
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(data2['time'], data2['T_coil'], 'bo', markersize=4, alpha=0.7, label='实验值')
    ax2.plot(data2['time'], validation['T_sim2'][:, 1], 'r-', linewidth=2, label='预测值')
    ax2.set_xlabel('时间 (s)', fontsize=11)
    ax2.set_ylabel('温度 (°C)', fontsize=11)
    ax2.set_title(f"工况 2 (J_loss = {data2['J_loss']}W)", fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 子图 3: 工况 1 误差曲线
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(data1['time'], validation['error1'], 'g-', linewidth=1.5, label='预测误差')
    ax3.axhline(y=5, color='r', linestyle='--', linewidth=2, label='+5°C 边界')
    ax3.axhline(y=-5, color='r', linestyle='--', linewidth=2, label='-5°C 边界')
    ax3.axhline(y=0, color='k', linestyle=':', linewidth=1)
    ax3.fill_between(data1['time'], -5, 5, alpha=0.2, color='green', label='允许误差范围')
    ax3.set_xlabel('时间 (s)', fontsize=11)
    ax3.set_ylabel('预测误差 (°C)', fontsize=11)
    ax3.set_title(f"工况 1 误差 (RMSE={validation['RMSE1']:.3f}°C, Max={validation['MaxError1']:.3f}°C)", 
                  fontsize=11, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 子图 4: 工况 2 误差曲线
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(data2['time'], validation['error2'], 'g-', linewidth=1.5, label='预测误差')
    ax4.axhline(y=5, color='r', linestyle='--', linewidth=2)
    ax4.axhline(y=-5, color='r', linestyle='--', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle=':', linewidth=1)
    ax4.fill_between(data2['time'], -5, 5, alpha=0.2, color='green')
    ax4.set_xlabel('时间 (s)', fontsize=11)
    ax4.set_ylabel('预测误差 (°C)', fontsize=11)
    ax4.set_title(f"工况 2 误差 (RMSE={validation['RMSE2']:.3f}°C, Max={validation['MaxError2']:.3f}°C)",
                  fontsize=11, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n结果图已保存：{save_path}")
    
    plt.show()
    
    return fig


def plot_parameter_sensitivity(params, data1):
    """
    参数敏感性分析
    """
    R1, R2, C1, C2 = params
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('参数敏感性分析', fontsize=14, fontweight='bold')
    
    # 基准仿真
    x0 = [data1['T_case'][0], data1['T_coil'][0]]
    _, T_base = simulate_thermal(params, data1['time'], data1['T_case'], data1['J_loss'], x0)
    
    # R1 敏感性
    ax = axes[0, 0]
    for factor in [0.5, 0.8, 1.0, 1.2, 1.5]:
        p = [R1*factor, R2, C1, C2]
        _, T = simulate_thermal(p, data1['time'], data1['T_case'], data1['J_loss'], x0)
        ax.plot(data1['time'], T[:, 1], '--', linewidth=1.5, label=f'R1 × {factor}')
    ax.plot(data1['time'], data1['T_coil'], 'k-', linewidth=2, alpha=0.5, label='实验值')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('T_coil (°C)')
    ax.set_title('R1 敏感性')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R2 敏感性
    ax = axes[0, 1]
    for factor in [0.5, 0.8, 1.0, 1.2, 1.5]:
        p = [R1, R2*factor, C1, C2]
        _, T = simulate_thermal(p, data1['time'], data1['T_case'], data1['J_loss'], x0)
        ax.plot(data1['time'], T[:, 1], '--', linewidth=1.5, label=f'R2 × {factor}')
    ax.plot(data1['time'], data1['T_coil'], 'k-', linewidth=2, alpha=0.5, label='实验值')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('T_coil (°C)')
    ax.set_title('R2 敏感性')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # C1 敏感性
    ax = axes[1, 0]
    for factor in [0.5, 0.8, 1.0, 1.2, 1.5]:
        p = [R1, R2, C1*factor, C2]
        _, T = simulate_thermal(p, data1['time'], data1['T_case'], data1['J_loss'], x0)
        ax.plot(data1['time'], T[:, 1], '--', linewidth=1.5, label=f'C1 × {factor}')
    ax.plot(data1['time'], data1['T_coil'], 'k-', linewidth=2, alpha=0.5, label='实验值')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('T_coil (°C)')
    ax.set_title('C1 敏感性')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # C2 敏感性
    ax = axes[1, 1]
    for factor in [0.5, 0.8, 1.0, 1.2, 1.5]:
        p = [R1, R2, C1, C2*factor]
        _, T = simulate_thermal(p, data1['time'], data1['T_case'], data1['J_loss'], x0)
        ax.plot(data1['time'], T[:, 1], '--', linewidth=1.5, label=f'C2 × {factor}')
    ax.plot(data1['time'], data1['T_coil'], 'k-', linewidth=2, alpha=0.5, label='实验值')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('T_coil (°C)')
    ax.set_title('C2 敏感性')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(__file__).parent / 'parameter_sensitivity.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"敏感性分析图已保存：{save_path}")
    
    plt.show()


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
            'MAPE1': float(validation['MAPE1']),
            'MAPE2': float(validation['MAPE2']),
            'MaxError1': float(validation['MaxError1']),
            'MaxError2': float(validation['MaxError2']),
            'accuracy_pass': bool(validation['accuracy_pass'])
        },
        'conditions': {
            'J_loss1': data1['J_loss'],
            'J_loss2': data2['J_loss'],
            'T_amb': data1['T_amb']
        }
    }
    
    # 保存为 JSON
    output_path = Path(__file__).parent / 'identified_parameters.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n辨识参数已保存：{output_path}")
    
    # 打印报告
    print("\n" + "━"*60)
    print("辨识报告")
    print("━"*60)
    print(f"辨识参数:")
    print(f"  R1      = {R1:.8f} °C/W  (机壳→中间节点热阻)")
    print(f"  R2      = {R2:.8f} °C/W  (中间节点→线圈热阻)")
    print(f"  C1      = {C1:.4f} J/°C   (中间节点热容)")
    print(f"  C2      = {C2:.4f} J/°C   (线圈热容)")
    print(f"  R_total = {R1 + R2:.8f} °C/W")
    print("━"*60)
    print(f"模型精度:")
    print(f"  工况 1 ({data1['J_loss']}W): RMSE = {validation['RMSE1']:.4f}°C, 最大误差 = {validation['MaxError1']:.4f}°C")
    print(f"  工况 2 ({data2['J_loss']}W): RMSE = {validation['RMSE2']:.4f}°C, 最大误差 = {validation['MaxError2']:.4f}°C")
    print("━"*60)
    print(f"稳态特性:")
    print(f"  {data1['J_loss']}W 稳态温升预测 = {(R1 + R2) * data1['J_loss']:.2f} °C")
    print(f"  {data2['J_loss']}W 稳态温升预测 = {(R1 + R2) * data2['J_loss']:.2f} °C")
    print("━"*60)
    
    return results


# ==================== 主程序 ====================

def main(mat_file=None, method='hybrid', do_sensitivity=False):
    """
    主程序入口
    
    Args:
        mat_file: .mat 数据文件路径
        method: 辨识方法 ('hybrid' | 'global' | 'local')
        do_sensitivity: 是否进行参数敏感性分析
    """
    print("="*60)
    print("电机热路参数辨识程序")
    print("高爆发永磁力矩电机 - 绕组温度预测")
    print("="*60)
    
    # 查找数据文件
    data_dir = Path(__file__).parent
    if mat_file is None:
        mat_files = list(data_dir.glob('*.mat'))
        if not mat_files:
            print("错误：未找到 .mat 数据文件")
            print("请将 temp_data.mat 放在同一目录下")
            return None
        
        # 优先使用包含 temp 或 experimental 的文件
        mat_file = None
        for f in mat_files:
            if 'temp' in f.name.lower() or 'experimental' in f.name.lower():
                mat_file = f
                break
        if mat_file is None:
            mat_file = mat_files[0]
    
    print(f"\n使用数据文件：{mat_file}")
    
    # 加载数据
    print("\n" + "="*60)
    print("加载实验数据")
    print("="*60)
    try:
        data1, data2 = load_experimental_data(str(mat_file))
        print(f"工况 1: {len(data1['time'])} 个数据点，J_loss = {data1['J_loss']}W")
        print(f"工况 2: {len(data2['time'])} 个数据点，J_loss = {data2['J_loss']}W")
        print(f"环境温度：T_amb = {data1['T_amb']}°C")
    except Exception as e:
        print(f"数据加载失败：{e}")
        print("请确保 mat 文件包含 con1 和 con2 结构体")
        return None
    
    # 参数辨识
    params = identify_parameters(data1, data2, method=method)
    
    # 模型验证
    validation = validate_model(params, data1, data2)
    
    # 可视化
    result_path = data_dir / 'parameter_identification_result.png'
    plot_results(params, data1, data2, validation, save_path=result_path)
    
    # 敏感性分析（可选）
    if do_sensitivity:
        print("\n进行参数敏感性分析...")
        plot_parameter_sensitivity(params, data1)
    
    # 保存结果
    save_results(params, validation, data1, data2)
    
    print("\n" + "="*60)
    print("辨识完成！")
    print("="*60)
    
    return {
        'params': params,
        'validation': validation
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='电机热路参数辨识程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python thermal_identification.py
  python thermal_identification.py --mat temp_data.mat
  python thermal_identification.py --method global
  python thermal_identification.py --sensitivity
        """
    )
    parser.add_argument('--mat', type=str, default=None, 
                        help='输入 .mat 数据文件路径')
    parser.add_argument('--method', type=str, default='hybrid',
                        choices=['hybrid', 'global', 'local'],
                        help='辨识方法 (默认：hybrid)')
    parser.add_argument('--sensitivity', action='store_true',
                        help='进行参数敏感性分析')
    
    args = parser.parse_args()
    
    main(mat_file=args.mat, method=args.method, do_sensitivity=args.sensitivity)
