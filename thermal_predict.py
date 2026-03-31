#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电机绕组温度实时预测工具
使用已辨识的热路参数进行温度预测

用法：
    python thermal_predict.py --J_loss 500 --T_case 35 --duration 300
"""

import numpy as np
import scipy.io
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse


def load_identified_parameters(param_file=None):
    """
    加载已辨识的参数
    """
    if param_file is None:
        param_file = Path(__file__).parent / 'identified_parameters.json'
    
    with open(param_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    params = data['parameters']
    print("已加载辨识参数:")
    print(f"  R1 = {params['R1']:.6f} °C/W")
    print(f"  R2 = {params['R2']:.6f} °C/W")
    print(f"  C1 = {params['C1']:.2f} J/°C")
    print(f"  C2 = {params['C2']:.2f} J/°C")
    
    return params


def thermal_ode(t, x, params, T_case, J_loss):
    """热路 ODE 方程"""
    R1, R2, C1, C2 = params['R1'], params['R2'], params['C1'], params['C2']
    
    T_1, T_coil = x
    
    dT_1 = ((T_case - T_1) / R1 + (T_coil - T_1) / R2) / C1
    dT_coil = ((T_1 - T_coil) / R2 + J_loss) / C2
    
    return [dT_1, dT_coil]


def predict_temperature(params, J_loss, T_case, duration=300, T_init=25, n_points=1000):
    """
    预测绕组温度
    
    Args:
        params: 辨识参数字典
        J_loss: 损耗功率 (W)
        T_case: 机壳温度 (°C)，可以是常数或数组
        duration: 预测时长 (s)
        T_init: 初始温度 (°C)
        n_points: 输出点数
    
    Returns:
        t, T_coil, T_1: 时间、线圈温度、中间节点温度
    """
    t = np.linspace(0, duration, n_points)
    
    # 处理 T_case（常数或时变）
    if np.isscalar(T_case):
        T_case_func = lambda t: T_case
    else:
        T_case_interp = np.interp
        T_case_func = lambda t: np.interp(t, np.linspace(0, duration, len(T_case)), T_case)
    
    x0 = [T_case, T_init]  # 初始状态
    
    def ode_func(t, x):
        return thermal_ode(t, x, params, T_case_func(t), J_loss)
    
    sol = solve_ivp(ode_func, [0, duration], x0, t_eval=t, method='RK45')
    
    return sol.t, sol.y[1], sol.y[0]


def plot_prediction(t, T_coil, T_1, T_case, J_loss, params):
    """
    绘制预测结果
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 温度曲线
    ax = axes[0]
    ax.plot(t, T_coil, 'r-', linewidth=2, label='T_coil (绕组)')
    ax.plot(t, T_1, 'b--', linewidth=1.5, label='T_1 (中间节点)')
    if np.isscalar(T_case):
        ax.axhline(y=T_case, color='g', linestyle=':', linewidth=1.5, label=f'T_case = {T_case}°C')
    else:
        ax.plot(t, T_case, 'g:', linewidth=1.5, label='T_case (机壳)')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('温度 (°C)')
    ax.set_title(f'绕组温度预测 (J_loss = {J_loss}W)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 温升曲线
    ax = axes[1]
    T_rise = T_coil - T_case if np.isscalar(T_case) else T_coil - T_case
    ax.plot(t, T_rise, 'r-', linewidth=2)
    ax.axhline(y=params['R_total'] * J_loss, color='k', linestyle='--', 
               label=f'稳态温升 = {params["R_total"] * J_loss:.2f}°C')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('温升 (°C)')
    ax.set_title('绕组温升曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = Path(__file__).parent / f'prediction_J{J_loss}W.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"预测图已保存：{output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='电机绕组温度预测工具')
    parser.add_argument('--J_loss', type=float, required=True, help='损耗功率 (W)')
    parser.add_argument('--T_case', type=float, default=25, help='机壳温度 (°C)')
    parser.add_argument('--duration', type=int, default=300, help='预测时长 (s)')
    parser.add_argument('--T_init', type=float, default=25, help='初始绕组温度 (°C)')
    parser.add_argument('--param_file', type=str, default=None, help='参数文件路径')
    
    args = parser.parse_args()
    
    print("="*50)
    print("电机绕组温度预测工具")
    print("="*50)
    
    # 加载参数
    params = load_identified_parameters(args.param_file)
    
    # 预测
    print(f"\n预测条件:")
    print(f"  损耗功率：{args.J_loss} W")
    print(f"  机壳温度：{args.T_case} °C")
    print(f"  初始温度：{args.T_init} °C")
    print(f"  预测时长：{args.duration} s")
    
    t, T_coil, T_1 = predict_temperature(
        params, args.J_loss, args.T_case, 
        duration=args.duration, T_init=args.T_init
    )
    
    # 输出结果
    print(f"\n预测结果:")
    print(f"  初始绕组温度：{T_coil[0]:.2f} °C")
    print(f"  最终绕组温度：{T_coil[-1]:.2f} °C")
    print(f"  稳态绕组温度：{args.T_case + params['R_total'] * args.J_loss:.2f} °C")
    print(f"  最大温升：{T_coil.max() - args.T_case:.2f} °C")
    
    # 检查是否超限
    if T_coil.max() > 155:  # 假设绝缘等级 H 级 (155°C)
        print("\n⚠️  警告：预测温度超过 155°C (H 级绝缘极限)")
    elif T_coil.max() > 130:
        print("\n⚠️  注意：预测温度超过 130°C (F 级绝缘极限)")
    else:
        print("\n✅ 温度在安全范围内")
    
    # 绘图
    plot_prediction(t, T_coil, T_1, args.T_case, args.J_loss, params)


if __name__ == "__main__":
    main()
