#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
永磁力矩电机集总参数热路模型
高爆发工况下温度场仿真

热路模型结构 (Cauer 型):

T_case ── R_1 ── T_1 ── R_2 ── T_coil
                  │           │
                 C_1         C_2
                  │           │
                 GND         T_a
                              │
T_a ── R_3 ── T_3 ── R_4 ── T_a
         │           │
        C_3         J_loss (热源)
         │
        GND
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class ThermalParameters:
    """热路模型参数"""
    # 热阻 (°C/W)
    R1: float  # 机壳到中间节点
    R2: float  # 中间节点到线圈
    R3: float  # 环境到节点 3
    R4: float  # 节点 3 到环境
    
    # 热容 (J/°C)
    C1: float  # 中间节点热容
    C2: float  # 线圈热容
    C3: float  # 节点 3 热容
    
    # 初始条件
    T_case_init: float = 25.0  # 机壳初始温度 (°C)
    T_amb: float = 25.0        # 环境温度 (°C)
    
    @classmethod
    def default_motor(cls) -> 'ThermalParameters':
        """默认电机参数（需要根据实验数据辨识）"""
        return cls(
            R1=0.5,   # °C/W
            R2=0.3,   # °C/W
            R3=0.8,   # °C/W
            R4=0.6,   # °C/W
            C1=100,   # J/°C
            C2=50,    # J/°C
            C3=80,    # J/°C
        )


class LumpedThermalModel:
    """
    集总参数热路模型
    
    状态变量: [T_1, T_coil, T_3]
    输入：损耗功率 P_loss (W)
    """
    
    def __init__(self, params: ThermalParameters):
        self.p = params
        self.n_states = 3  # T_1, T_coil, T_3
        
    def state_derivative(self, t: float, x: np.ndarray, P_loss: float) -> np.ndarray:
        """
        计算状态导数 dx/dt
        
        热路方程:
        C_1 * dT_1/dt = (T_case - T_1)/R_1 + (T_coil - T_1)/R_2
        C_2 * dT_coil/dt = (T_1 - T_coil)/R_2 + P_loss
        C_3 * dT_3/dt = (T_amb - T_3)/R_3 + (P_loss - T_3)/R_4
        
        注意：这里 T_case 作为边界条件，T_coil 是主要关注节点
        """
        T_1, T_coil, T_3 = x
        
        # 机壳温度（可能随时间变化）
        if callable(self.p.T_case_init):
            T_case = self.p.T_case_init(t)
        else:
            T_case = self.p.T_case_init
        
        T_amb = self.p.T_amb
        
        # 状态方程
        dT_1 = ((T_case - T_1) / self.p.R1 + (T_coil - T_1) / self.p.R2) / self.p.C1
        dT_coil = ((T_1 - T_coil) / self.p.R2 + P_loss) / self.p.C2
        dT_3 = ((T_amb - T_3) / self.p.R3 + (P_loss - T_3) / self.p.R4) / self.p.C3
        
        return np.array([dT_1, dT_coil, dT_3])
    
    def simulate(self, t_span: Tuple[float, float], P_loss: float,
                 x0: Optional[np.ndarray] = None, n_points: int = 1000) -> dict:
        """
        仿真热路动态响应
        
        Args:
            t_span: (t_start, t_end) 时间范围 (s)
            P_loss: 损耗功率 (W)，可以是常数或函数 P_loss(t)
            x0: 初始状态 [T_1, T_coil, T_3]，默认从环境温度开始
            n_points: 输出点数
            
        Returns:
            包含时间、温度响应的字典
        """
        if x0 is None:
            x0 = np.array([self.p.T_case_init, self.p.T_amb, self.p.T_amb])
        
        # 处理 P_loss（常数或时变）
        if callable(P_loss):
            P_loss_func = P_loss
        else:
            P_loss_func = lambda t: P_loss
        
        def ode_func(t, x):
            return self.state_derivative(t, x, P_loss_func(t))
        
        # 求解
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(ode_func, t_span, x0, t_eval=t_eval, method='RK45')
        
        # 提取结果
        T_1 = sol.y[0]
        T_coil = sol.y[1]
        T_3 = sol.y[2]
        
        # 计算机壳温度（如果是时变的）
        if callable(self.p.T_case_init):
            T_case = np.array([self.p.T_case_init(t) for t in sol.t])
        else:
            T_case = np.full_like(sol.t, self.p.T_case_init)
        
        return {
            't': sol.t,
            'T_1': T_1,
            'T_coil': T_coil,
            'T_3': T_3,
            'T_case': T_case,
            'T_amb': np.full_like(sol.t, self.p.T_amb),
        }
    
    def plot_response(self, result: dict, title: str = "热路模型动态响应") -> None:
        """绘制温度响应曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(result['t'], result['T_coil'], 'r-', linewidth=2, label='T_coil (线圈)')
        ax.plot(result['t'], result['T_1'], 'b--', linewidth=1.5, label='T_1 (中间节点)')
        ax.plot(result['t'], result['T_3'], 'g-.', linewidth=1.5, label='T_3 (节点 3)')
        ax.plot(result['t'], result['T_case'], 'k:', linewidth=1, label='T_case (机壳)')
        ax.axhline(y=self.p.T_amb, color='gray', linestyle='--', alpha=0.5, label=f'T_amb={self.p.T_amb}°C')
        
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('温度 (°C)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def estimate_steady_state(P_loss: float, params: ThermalParameters) -> dict:
    """
    计算稳态温度（t→∞）
    
    稳态时 dT/dt = 0，可以解析求解
    """
    T_amb = params.T_amb
    T_case = params.T_case_init if not callable(params.T_case_init) else T_amb
    
    # 简化估算：稳态时线圈温升 ≈ P_loss * (R1 + R2)
    # 这是一个粗略估计，精确值需要解代数方程
    T_coil_ss = T_case + P_loss * (params.R1 + params.R2)
    T_1_ss = T_case + P_loss * params.R1
    T_3_ss = T_amb + P_loss * params.R3  # 简化
    
    return {
        'T_coil_ss': T_coil_ss,
        'T_1_ss': T_1_ss,
        'T_3_ss': T_3_ss,
        'temp_rise_coil': T_coil_ss - T_amb,
    }


if __name__ == "__main__":
    # 示例：仿真阶跃损耗下的温度响应
    params = ThermalParameters.default_motor()
    model = LumpedThermalModel(params)
    
    # 阶跃损耗：100W
    P_loss = 100  # W
    
    # 仿真 0-500s
    result = model.simulate((0, 500), P_loss)
    
    # 计算稳态
    ss = estimate_steady_state(P_loss, params)
    print("=" * 50)
    print("热路模型仿真结果")
    print("=" * 50)
    print(f"损耗功率：{P_loss} W")
    print(f"稳态线圈温度：{ss['T_coil_ss']:.2f} °C")
    print(f"稳态线圈温升：{ss['temp_rise_coil']:.2f} °C")
    print(f"500s 时线圈温度：{result['T_coil'][-1]:.2f} °C")
    
    # 绘图
    model.plot_response(result, f"阶跃损耗 {P_loss}W 下的温度响应")
