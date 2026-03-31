#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电机热拟合实验数据加载模块
高爆发永磁力矩电机 - 温度测试数据处理
"""

import scipy.io
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ThermalData:
    """温度测试数据结构"""
    ambient_temp: float  # 环境温度 (°C)
    coil_front: float    # 前线圈温度 (°C)
    coil_back: float     # 后线圈温度 (°C)
    coil_middle: float   # 中间线圈温度 (°C)
    time: float          # 时间 (s)
    
    @property
    def temp_rise(self) -> Dict[str, float]:
        """计算各位置温升"""
        return {
            'coilF': self.coil_front - self.ambient_temp,
            'coilB': self.coil_back - self.ambient_temp,
            'coilM': self.coil_middle - self.ambient_temp,
        }
    
    @property
    def avg_coil_temp(self) -> float:
        """平均线圈温度"""
        return (self.coil_front + self.coil_back + self.coil_middle) / 3


def load_thermal_data(mat_path: str) -> Dict[str, ThermalData]:
    """
    加载 MATLAB 格式的温度测试数据
    
    Args:
        mat_path: .mat 文件路径
        
    Returns:
        包含 con1, con2 等工况的字典
    """
    data = scipy.io.loadmat(mat_path)
    result = {}
    
    for key in ['con1', 'con2']:
        if key in data:
            d = data[key][0, 0]
            result[key] = ThermalData(
                ambient_temp=float(d['case'][0, 0]),
                coil_front=float(d['coilF'][0, 0]),
                coil_back=float(d['coilB'][0, 0]),
                coil_middle=float(d['coilM'][0, 0]),
                time=float(d['time'][0, 0]),
            )
    
    return result


def print_data_summary(data: Dict[str, ThermalData]) -> None:
    """打印数据摘要"""
    print("=" * 60)
    print("电机温度测试数据摘要")
    print("=" * 60)
    
    for name, d in data.items():
        print(f"\n[{name}]")
        print(f"  环境温度：   {d.ambient_temp:7.2f} °C")
        print(f"  前线圈温度： {d.coil_front:7.2f} °C  (温升：{d.temp_rise['coilF']:6.2f} °C)")
        print(f"  后线圈温度： {d.coil_back:7.2f} °C  (温升：{d.temp_rise['coilB']:6.2f} °C)")
        print(f"  中间线圈温度：{d.coil_middle:7.2f} °C  (温升：{d.temp_rise['coilM']:6.2f} °C)")
        print(f"  平均线圈温度：{d.avg_coil_temp:7.2f} °C")
        print(f"  时间：       {d.time:7.2f} s")


if __name__ == "__main__":
    # 加载数据
    data_dir = Path(__file__).parent
    mat_file = data_dir / "experimental_data.mat"
    
    if mat_file.exists():
        data = load_thermal_data(str(mat_file))
        print_data_summary(data)
    else:
        print(f"未找到数据文件：{mat_file}")
