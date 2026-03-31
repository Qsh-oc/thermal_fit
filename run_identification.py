#!/usr/bin/env python3
"""
快速运行参数辨识 - 降采样版本
"""

import numpy as np
import scipy.io
from pathlib import Path
import json

# 加载数据
mat_path = Path(__file__).parent / 'temp_data.mat'
data = scipy.io.loadmat(str(mat_path))

def extract_data(con):
    """提取数据并降采样"""
    if con.ndim == 2 and con.shape == (1, 1):
        con = con[0, 0]
    
    time = con['time'].flatten()
    case = con['case'].flatten()
    coilF = con['coilF'].flatten()
    coilB = con['coilB'].flatten()
    coilM = con['coilM'].flatten()
    
    # 降采样：每 10 个点取 1 个
    step = 10
    return {
        'time': time[::step],
        'T_case': case[::step],
        'coilF': coilF[::step],
        'coilB': coilB[::step],
        'coilM': coilM[::step],
    }

con1 = extract_data(data['con1'])
con2 = extract_data(data['con2'])

con1['T_coil'] = (con1['coilF'] + con1['coilB'] + con1['coilM']) / 3
con2['T_coil'] = (con2['coilF'] + con2['coilB'] + con2['coilM']) / 3

con1['J_loss'] = 652
con2['J_loss'] = 452
con1['T_amb'] = 14
con2['T_amb'] = 14

print(f"工况 1: {len(con1['time'])} 个数据点")
print(f"工况 2: {len(con2['time'])} 个数据点")

# 导入辨识模块
import sys
sys.path.insert(0, str(Path(__file__).parent))
from thermal_identification import identify_parameters, validate_model, plot_results, save_results

# 参数辨识
params = identify_parameters(con1, con2, method='hybrid')

# 模型验证
validation = validate_model(params, con1, con2)

# 可视化
plot_results(params, con1, con2, validation, 
             save_path=Path(__file__).parent / 'parameter_identification_result.png')

# 保存结果
save_results(params, validation, con1, con2)

print("\n辨识完成！")
