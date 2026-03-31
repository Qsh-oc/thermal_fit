# 电机热路参数辨识与绕组温度预测

高爆发永磁力矩电机 - 绕组温度预测模型

## 📋 项目说明

本项目针对**高爆发永磁力矩电机**在爆发工况（高过载）下的绕组温度预测问题，基于集总参数热路模型，实现：

- ✅ 热路参数自动辨识（R1, R2, C1, C2）
- ✅ 绕组温度动态预测
- ✅ 预测精度验证（目标：误差 < ±5°C）
- ✅ 可视化分析工具

## 🔧 文件说明

| 文件 | 说明 |
|------|------|
| `thermal_identification.m` | MATLAB 参数辨识主程序 |
| `thermal_identification.py` | Python 参数辨识程序（无需 MATLAB） |
| `thermal_predict.py` | 温度预测工具 |
| `load_data.py` | 实验数据加载模块 |
| `thermal_model.py` | 热路模型仿真 |
| `temp_data.mat` | 温升实验数据（需用户提供） |
| `辨识热路.pdf` | 热路模型图 |
| `热平衡方程与参数辨识说明.md` | 理论推导文档 |

## 📐 热路模型结构

```
T_case ── R_1 ── T_1 ── R_2 ── T_coil
                  │           │
                 C_1         C_2
                  │           │
                 GND         GND
```

### 热平衡方程

**节点 1 (T_1):**
$$C_1\frac{dT_1}{dt} = \frac{T_{case} - T_1}{R_1} + \frac{T_{coil} - T_1}{R_2}$$

**节点 2 (T_coil):**
$$C_2\frac{dT_{coil}}{dt} = \frac{T_1 - T_{coil}}{R_2} + J_{loss}$$

### 参数物理意义

| 参数 | 物理意义 | 单位 |
|------|---------|------|
| R1 | 机壳→中间节点热阻 | °C/W |
| R2 | 中间节点→线圈热阻 | °C/W |
| C1 | 中间节点热容 | J/°C |
| C2 | 线圈热容 | J/°C |

## 🚀 使用方法

### 方法 1: Python（推荐）

```bash
# 1. 准备实验数据
# 将 temp_data.mat 放在项目目录

# 2. 运行参数辨识
python thermal_identification.py

# 3. 温度预测
python thermal_predict.py --J_loss 500 --T_case 35 --duration 300
```

### 方法 2: MATLAB

```matlab
% 1. 准备实验数据
% 将 temp_data.mat 放在项目目录

% 2. 运行辨识程序
thermal_identification

% 3. 查看结果
% 辨识参数保存在 identified_parameters.mat
% 结果图保存在 parameter_identification_result.png
```

## 📊 实验数据格式

`temp_data.mat` 应包含以下结构：

```matlab
con1: 工况 1 (J_loss = 652W)
  - time: 时间数组 (s)
  - case: 机壳温度 (°C)
  - coilF: 绕组前端温度 (°C)
  - coilB: 绕组后端温度 (°C)
  - coilM: 绕组中部温度 (°C)

con2: 工况 2 (J_loss = 452W)
  - 结构同 con1
```

绕组温度计算：`T_coil = (coilF + coilB + coilM) / 3`

## 📈 输出结果

### 辨识参数
- `identified_parameters.json` (Python)
- `identified_parameters.mat` (MATLAB)

### 可视化
- `parameter_identification_result.png` - 辨识结果对比图
- `prediction_J*W.png` - 温度预测图

### 精度指标
- RMSE: 均方根误差
- MAE: 平均绝对误差
- MaxError: 最大绝对误差

**目标：MaxError < ±5°C**

## 🔍 预测工具参数

```bash
python thermal_predict.py \
    --J_loss 500 \        # 损耗功率 (W)
    --T_case 35 \         # 机壳温度 (°C)
    --duration 300 \      # 预测时长 (s)
    --T_init 25           # 初始温度 (°C)
```

## ⚠️ 注意事项

1. **数据质量**: 实验数据应包含完整的温升过程（从初始到稳态）
2. **工况覆盖**: 建议至少 2 个不同功率工况用于辨识
3. **机壳温度**: 预测时需提供实时机壳温度（可测量）
4. **绝缘等级**: 预测温度不应超过电机绝缘等级限制
   - F 级：155°C
   - H 级：180°C

## 📝 待办事项

- [ ] 添加更多工况数据验证
- [ ] 考虑变工况（J_loss 时变）预测
- [ ] 添加参数敏感性分析
- [ ] 导出 Simulink 模型

## 📄 License

MIT License

## 👤 作者

Qsh-oc <qsh.cas@gmail.com>

---

**高爆发永磁力矩电机研究组**
