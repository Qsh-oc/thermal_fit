%% =========================================================================
%  电机热路参数辨识程序
%  高爆发永磁力矩电机 - 绕组温度预测
%  
%  功能：
%  1. 加载温升实验数据
%  2. 辨识热路参数 (R1, R2, C1, C2)
%  3. 模型验证与精度评估
%  4. 可视化预测结果
%
%  要求：温升预测误差 < ±5°C
%  =========================================================================

clear; clc; close all;

%% ==================== 1. 加载实验数据 ====================
fprintf('=== 加载实验数据 ===\n');

% 加载 temp_data.mat
data = load('temp_data.mat');

% 提取工况 1 数据 (J_loss = 652W)
time1 = data.con1.time(:);
T_case1 = data.con1.case(:);
T_coil1 = (data.con1.coilF + data.con1.coilB + data.con1.coilM) / 3;  % 平均绕组温度
J_loss1 = 652;  % W

% 提取工况 2 数据 (J_loss = 452W)
time2 = data.con2.time(:);
T_case2 = data.con2.case(:);
T_coil2 = (data.con2.coilF + data.con2.coilB + data.con2.coilM) / 3;  % 平均绕组温度
J_loss2 = 452;  % W

fprintf('工况 1: %d 个数据点，J_loss = %dW\n', length(time1), J_loss1);
fprintf('工况 2: %d 个数据点，J_loss = %dW\n', length(time2), J_loss2);

%% ==================== 2. 稳态参数初步估算 ====================
fprintf('\n=== 稳态参数估算 ===\n');

% 取最后 10% 数据作为稳态值
steady_idx1 = round(length(time1)*0.9):length(time1);
steady_idx2 = round(length(time2)*0.9):length(time2);

T_coil_ss1 = mean(T_coil1(steady_idx1));
T_case_ss1 = mean(T_case1(steady_idx1));
T_coil_ss2 = mean(T_coil2(steady_idx2));
T_case_ss2 = mean(T_case2(steady_idx2));

% 计算总热阻 R_total = R1 + R2
R_total1 = (T_coil_ss1 - T_case_ss1) / J_loss1;
R_total2 = (T_coil_ss2 - T_case_ss2) / J_loss2;
R_total = (R_total1 + R_total2) / 2;

fprintf('工况 1 稳态：T_coil = %.2f°C, T_case = %.2f°C, ΔT = %.2f°C\n', ...
    T_coil_ss1, T_case_ss1, T_coil_ss1 - T_case_ss1);
fprintf('工况 2 稳态：T_coil = %.2f°C, T_case = %.2f°C, ΔT = %.2f°C\n', ...
    T_coil_ss2, T_case_ss2, T_coil_ss2 - T_case_ss2);
fprintf('估算总热阻 R_total = %.6f °C/W\n', R_total);

%% ==================== 3. 定义热路模型 ====================
% 状态空间模型：dx/dt = A*x + B*u
% x = [T_1; T_coil], u = [T_case; J_loss]

thermal_model = @(params, t, x, T_case, J_loss) thermal_ode(params, t, x, T_case, J_loss);

%% ==================== 4. 参数辨识（优化方法） ====================
fprintf('\n=== 开始参数辨识 ===\n');

% 待辨识参数：[R1, R2, C1, C2]
% 初始猜测（基于稳态估算和典型值）
R1_init = R_total * 0.6;  % R1 约占总热阻的 60%
R2_init = R_total * 0.4;  % R2 约占 40%
C1_init = 150;  % J/°C (典型值)
C2_init = 80;   % J/°C (典型值)

params0 = [R1_init, R2_init, C1_init, C2_init];

% 参数边界
lb = [0.001, 0.001, 1, 1];      % 下界
ub = [10, 10, 10000, 10000];    % 上界

% 定义目标函数（最小化预测误差）
objective = @(params) calc_objective(params, time1, T_case1, J_loss1, T_coil1, ...
                                              time2, T_case2, J_loss2, T_coil2);

% 使用 fmincon 进行约束优化
options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'sqp', ...
    'MaxFunctionEvaluations', 10000, ...
    'MaxIterations', 1000, ...
    'FunctionTolerance', 1e-6, ...
    'StepTolerance', 1e-6);

fprintf('初始参数：R1=%.4f, R2=%.4f, C1=%.2f, C2=%.2f\n', params0(1), params0(2), params0(3), params0(4));

[params_opt, fval] = fmincon(objective, params0, [], [], [], [], lb, ub, [], options);

R1 = params_opt(1);
R2 = params_opt(2);
C1 = params_opt(3);
C2 = params_opt(4);

fprintf('\n=== 辨识结果 ===\n');
fprintf('R1 = %.6f °C/W\n', R1);
fprintf('R2 = %.6f °C/W\n', R2);
fprintf('C1 = %.2f J/°C\n', C1);
fprintf('C2 = %.2f J/°C\n', C2);
fprintf('R_total = %.6f °C/W\n', R1 + R2);
fprintf('目标函数值：%.4f\n', fval);

%% ==================== 5. 模型验证 ====================
fprintf('\n=== 模型验证 ===\n');

% 用工况 1 数据验证
x0 = [T_case1(1); T_coil1(1)];  % 初始状态
[t_sim1, T_sim1] = simulate_thermal(params_opt, time1, T_case1, J_loss1, x0);

% 用工况 2 数据验证
x0_2 = [T_case2(1); T_coil2(1)];
[t_sim2, T_sim2] = simulate_thermal(params_opt, time2, T_case2, J_loss2, x0_2);

% 计算误差
error1 = T_sim1(:,2) - T_coil1;
error2 = T_sim2(:,2) - T_coil2;

RMSE1 = sqrt(mean(error1.^2));
RMSE2 = sqrt(mean(error2.^2));
MAE1 = mean(abs(error1));
MAE2 = mean(abs(error2));
MaxError1 = max(abs(error1));
MaxError2 = max(abs(error2));

fprintf('工况 1 验证:\n');
fprintf('  RMSE = %.3f °C, MAE = %.3f °C, MaxError = %.3f °C\n', RMSE1, MAE1, MaxError1);
fprintf('工况 2 验证:\n');
fprintf('  RMSE = %.3f °C, MAE = %.3f °C, MaxError = %.3f °C\n', RMSE2, MAE2, MaxError2);

% 检查是否满足精度要求
if MaxError1 <= 5 && MaxError2 <= 5
    fprintf('\n✅ 精度要求满足！最大误差 < ±5°C\n');
else
    fprintf('\n⚠️  精度要求未完全满足，最大误差：%.2f°C\n', max(MaxError1, MaxError2));
end

%% ==================== 6. 可视化结果 ====================
figure('Position', [100, 100, 1200, 800]);

% 子图 1: 工况 1 温度对比
subplot(2, 2, 1);
plot(time1, T_coil1, 'bo-', 'LineWidth', 1.5, 'DisplayName', '实验值');
hold on;
plot(t_sim1, T_sim1(:,2), 'r--', 'LineWidth', 2, 'DisplayName', '预测值');
xlabel('时间 (s)');
ylabel('温度 (°C)');
title(['工况 1 (J_loss = ' num2str(J_loss1) 'W) - 绕组温度']);
legend('Location', 'best');
grid on;
ax = gca;
ax.FontSize = 11;

% 子图 2: 工况 2 温度对比
subplot(2, 2, 2);
plot(time2, T_coil2, 'bo-', 'LineWidth', 1.5, 'DisplayName', '实验值');
hold on;
plot(t_sim2, T_sim2(:,2), 'r--', 'LineWidth', 2, 'DisplayName', '预测值');
xlabel('时间 (s)');
ylabel('温度 (°C)');
title(['工况 2 (J_loss = ' num2str(J_loss2) 'W) - 绕组温度']);
legend('Location', 'best');
grid on;
ax = gca;
ax.FontSize = 11;

% 子图 3: 工况 1 误差曲线
subplot(2, 2, 3);
plot(time1, error1, 'g-', 'LineWidth', 1.5);
hold on;
yline(5, 'r--', 'LineWidth', 1.5, 'DisplayName', '+5°C 边界');
yline(-5, 'r--', 'LineWidth', 1.5, 'DisplayName', '-5°C 边界');
yline(0, 'k:', 'LineWidth', 1);
xlabel('时间 (s)');
ylabel('预测误差 (°C)');
title(['工况 1 预测误差 (RMSE = ' num2str(RMSE1, '%.2f') '°C)']);
legend('Location', 'best');
grid on;
ax = gca;
ax.FontSize = 11;
ylim([max(-10, min(error1)*1.1), min(10, max(error1)*1.1)]);

% 子图 4: 工况 2 误差曲线
subplot(2, 2, 4);
plot(time2, error2, 'g-', 'LineWidth', 1.5);
hold on;
yline(5, 'r--', 'LineWidth', 1.5, 'DisplayName', '+5°C 边界');
yline(-5, 'r--', 'LineWidth', 1.5, 'DisplayName', '-5°C 边界');
yline(0, 'k:', 'LineWidth', 1);
xlabel('时间 (s)');
ylabel('预测误差 (°C)');
title(['工况 2 预测误差 (RMSE = ' num2str(RMSE2, '%.2f') '°C)']);
legend('Location', 'best');
grid on;
ax = gca;
ax.FontSize = 11;
ylim([max(-10, min(error2)*1.1), min(10, max(error2)*1.1)]);

sgtitle('电机热路参数辨识结果 - 绕组温度预测精度验证', 'FontSize', 14, 'FontWeight', 'bold');

% 保存图形
saveas(gcf, 'parameter_identification_result.png', 'png');
fprintf('\n结果图已保存：parameter_identification_result.png\n');

%% ==================== 7. 保存辨识参数 ====================
results.R1 = R1;
results.R2 = R2;
results.C1 = C1;
results.C2 = C2;
results.R_total = R1 + R2;
results.RMSE1 = RMSE1;
results.RMSE2 = RMSE2;
results.MAE1 = MAE1;
results.MAE2 = MAE2;
results.MaxError1 = MaxError1;
results.MaxError2 = MaxError2;
results.J_loss1 = J_loss1;
results.J_loss2 = J_loss2;

save('identified_parameters.mat', 'results');
fprintf('辨识参数已保存：identified_parameters.mat\n');

%% ==================== 8. 生成参数报告 ====================
fprintf('\n=== 辨识报告 ===\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('辨识参数:\n');
fprintf('  R1 = %.6f °C/W  (机壳→中间节点热阻)\n', R1);
fprintf('  R2 = %.6f °C/W  (中间节点→线圈热阻)\n', R2);
fprintf('  C1 = %.2f J/°C   (中间节点热容)\n', C1);
fprintf('  C2 = %.2f J/°C   (线圈热容)\n', C2);
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('模型精度:\n');
fprintf('  工况 1 (652W): RMSE = %.3f°C, 最大误差 = %.3f°C\n', RMSE1, MaxError1);
fprintf('  工况 2 (452W): RMSE = %.3f°C, 最大误差 = %.3f°C\n', RMSE2, MaxError2);
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('稳态特性:\n');
fprintf('  总热阻 R_total = %.6f °C/W\n', R1 + R2);
fprintf('  652W 稳态温升预测 = %.2f °C\n', (R1 + R2) * J_loss1);
fprintf('  452W 稳态温升预测 = %.2f °C\n', (R1 + R2) * J_loss2);
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

%% =========================================================================
%  辅助函数
%  =========================================================================

function dxdt = thermal_ode(params, ~, x, T_case, J_loss)
    % 热路 ODE 方程
    % params: [R1, R2, C1, C2]
    % x: [T_1; T_coil]
    
    R1 = params(1);
    R2 = params(2);
    C1 = params(3);
    C2 = params(4);
    
    T_1 = x(1);
    T_coil = x(2);
    
    % 热平衡方程
    dT_1 = ((T_case - T_1)/R1 + (T_coil - T_1)/R2) / C1;
    dT_coil = ((T_1 - T_coil)/R2 + J_loss) / C2;
    
    dxdt = [dT_1; dT_coil];
end

function [t_sim, T_sim] = simulate_thermal(params, t_exp, T_case_exp, J_loss, x0)
    % 仿真热路模型
    % 使用 ode45 求解，输出时间点与实验数据对齐
    
    t_sim = t_exp;
    T_case_interp = interp1(t_exp, T_case_exp, t_exp);  % 机壳温度插值
    
    % 创建带插值的 ODE 函数
    ode_func = @(t, x) thermal_ode(params, t, x, ...
        interp1(t_exp, T_case_exp, t, 'linear', 'extrap'), ...
        J_loss);
    
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
    [~, T_sim] = ode45(ode_func, t_sim, x0, options);
end

function obj = calc_objective(params, time1, T_case1, J_loss1, T_coil1, ...
                                     time2, T_case2, J_loss2, T_coil2)
    % 目标函数：最小化两个工况的预测误差平方和
    
    x0_1 = [T_case1(1); T_coil1(1)];
    x0_2 = [T_case2(1); T_coil2(1)];
    
    [~, T_sim1] = simulate_thermal(params, time1, T_case1, J_loss1, x0_1);
    [~, T_sim2] = simulate_thermal(params, time2, T_case2, J_loss2, x0_2);
    
    % 加权误差平方和
    error1 = T_sim1(:,2) - T_coil1;
    error2 = T_sim2(:,2) - T_coil2;
    
    obj = sum(error1.^2) + sum(error2.^2);
end
