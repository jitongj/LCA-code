% This file is to check the performance of the 50 different subset (1200 processes) of testsets
%%
% Set the seed for trainingsets
seed_range = 1:50;
s=2; % the parameter to be changed to calculate different percentage of missing data
p=[0.01,0.05,0.1,0.2,0.5,0.8]; % define the percentage of missing data
sample_size = 2000;
for r = seed_range

    ecodataoriginal1 = readmatrix('rawdata.csv');
    ecodataoriginal = abs(ecodataoriginal1); 
    % 假设ecodataoriginal是原始数据，allProcs和allFlows分别代表过程和流程的索引
    data = ecodataoriginal.'; % 转置以便操作
    allFlows = 1:size(ecodataoriginal, 1);
    % orginal process index
    allProcs = 1:size(ecodataoriginal, 2);
    
    
    % 初始化过程和流程的索引
    procInd = allProcs;
    flowInd = allFlows;
    
    % 先进行行的唯一性处理
    [dataUnique, ia, ~] = unique(data, 'rows', 'stable');
    data = dataUnique; % 更新data为只包含唯一行的矩阵
    procInd = procInd(ia); % 根据唯一行的原始索引更新过程索引
    
    % 标记，用于判断是否需要继续循环
    changesMade = true;
    count1 = 0;
    count2 = 0;
    while changesMade
        changesMade = false; % 假设本轮没有改动
        
        % 步骤1: 删除全是0或只有一个非零元素的行(process)
        rowSum = sum(data ~= 0, 2); % 计算每行非零元素的数量
        toKeepRows = rowSum > 1; % 需要保留的行标记
        if any(~toKeepRows) % 如果有行被删除
            data = data(toKeepRows, :); % 保留需要的行
            procInd = procInd(toKeepRows); % 更新过程索引
            changesMade = true; % 标记改动
            count1 = count1 + 1;
        end
        
        % 步骤2: 删除全是0或只有一个非零元素的列(flow)
        colSum = sum(data ~= 0, 1); % 计算每列非零元素的数量
        toKeepCols = colSum > 1; % 需要保留的列标记
        if any(~toKeepCols) % 如果有列被删除
            data = data(:, toKeepCols); % 保留需要的列
            flowInd = flowInd(toKeepCols); % 更新流程索引
            changesMade = true; % 标记改动
            count2 = count2 + 1;
        end
    end
    
    
    [m,n]=size(data); % process*flow
    
    % normalized based on flow
    maxValues = max(data, [], 1);  % 找到每列的最大值，1表示按列操作
    data = data ./ maxValues;
    
    
    x=ceil(p(s)*n); % missing number of
    rng default
    mi_ind = randperm(n,x);% missing flows
    data_mi=data(:,mi_ind);
    data_re=data;
    data_re(:,mi_ind)=[];% Remove data at missing data positions
    
    % Choose 2046 processes for trainingset and 200 processes for testset
    %sample_size = 2000;
    rng default
    sample_ind = randperm(m,sample_size);
    rng(r)
    sample_ind_seed = sample_ind(randperm(sample_size,1200));


    %data = data(sample_ind,:);
    data(sample_ind,:) = [];
    
    data_mi = data_mi(sample_ind_seed,:);
    data_re_testing = data_re(sample_ind_seed,:);
    data_re_training = data_re;
    data_re_training(sample_ind,:)=[];
    

    % missing-data's structure
    data_mi_str = (data_mi~=0);
    data_mi_str = data_mi_str.';
    q=0.075;
    k=2;

    D = pdist2(data_re_training,data_re_testing,'minkowski',q);% Minkowski
    S=1.0./(1+D); 

% Initialize matrices for storing results
    [B,I] = sort(S,1,'descend');% sort in each column, B is the value, I is the index of the value
%     B(1,:)=[]; % Remove the top row (self-comparison)
%     I(1,:)=[]; % Remove the top row (self-comparison)
    E_1 = zeros (x,1200);



    for w = 1:1200 % 按process索引
        E_1 (:,w)= data(I(1:k,w),mi_ind)'*B(1:k,w)./sum(B(1:k,w),1);%.*nonzero_ind(i,:)';
        E_1(isnan(E_1)) = 0; % 将所有NaN值替换为0
        MSE1(w) = sum((E_1 (:,w)'-data_mi(w,:)).^2)/x; % mse for all variables
        MSE2(w) = sum(((E_1 (:,w).*data_mi_str(:,w))'-data_mi(w,:)).^2)/sum(sum(data_mi_str)); % mse for non 0 variables
    end

%另存变量
    data_mi_1 = data_mi.';
    E_2 = E_1;

    % 计算 0 -> 1 的个数
    condition1 = (data_mi_1 == 0) & (E_2 ~= 0);
    count1 = sum(condition1(:)); % 计算满足条件的元素总数
    
    % 计算1 -> 0/1
    condition2 = (data_mi_1 ~= 0);
    data_mi_1_values_1 = data_mi_1(condition2); % data_mi_1中满足条件的数据值
    E_2_values_1 = E_2(condition2); % E_2中满足条件的数据值
    difference_abs_1 = abs(data_mi_1_values_1 - E_2_values_1); % 计算差的绝对值
    ratio_1 = difference_abs_1 ./ data_mi_1_values_1; % 计算差的绝对值与 data_mi_1_values 的比值
    
    max_ratio_1 = max(ratio_1); % max ratio
    ad_ratio_1 = (sum(ratio_1)-max_ratio_1)/(length(data_mi_1_values_1)-1); % adjusted ratio: drop the max
    q99_1 = quantile(ratio_1, 0.98); 
    q_ratio_1 = ratio_1(ratio_1 <= q99_1); % 98 quantile ratio
    
    % q, k, 0->1%, max ratio, median ratio, mean ratio, adjust ratio, quantile ration, median mse for all, mean mse for all
    Perf1(r, :) = [q, k, count1/sum(data_mi_1 == 0, 'all'), max_ratio_1, median(ratio_1), mean(ratio_1), ad_ratio_1, mean(q_ratio_1), median(MSE1), mean(MSE1)];
    
    [minValue, rowIndex] = min(Perf1(:, 8));

    C2{r,1} =q;%q
    C2{r,2} =k;%k
    C2{r,3} =Perf1(r, 8);%percentage error
    C2{r,4} =Perf1(r, 3);% 0-1

    r
end



%% Plot the Median MPE of each subset with an average line
y = C2{:,3};
scatter(seed_range, y)
hold on
line([min(seed_range), max(seed_range)], [y, y], 'Color', 'r', 'LineStyle', '--');
hold off

title('5% missing: median MPE of 100 sub-testset, q=0.19 k=3', 'FontSize', 18, 'Units', 'normalized', 'Position', [0.5, 1.04]);
xlabel('Random selection of 300 processes', 'FontSize', 16);
ylabel('Median MPE', 'FontSize', 16);
set(gca, 'FontSize', 14);

%% Load data to save time
% load('randomsplit_testing.mat')
%%

% 定义seed_range
seed_range = 1:50;

% 提取C2{:,3}中的值
values = cell2mat(C2(:, 3));

% 计算均值
values_mean = mean(values);

% 创建图形并设置大小
figure('Position', [100, 100, 1000, 800]);

% 绘制散点图
scatter(seed_range, values, 80, 'filled', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'LineWidth', 2.5);
hold on

% 绘制均值线
line([min(seed_range), max(seed_range)], [values_mean, values_mean], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2.5);
hold off

% 设置图表属性
title('5% missing: Percentage Error of 50 sub-testset, q=0.075 k=2', 'FontSize', 22, 'FontWeight', 'bold');
xlabel('Random selection of 1200 processes', 'FontSize', 24);
ylabel('Percentage Error', 'FontSize', 24);

% 设置坐标轴刻度字体大小
set(gca, 'FontSize', 18);

% 显示网格线
grid on;


% 显示全边框
box on;
% set(gca, 'BoxStyle', 'full');

% 调整坐标轴范围，使点不会太靠近边缘
axis tight;
ax = axis;
axis([ax(1)-0.1*(ax(2)-ax(1)) ax(2)+0.1*(ax(2)-ax(1)) ax(3)-0.1*(ax(4)-ax(3)) ax(4)+0.1*(ax(4)-ax(3))]);