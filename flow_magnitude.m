% This file is to calculate the flow magnitude
%% Load data
% This data set cannot be made public. If you need to access, please contact the authors.
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


%% Plot the flow magnitude
%find the maximum value in each flow
a = max(data);
% plot the maximum value in each flow with the flow index
plot(a)
ylabel('Magnitude','FontSize', 20);
xlabel('Flow index','FontSize', 20);
%title('Flow magnitude','FontSize', 22);
xlim([0, n]);
% Enlarge the size of xtick and ytick labels
set(gca, 'FontSize', 18); % Adjust FontSize as needed for both x and y ticks

%% Plot the top 2 flows's magnitude and the last 2 non-zero flow magnitude

% load flows' name
flowname = Flowinfo(flowInd,2);


% Find the indices of the top 2 and last 2 non-zero values
sorted_data = sort(a(a > 0), 'descend'); % Sort non-zero values in descending order
top_2_values = sorted_data(1:2);
last_2_values = sorted_data(end-1:end);

% Find the corresponding flow indices for the top 2 non-zero values
top_2_indices = find(a == top_2_values(1) | a == top_2_values(2));

% Find the corresponding flow indices for the last 2 non-zero values
last_2_indices = find(a == last_2_values(1) | a == last_2_values(2));

% Extract the top 2 and last 2 flow names
top_1_flowname = flowname(top_2_indices(1))
top_2_flowname = flowname(top_2_indices(2))
last_1_flowname = flowname(last_2_indices(1))
last_2_flowname = flowname(last_2_indices(2))

% Create a bar graph with the selected values and adjust BarWidth and BarSpacing
bar([top_2_values, NaN, last_2_values]);
box off;

% Add annotations above each bar
% Modify the annotations to replace NaN with "......"
annotations = [top_2_values, NaN, last_2_values];
for i = 1:numel(annotations)
    if ~isnan(annotations(i))
        text(i, annotations(i), num2str(annotations(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom','FontSize', 16)
    end
end

% Set the title and axis labels
title('Top 2 and Last 2 Flow Magnitude', 'FontSize', 18);
% Narrow the space between the y-axis and the first bar
xlim([0.5, 5.5]);
set(gca, 'FontSize', 12); % Adjust FontSize as needed
% Hide y-axis ticks
set(gca, 'YTickLabel', []);
set(gca, 'xTickLabel', []);





%% corresponding process name
processname=Processinfo(procInd,1);
% 初始化存储行索引的变量
top_2_proc_indices = zeros(1, 2);
last_2_proc_indices = zeros(1, 2);

% 遍历 data 矩阵中的每一行，找到确切的索引
for i = 1:size(data, 1)
    % 检查当前行是否包含 top_2_values(1) 并且还未找到索引
    if any(data(i, :) == top_2_values(1)) && top_2_proc_indices(1) == 0
        top_2_proc_indices(1) = i;
    end
    % 检查当前行是否包含 top_2_values(2) 并且还未找到索引
    if any(data(i, :) == top_2_values(2)) && top_2_proc_indices(2) == 0
        top_2_proc_indices(2) = i;
    end
    % 检查当前行是否包含 last_2_values(1) 并且还未找到索引
    if any(data(i, :) == last_2_values(1)) && last_2_proc_indices(1) == 0
        last_2_proc_indices(1) = i;
    end
    % 检查当前行是否包含 last_2_values(2) 并且还未找到索引
    if any(data(i, :) == last_2_values(2)) && last_2_proc_indices(2) == 0
        last_2_proc_indices(2) = i;
    end
end

% Extract the top 2 and last 2 process names
top_1_procname = processname(top_2_proc_indices(1))
top_2_procname = processname(top_2_proc_indices(2))
last_1_procname = processname(last_2_proc_indices(1))
last_2_procname = processname(last_2_proc_indices(2))


%% 
a = max(data);
b = min(data(data > 0));
c=abs(a-b);
[c_sorted, indices] = sort(c, 'descend');

a1 = a(indices(1:10)).';
b1 = b(indices(1:10)).';