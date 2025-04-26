% This file is to calculate the flow frequency
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

% normalized based on flow
maxValues = max(data, [], 1);  % 找到每列的最大值，1表示按列操作
data = data ./ maxValues;

% use data's structure to calculate frequency
data_str = (data~=0);
H_1 = data_str.'; % missing flow * process

%% plot frequency
plot(sum(H_1,2))
title("Flow frequency",'FontSize', 22);
xlabel('Flow index','FontSize', 20);
ylabel('Apperance counts','FontSize', 20);
xlim([0, n]);
set(gca, 'FontSize', 18); % Adjust FontSize as needed for both x and y ticks

%% Get the top 5 flows
a = sum(H_1,2);
[sorted_values, sorted_indices] = sort(a, 'descend');
top_5_values = sorted_values(1:5);
top_5_indices = sorted_indices(1:5);



flowname=Flowinfo(flowInd,1:2);
top_5_flowname = flowname(top_5_indices,1:2);

% Initialize the combined column
combined_col = strings(size(top_5_flowname, 1), 1);

% Loop through each row
for i = 1:size(top_5_flowname, 1)
    % Extract the row
    row = top_5_flowname(i, :);
    % Filter out empty strings
    filtered_row = row(row ~= "");
    % Join non-empty elements with commas
    combined_col(i) = join(filtered_row, ", ");
end


%% Plot the top 5 flows with its names and values
bar(top_5_values);
box off;
yticks(0:1000:1000);
set(gca, 'FontSize', 12);
for i = 1:numel(top_5_values)
    text(i, top_5_values(i), num2str(top_5_values(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% Set the axis labels and title
xticks(1:5);
xticklabels(string(combined_col));
xtickangle(45);
title("Top 5 flow frequency")
% Adjust the font size
set(gca, 'FontSize', 12);
% Hide y-axis ticks
set(gca, 'YTickLabel', []);




