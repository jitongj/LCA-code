% This file is to calculate the ISIC classification name
%%
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


%% Load data

% import 'Processinfo.xlsx' as string

sheetName = 'activity overview'; % Replace with the name of the sheet you want to load

% Load the data from the specified sheet
tbl = readtable('activity_overview_for_users_3.1_default.xlsx', 'Sheet', sheetName);

activityname = string(tbl.activityName);
geo = string(tbl.geography);
productname = string(tbl.productName);
isicall =string(tbl.ISICClassificationNumber);
isicall=str2double(isicall);
processname=Processinfo(procInd,1);
isicclass_name = string(tbl.ISICClassificationName);

%% match the isic name with our preprocessed data
isicnumber=[];
for i = 1:m
    str1 = string(processname(i,1));
    for j = 1:11332
        a = productname(j,1);
        b = geo(j,1);
        c = activityname(j,1);
        str2 =  append(a, '//[', b, '] ', c);
        if strcmp(str1, str2)
            isicnumber(end+1) = isicall(j,1);
        end
    end
end


%% Calculate the frequency of each isic and sort them

% Calculate the frequency of each unique value
uniqueValues = unique(isicnumber);
valueCounts = zeros(size(uniqueValues));

for i = 1:length(uniqueValues)
    valueCounts(i) = sum(isicnumber == uniqueValues(i));
end

% Sort the unique values by frequency in descending order
[sortedCounts, sortedIndices] = sort(valueCounts, 'descend');
sortedValues = uniqueValues(sortedIndices);
%% Find the top 10 percentile isic
% Calculate the top 10 percentile threshold
percentile = 98;
threshold = prctile(sortedCounts, percentile);

% Select isic index with counts above the threshold
selectedInd = sortedValues(sortedCounts > threshold);
selectedCounts = sortedCounts(sortedCounts > threshold);

% Find the corresponding name of each index
isic_name=strings(1, length(selectedInd));
for i = 1:length(selectedInd)
    b=find(isicall==selectedInd(1,i));
    b=b(1);
    isic_name(i) = isicclass_name(b);
end



%% Create a bar graph of isic

% Reset the color map to default
colormap('default');

% Create a bar graph with custom colors
bar(selectedCounts, 'FaceColor', 'flat');

% Set the axis labels
xticks(1:length(selectedCounts));
xticklabels(isic_name);
xtickangle(45);

% Label the axes and title
xlabel('ISIC classification name')
ylabel('Process counts');
title('# of processes per ISIC classification (only top 2 percentile of all data shown here)', 'FontSize', 12, 'Units', 'normalized', 'Position', [0.5, 1.04]);
for i = 1:numel(selectedCounts)
    text(i, selectedCounts(i), num2str(selectedCounts(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% Set custom colors for individual bars
%colormap(customColors);