% this file is for heat map whith 1 random seed
load('big_data_seed1_new.mat')  % bigdata small range, r=1
%% full performance for test set
s=2; % the parameter to be changed to calculate different percentage of missing data
p=[0.01,0.05,0.1,0.2,0.5,0.8]; % define the percentage of missing data

q = 0.001:0.01:0.5;
l = 1:10;
sample_size = 2000;
% Set the seed for trainingsets

seed_range = 1:1;%1:20;
Perf1 = zeros (length(seed_range),10);
E = cell(length(seed_range));
C2 = cell(length(seed_range), 4);


for r = seed_range


MPE_mean = zeros(length(q),length(l));
MPE_median = zeros(length(q),length(l));
q= C1{r,1};
l = C1{r,2};

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
rng(r)
sample_ind = randperm(m,sample_size);

%data = data(sample_ind,:);
data(sample_ind,:) = [];

data_mi = data_mi(sample_ind,:);
data_re_testing = data_re(sample_ind,:);
data_re_training = data_re;
data_re_training(sample_ind,:)=[];

rng(r)
sample_ind = randperm(m,sample_size);

procInd_test = procInd(sample_ind);
procInd_train = procInd;
procInd_train(sample_ind) = [];
procName_test = Processinfo(procInd_test);
procName_train = Processinfo(procInd_train);
% missing-data's structure
data_mi_str = (data_mi~=0);
data_mi_str = data_mi_str.';


%E = cell(20);
%Perf2 = zeros (20,11);




% 按q索引

% Calculate the Minkowski distance between data_re and itself using parameter q(t)
D = pdist2(data_re_training,data_re_testing,'minkowski',q);% Minkowski
S=1.0./(1+D); 

% Initialize matrices for storing results
[B,I] = sort(S,1,'descend');% sort in each column, B is the value, I is the index of the value
%     B(1,:)=[]; % Remove the top row (self-comparison)
%     I(1,:)=[]; % Remove the top row (self-comparison)
E_1 = zeros (x,sample_size);

k=l;

for w = 1:sample_size % 按process索引
    E_1 (:,w)= data(I(1:k,w),mi_ind)'*B(1:k,w)./sum(B(1:k,w),1);%.*nonzero_ind(i,:)';
    E_1(isnan(E_1)) = 0; % 将所有NaN值替换为0
    MSE1(w) = sum((E_1 (:,w)'-data_mi(w,:)).^2)/x; % mse for all variables
    MSE2(w) = sum(((E_1 (:,w).*data_mi_str(:,w))'-data_mi(w,:)).^2)/sum(sum(data_mi_str)); % mse for non 0 variables
end
E{r} = E_1;

%另存变量
data_mi_1 = data_mi.';
E_2 = E_1;



%%%%%% 分类：0 -> 1， 1 -> 0/1 %%%%%%%%

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



    %%%%%% 分类：0 -> 1 & 1 -> 0; 1 -> 1 %%%%%%%%

%         % 计算 0 -> 1 的个数
%         condition1 = (data_mi_1 == 0) & (E_2 ~= 0);
%         count1 = sum(condition1(:)); % 计算满足条件的元素总数
% 
%         % 计算 1 -> 0 的个数
%         condition3 = (data_mi_1 ~= 0) & (E_2 == 0);
%         count2 = sum(condition3(:)); % 计算满足条件的元素总数        
% 
%         % 计算1 -> 1
%         condition4 = (data_mi_1 ~= 0)& (E_2 ~= 0);
%         a = sum(sum(condition4));
%         data_mi_1_values_2 = data_mi_1(condition4); % data_mi_1中满足条件的数据值
%         E_2_values_2 = E_2(condition4); % E_2中满足条件的数据值
%         difference_abs_2 = abs(data_mi_1_values_2 - E_2_values_2); % 计算差的绝对值
%         ratio_2 = difference_abs_2 ./ data_mi_1_values_2; % 计算差的绝对值与 data_mi_1_values 的比值
%         max_ratio_2 = max(ratio_2);% ratio 中的最大异常值
%         ad_ratio_2 = (sum(ratio_2)-max_ratio_2)/(length(data_mi_1_values_2)-1);
%         q99_2 = quantile(ratio_2, 0.98);
%         q_ratio_2 = ratio_2(ratio_2 <= q99_2);
% 
%         % q, k, 0->1&1->0%, 1->1%, max ratio, median ratio, mean ratio, adjusted ratio, quantile ratio, mean q_ratio, median mse for non 0, mean mse for non 0
%         Perf2(count, :) = [q(t), k, (count1+count2)/numel(data_mi_1), a/sum(data_mi_1 ~= 0, 'all'), max_ratio_2, median(ratio_2), mean(ratio_2), ad_ratio_2, mean(q_ratio_2), median(MSE2), mean(MSE2)];
[minValue, rowIndex] = min(Perf1(:, 8));

C2{r,1} =q;%q
C2{r,2} =k;%k
C2{r,3} =Perf1(r, 8);%percentage error
C2{r,4} =Perf1(r, 3);% 0-1
r
end

%%
load("big_data_test_new.mat")
%%
% 提取数据
y_labels = Perf1(:, 1);  % 第一列：y 轴标签
x_labels = Perf1(:, 2);  % 第二列：x 轴标签
values = Perf1(:, 8);    % 第八列：值

% 获取唯一的 y 和 x 标签，并按数值排序
unique_y_labels = unique(y_labels, 'stable');
unique_x_labels = unique(x_labels);
unique_x_labels = sort(unique_x_labels);  % 按数值排序

% 初始化热图矩阵
heatmap_data = nan(length(unique_y_labels), length(unique_x_labels));

% 填充热图矩阵
for i = 1:size(Perf1, 1)
    y_idx = find(unique_y_labels == y_labels(i));
    x_idx = find(unique_x_labels == x_labels(i));
    heatmap_data(y_idx, x_idx) = values(i);
end

% 绘制热图
figure;
imagesc(heatmap_data);
colorbar;

% 设置轴标签
set(gca, 'XTick', 1:length(unique_x_labels), 'XTickLabel', unique_x_labels);
set(gca, 'YTick', 1:length(unique_y_labels), 'YTickLabel', unique_y_labels);
set(gca, 'YDir', 'normal');
% 设置标题和轴标签
title('My Heatmap');
xlabel('k');
ylabel('q');


% 用红色标注一个点
hold on;
% 假设要标注的位置
[minValue, rowIndex] = min(Perf1(:, 8));
x_label = Perf1(rowIndex,2);  % 要标注点的 x 轴标签值
y_label = Perf1(rowIndex,1);  % 要标注点的 y 轴标签值

% 找到对应的索引
x_idx = find(unique_x_labels == x_label);
y_idx = find(unique_y_labels == y_label);

% 标注点
% 标注实心点

plot(x_idx, y_idx, 'r.', 'MarkerSize', 20); % 'r.' 表示实心红色点，MarkerSize 控制点的大小
%% find example
strMatrix = strings(size(I));

% 循环遍历每个位置，并填充新的字符串矩阵
for i = 1:size(I, 1)
    for j = 1:size(I, 2)
        strMatrix(i, j) = procName_train{I(i, j)};
    end
end


