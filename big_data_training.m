% this file is for heat map with 1 random seed
%% full performance for training set
s=2; % the parameter to be changed to calculate different percentage of missing data
p=[0.01,0.05,0.1,0.2,0.5,0.8]; % define the percentage of missing data

q = 0.065:0.001:0.085;
l = 1:5;
%q = 0.001:0.01:0.5;
%l = 1:10;
sample_size = 2000;
% q=0.071;
% l = 2;
% 0.347

% Set the seed for trainingsets
%seed_range = 1:20;
seed_range = 1;
C1 = cell(length(seed_range), 4);


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


x=ceil(p(s)*n); % missing number of flow
rng default
mi_ind = randperm(n,x);
data_mi=data(:,mi_ind);
data_re=data;
data_re(:,mi_ind)=[];% Remove data at missing data positions
flowInd_mi = flowInd(mi_ind);
flowInd_re = flowInd;
flowInd_re(mi_ind) = [];
%flowName_mi = Flowinfo(flowInd_mi);
%flowName_re = Flowinfo(flowInd_re);


% Choose 2046 processes for trainingset and 200 processes for testset
%sample_size = 200;
rng(r)
sample_ind = randperm(m,sample_size);

data(sample_ind,:) = [];
data_mi(sample_ind,:)=[];
data_re(sample_ind,:)=[];
procInd_test = procInd(sample_ind);
procInd_train = procInd;
procInd_train(sample_ind) = [];

%procName_test = Processinfo(procInd_test);
%procName_train = Processinfo(procInd_train);



% missing-data's structure
data_mi_str = (data_mi~=0);
data_mi_str = data_mi_str.';


E = cell(size(q,2), length(l));
Perf1 = zeros (size(q,2)*length(l),10);
Perf2 = zeros (size(q,2)*length(l),11);


count = 1;
% 按q索引
for t = 1:size(q,2) 
     % Calculate the Minkowski distance between data_re and itself using parameter q(t)
    D = pdist2(data_re,data_re,'minkowski',q(t));% Minkowski
    S=1.0./(1+D); 

     % Initialize matrices for storing results
    [B,I] = sort(S,1,'descend');% sort in each column, B is the value, I is the index of the value
    B(1,:)=[]; % Remove the top row (self-comparison)
    I(1,:)=[]; % Remove the top row (self-comparison)
    E_1 = zeros (x,m-sample_size);


    for k = l % 按k索引
        for w = 1:size(data,1) % 按process索引
            E_1 (:,w)= data(I(1:k,w),mi_ind)'*B(1:k,w)./sum(B(1:k,w),1);%.*nonzero_ind(i,:)';
            E_1(isnan(E_1)) = 0; % 将所有NaN值替换为0
            MSE1(w) = sum((E_1 (:,w)'-data_mi(w,:)).^2)/x; % mse for all variables
            MSE2(w) = sum(((E_1 (:,w).*data_mi_str(:,w))'-data_mi(w,:)).^2)/sum(sum(data_mi_str)); % mse for non 0 variables
        end
        E{t,k} = E_1;

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
        Perf1(count, :) = [q(t), k, count1/sum(data_mi_1 == 0, 'all'), max_ratio_1, median(ratio_1), mean(ratio_1), ad_ratio_1, mean(q_ratio_1), median(MSE1), mean(MSE1)];




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
% 

        count = count+1;
    end
end

[minValue, rowIndex] = min(Perf1(:, 8));

C1{r,1} =Perf1(rowIndex,1);%q
C1{r,2} =Perf1(rowIndex,2);%k
C1{r,3} =minValue;%percentage error
C1{r,4} =Perf1(rowIndex,3);% 0-1
end
%%
% % Perf = rand(100, 4); % 示例：创建一个100x4的随机矩阵
% 
% % 提取X轴和Y轴的数据
% X = Perf2(:, 9); % 第四列作为X轴
% Y = Perf2(:, 4); % 第三列作为Y轴
% plot(X, Y, 'o'); % 使用点来标记每个点
% xlabel('mean q ratio error'); % 添加X轴标签
% ylabel('1-1 %'); % 添加Y轴标签
% 
% 
% %%
% % 提取X轴和Y轴的数据
% X = Perf1(:, 8); % 第四列作为X轴
% Y = Perf1(:, 3); % 第三列作为Y轴
% plot(X, Y, 'o'); % 使用点来标记每个点
% xlabel('mean q ratio error ( 1-1&1-0)'); % 添加X轴标签
% ylabel('0-1 %'); % 添加Y轴标签
% 
% 
% %%
% X = Perf1(:, 8);
% Y = Perf2(:, 9); % 
% plot(X, Y, 'o'); % 使用点来标记每个点
% xlabel('mean q ratio error (1-1&1-0)'); % 添加X轴标签
% ylabel('mean q ratio error (1-1)'); % 添加Y轴标签
% xlim([0.3, 0.35]);
% ylim([0.1, 0.25]);
% %%
% h(:,:) = [data_mi_1_values_1,E_2_values_1,ratio_1 ];
% % 假设Perf是你的矩阵
% % Perf = rand(100, 4); % 示例：创建一个100x4的随机矩阵
% 
% % 创建逻辑索引，标记第四列元素小于0.3的行
% index = Perf(:, 4) < 0.80;
% 
% % 使用逻辑索引筛选行
% newMatrix = Perf(index, :);
% 
% % 显示新的矩阵
% disp(newMatrix)
% %%
% % 根据newMatrix的第三列从小到大排序所有行
% sortedMatrix = sortrows(Perf, 8);
% 
% % 显示排序后的矩阵
% disp(sortedMatrix);
% 
% 
% %%
% % 筛选出横坐标小于1，纵坐标小于5*10^-3的点
% %indices = X < 0.5 & Y < 5.5e-3;
% indices = X < 0.5 & Y > 0.3;
% X_filtered = 1 - X(indices);
% Y_filtered = Y(indices);
% Perf_filtered = Perf2(indices, :);
% 
% % 绘制筛选后的点
% plot(X_filtered, Y_filtered, 'o');
% xlabel('mean q_ratio: accuracy'); % 添加X轴标签
% ylabel('1-1 %'); % 添加Y轴标签
% 
% % 为筛选后的每个点添加文本标签
% for i = 1:length(X)
%     % 提取第一和第二列的值
%     label = sprintf('(%0.3f, %d)', Perf1(i, 1), Perf1(i, 2));
%     % 在点旁边添加文本
%     text(X(i), Y(i), label, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
% end


%% plot
%load('panfeiyu_bigdata.mat') % bigdata small range, r=1
load('big_data_seed1_new.mat') % bigdata small range, r=1
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

% 绘制另一个较小的数据集，并细化其颜色显示
figure;

% 将 heatmap_data1 映射到 [0, 1] 范围内
mapped_data1 = (heatmap_data1 - min(heatmap_data1(:))) / (max(heatmap_data1(:)) - min(heatmap_data1(:)));

% 线性插值，将 heatmap_data1 的数据范围（0到0.3）映射到 colorbar 的深蓝到浅蓝部分
min_val = 0.3;
max_val = 0.4;
heatmap_data2 = min_val + mapped_data1 * (max_val - min_val);

% 使用新的数据绘制热图
imagesc(heatmap_data2);

% 保持 colormap 一致
colormap(parula);

% 设置 colorbar
colorbar;
caxis([0.3 1]);  % 保持 colorbar 范围为 [0.3 1]


% 设置轴标签
set(gca, 'XTick', 1:length(unique_x_labels), 'XTickLabel', unique_x_labels);
set(gca, 'YTick', 1:length(unique_y_labels), 'YTickLabel', unique_y_labels);
set(gca, 'YDir', 'normal');
% 设置标题和轴标签
title('Zoomed-in Heatmap: 5% missing');
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

%%
load('big_data_seed1_new.mat') % bigdata small range, r=1
 % bigdata small range, r=1
combined_matrix_5per = [ratio_1, data_mi_1_values_1]; % percenatge error vs actual value



logical_matrix = double(condition2);
data_mi_1_matrix = data_mi_1 .* logical_matrix;
E_2_matrix = E_2 .* logical_matrix;

% 计算 data_mi_1_matrix 每行非零元素的个数
non_zero_count_per_row = sum(logical_matrix, 2);

% 计算每行非零元素对应位置的百分比误差
percentage_error_matrix = abs(data_mi_1_matrix - E_2_matrix) ./ data_mi_1_matrix;
percentage_error_matrix(~logical_matrix) = NaN; % 将非零位置设为NaN，以便计算中位数时忽略这些位置

% 计算每行百分比误差的中位数
median_percentage_error_per_row = arrayfun(@(i) median(percentage_error_matrix(i, :), 'omitnan'), (1:size(percentage_error_matrix, 1))');

% 显示结果
disp('每行百分比误差的中位数:');
disp(median_percentage_error_per_row);

combined_freq_error_5per = [non_zero_count_per_row, median_percentage_error_per_row]; % flow freq vs flow's median percenatge error




