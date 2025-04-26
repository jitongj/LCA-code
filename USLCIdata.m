%% change all NaN entries in two matrix into 0
S1 = USLCIdataS1;
S2 = USLCIdataS2;
for j = 1:638 %nunmber of process
    for i = 1: 4074 %number of flow
        %change the nan entry into 0
        if isnan(S1(i,j)) == 1
            S1(i,j) = 0;
        end
        %change the nan entry into 0
        if isnan(S2(i,j)) == 1
            S2(i,j) = 0;
        end

    end
end

%% 
input=[];
%import the non negative data from input
for j = 1:638 %nunmber of process
    for i = 1: 4074 %number of flow
        if S1(i,j)<0
            input(i,j) = 0;
        else
            input(i,j) = S1(i,j);
        end
    end
end

% import the negative data from output

for j = 1:638 %nunmber of process
    for i = 1: 4074 %number of flow
        if S2(i,j)<0
            input(i,j) = input(i,j) - S2(i,j);
        end
    end
end

% sum(input(:)<0) for checking
%%
output=[];

for j = 1:638 %nunmber of process
    for i = 1: 4074 %number of flow
        if S2(i,j)<0
            output(i,j) = 0;
        else
            output(i,j) = S2(i,j);
        end
    end
end

% import the negative data from output
for j = 1:638 %nunmber of process
    for i = 1: 4074 %number of flow
        if S1(i,j)<0
            output(i,j) = output(i,j) - S1(i,j);
        end
    end
end

% sum(output(:)<0) for checking

%%
save('USCLIdata.mat', 'input', 'output')
