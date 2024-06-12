function [Feat_Index,Accuracy,ACC_OVO_OVA,Results]=  BFS_GA_SVM_GridSearch(FM,ClassNum,...
    Kfold,Optimize,KernelFunction,Num_Grid,Iteration,POPsize,Generation)

%% ||||||||||||||||**** IMPORTANT *****||||||||||||||||||||||||||
% before use this function in SVM_GridSearch function,  disable    rng(1);


%%
[~, C]=size(FM);

global Data
global ClassN
global K_fold
global Accuracy0
global Optimize1
global KernelFunction1
global Num_Grid1
global Iteration1
global Results0
global Acc_OVO_OVA
Accuracy0=0;
Data  = FM; 
ClassN=ClassNum;
K_fold=Kfold;
Optimize1=Optimize;
KernelFunction1=KernelFunction;
Num_Grid1=Num_Grid;
Iteration1=Iteration;



tournamentSize = 2;
options = gaoptimset('CreationFcn', {@PopFunction},...
                     'PopulationSize',POPsize,...
                     'Generations',Generation,...
                     'PopulationType', 'bitstring',... 
                     'SelectionFcn',{@selectiontournament,tournamentSize},...
                     'MutationFcn',{@mutationuniform, 0.1},...
                     'CrossoverFcn', {@crossoverarithmetic,0.8},...
                     'EliteCount',2,... Number of best individuals that survive to next generation without any change
                     'StallGenLimit',100,...  If after this number of generations there is no improvement, the simulation will end
                     'PlotFcns',{@gaplotbestf},...  
                     'Display', 'iter'); 
rand('seed',1)

nVars = C; % 
FitnessFcn = @FitFunc; 
[chromosome] = ga(FitnessFcn,nVars,options);
Best_chromosome = chromosome; % Best Chromosome
Feat_Index = find(Best_chromosome==1); % Index of Chromosome
Accuracy=Accuracy0;
Results=Results0;
ACC_OVO_OVA=Acc_OVO_OVA;
end

%%% POPULATION FUNCTION
function [pop] = PopFunction(GenomeLength,~,options)
RD = rand;  
pop = (rand(options.PopulationSize, GenomeLength)> RD); % Initial Population
end

%%% FITNESS FUNCTION 
function [FitVal] = FitFunc(pop)
global Data
global ClassN
global K_fold
global Optimize1
global KernelFunction1
global Num_Grid1
global Iteration1

FeatIndex = find(pop==1); 
FM1 = Data;
FM2 = FM1(:,[FeatIndex]);
% [Accuracy Sensitivity Specificity ConfM1 Kappa]=MLP_Kfold(FM2,ClassN,K_fold,Layer0,neuron0);

[Accuracy, Results]=SVM_GridSearch(FM2,ClassN,K_fold,Optimize1,KernelFunction1,Num_Grid1,Iteration1);
FitVal=100-Accuracy;

bovo=find(Results.BestResults.Coding== 'onevsone');
bova=find(Results.BestResults.Coding== 'onevsall');


if isempty(bovo)==1
Acc0(1,1)= 1;
else
Acc0(1,1)= Results.BestResults.Objective(bovo(1));    
end

if isempty(bova)==1
Acc0(1,2)= 1;
else
Acc0(1,2)= Results.BestResults.Objective(bova(1));
end

Acc=(1-Acc0)*100;

global Results0
global Accuracy0
global Acc_OVO_OVA
if Accuracy>Accuracy0
Results0=Results;
Accuracy0=Accuracy;
Acc_OVO_OVA=Acc;
end
end


