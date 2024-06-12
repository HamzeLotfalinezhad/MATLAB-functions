function [Feat_Index Accuracy Sensitivity Specificity ConfM1 Kappa ]=  BFS_GA_NBC(FM,ClassNum,...
    Kfold,POPsize,Generation)

[~, C]=size(FM);

global Data
global ClassN
global K_fold
global CM0
global Accuracy0
global Sensitivity0
global Specificity0
global Kappa0
CM0=0;
Accuracy0=0;
Data  = FM; 
ClassN=ClassNum;
K_fold=Kfold;

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
FitnessFcn = @FitFunc_KNN; 
[chromosome] = ga(FitnessFcn,nVars,options);
Best_chromosome = chromosome; % Best Chromosome
Feat_Index = find(Best_chromosome==1); % Index of Chromosome
CM2=CM0;
ACC2=Accuracy0;
Accuracy=Accuracy0;
Sensitivity=Sensitivity0;
Specificity=Specificity0;
ConfM1=CM0;
Kappa=Kappa0;
end

%%% POPULATION FUNCTION
function [pop] = PopFunction(GenomeLength,~,options)
RD = rand;  
pop = (rand(options.PopulationSize, GenomeLength)> RD); % Initial Population
end

%%% FITNESS FUNCTION 
function [FitVal, ConfM1] = FitFunc_KNN(pop)
global Data
global ClassN
global K_fold
FeatIndex = find(pop==1); 
FM1 = Data;
FM2 = FM1(:,[FeatIndex]);
[Accuracy Sensitivity Specificity ConfM1 Kappa]=NBC_Kfold(FM2,ClassN,K_fold);
FitVal=100-Accuracy;
global CM0
global Accuracy0
global Sensitivity0
global Specificity0
global Kappa0
if Accuracy>Accuracy0
CM0=0;
CM0=CM0+ConfM1;
Accuracy0=Accuracy;
Sensitivity0=Sensitivity;
Specificity0=Specificity;
Kappa0=Kappa;
end
end


