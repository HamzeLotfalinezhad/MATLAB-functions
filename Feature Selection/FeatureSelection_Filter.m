function [BestF_ind Values]=FeatureSelection_Filter(FM,ClassNum,FilterType) 
% #codegen
[~, S]=size(FM);
Features=DefineClass(FM,ClassNum);


%% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
switch FilterType
    case 'ttest' % for just 2 class

        C1=FM(Features(:,end)==1,:);
        C2=FM(Features(:,end)==2,:);
        
        M1=mean(C1);
        M2=mean(C2);
        
        V1=var(C1);
        V2=var(C2);
        
        N1=size(C1,1);
        N2=size(C2,1);
        
        TTest=abs(M1-M2)./(sqrt( ((V1)/N1)+((V2)/N2)  ));

        [Values BestF_ind]=sort(TTest,'descend');
%% |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
case 'entropy' % for just 2 class
        % Theodoridis & Koutroumbas, Acad.Press, pp 152. 
        C1=FM(Features(:,end)==1,:);
        C2=FM(Features(:,end)==2,:);
        
        M1=mean(C1);
        M2=mean(C2);
        
        V1=var(C1);
        V2=var(C2);
        
        N1=size(C1,1);
        N2=size(C2,1);
                
        Entropy = (V2./V1+V1./V2-2)/2+(M2-M1).^2.*(1./V2+1./V1)/2;
        
         [Values BestF_ind]=sort(Entropy,'descend');


%% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||                
    case 'variance' % for  2 and more than 2 class
        VAR=var(FM); 
        % the greather the variance the more seprable the two classes
        % also you can add threshols to eliminate lowest variance
        [Values BestF_ind]=sort(VAR,'descend');
%% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||                
    case 'infogain' % for  2 and more than 2 class
        % information Gain=Entropy(parent)-mean(Entropy(children))
        % InfoGain.pdf
        % the bigger the information gain the better for classification
for i=1:S
IG(:,i)=InfoGain(FM(:,i),ClassNum,2);
end
    [Values BestF_ind]=sort(IG,'descend');

%% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    case 'pcc' % for just 2 class
%  Pearson correlation coefficient (PCC), is a measure of the linear dependence (correlation) between two variables X and Y.
% It has a value between +1 and -1 inclusive, where 
% 1 is total positive linear correlation, 
% 0 is no linear correlation,and 
% -1 is total negative linear correlation

        C1=FM(Features(:,end)==1,:);
        C2=FM(Features(:,end)==2,:);
        %     delete extra values or we can n
sc1=size(C1,1);
sc2=size(C2,1);
if sc1<sc2
CC2=C2(1:sc1,:);  
CC1=C1;
SC=sc1;
elseif sc1>sc2
CC1=C1(1:sc2,:); 
SC=sc2;
CC2=C2;
elseif sc1==sc2
CC1=C1;
CC2=C2; 
SC=sc1;
end
    
        for i=1:size(C1,2)
        COV=cov(CC1(:,i),CC2(:,i));
        PCC(1,i)=COV(1,2)./sqrt(var(CC1(:,i)).*var(CC2(:,i)));
        end
                [Values, BestF_ind]=sort(PCC);
 %% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||                               
  case 'fisher'  % two and more than two class            
   [Jm2, Feature_Ind]=CSC(FM,ClassNum);
%      class separability criterion for feature selection

Values=Jm2;
BestF_ind=Feature_Ind;
 %% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||   
 case 'snr'
%      large value in SNR means strong corrolation
C1=FM(Features(:,end)==1,:);
C2=FM(Features(:,end)==2,:);
        
        M1=mean(C1);
        M2=mean(C2);
        
        s1=std(C1);
        s2=std(C2);
        
        SNR=abs((M1-M2)./(s1-s2));
        
        [Values, BestF_ind]=sort(SNR,'descend');
  
     
 %% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||   
    case 'MI' % JUST 2 CLASS
C1=FM(Features(:,end)==1,:);
C2=FM(Features(:,end)==2,:);

        for ii=1:size(FM,2)
%         MI0=mutual(FM(:,i));
%         MI(1,i)=min(MI0);
Mutualinfo(ii)=MI(C1(:,ii),C2(:,ii));
        end
      [Values, BestF_ind]=sort(Mutualinfo,'descend');
 %% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||               
    case 'relieff'
    K_nearest_neighbors=10;
    [BestF_ind,Values] = relieff(FM,Features(:,end),K_nearest_neighbors);
 %% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||               
     case 'fdr' % for  2 and more than 2 class
[Eigenvalue, Order]=FDA1(FM,ClassNum);
% [Eigenvalue Order]=FDA(FM,ClassNum);

Values=Eigenvalue;
BestF_ind=Order';
 %% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||               
     case 'fen' % for  2 class
C1=FM(Features(:,end)==1,:);
C2=FM(Features(:,end)==2,:);

for i=1:size(C1,2)
% F1 = FuzzyEn( C1(:,i),3,1,std(C1),2);
% F2 = FuzzyEn( C2(:,i),3,1,std(C2),2);

% F1 = SampEn( C1(:,i),3,1,0.25*var(C1));
% F2 = SampEn( C2(:,i),3 );

F1 = GHurst( C1(:,i),1,5);
F2 = GHurst( C2(:,i),1,5);

F(1,i)=abs(F1-F2);
end

      [Values, BestF_ind]=sort(F,'descend');


end



% if CDF_Plot=='yes'
% ecdf(p);
% xlabel('P value');
% ylabel('CDF value')
% end  

end
%% ||||||||||||||||||||||||||||||||||||||||||| sub function |||||||||||||||||||||||||||||||||||||||||||||||||
function Features=DefineClass(Features0,ClassNum)
sf=size(Features0,2);
if length(ClassNum)>1 % if class sizes dont have same number of Trials
    kj=0;
    for i2=1:length(ClassNum)
        for ii=1:ClassNum(i2)
            kj=kj+1;
Features0(kj,sf+1)=i2;
        end
    end
    
else
FMC=size(Features0,1)/ClassNum;
kj=0;
for i2=1:ClassNum
    for j=1:FMC
        kj=kj+1;
Features0(kj,sf+1)=i2;
    end
end

end
Features=Features0;
end
