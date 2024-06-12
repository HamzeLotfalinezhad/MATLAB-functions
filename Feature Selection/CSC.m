function [Jm2 Feature_Ind]=CSC(FM,ClassNum)
% CSC class separability criterion for feature selection

% refrence
% Feature selection method based on mutual information and class
% separability for dimension reduction in multidimensional time series
% for clinical data

x=DefineClass(FM,ClassNum);
if length(ClassNum)>1
cn=length(ClassNum);
else
cn=ClassNum;
end

%% within class scatter matrix
% calculate Sj
n=size(FM,1);
Jw=0;
for c=1:cn
    wm=0;
    y=0;Jj=0;
    
    y=find(x(:,end)==c);
    nj=length(y);
    Pj=nj/n; % prior probability of class
    % step 1     within class mean
    wm=mean(FM(y,:));
    wm1=repmat(wm,length(y),1);
    % calculate Sj
    Jj=Pj.*(1/n).*sum((FM(y,:)-wm1).^2);
    
    Jw=Jj+Jw;
end

%% Step 2.b   Sb=between class scatter matrix
Jb=0;
% between class mean
M=mean(FM);
for c=1:cn
    wm=0;
    y=0;
    
    y=find(x(:,end)==c);
    wm=mean(FM(y,:));
    nj=length(y);
    Pj=nj/n;
    
   s1=Pj.*((wm-M).^2);
   Jb=Jb+s1;
end

%%
Jm=Jb./Jw;

[Jm2, Feature_Ind]=sort(Jm,'descend');
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
