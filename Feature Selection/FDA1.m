function [Eigenvalue Order]=FDA1(FM,ClassNum)
% fisher discreminate analysis for filter based feature selection


x=DefineClass(FM,ClassNum);
if length(ClassNum)>1
cn=length(ClassNum);
else
cn=ClassNum;
end
%% step 1     within class mean
% we calculate mean of each classes so we colud know where is center of
% each classes to calculate distance within classses or we can compute one
% by one sample of each class distance
wm=mean(FM);

%% step 2.a     Sw=within class scatter matrix
% calculate Sj
Sw=0;
for c=1:cn
    wm=0;
    y=0;sj=0;
    
    y=find(x(:,end)==c);
    % step 1     within class mean
    wm=mean(FM(y,:));
    wm1=repmat(wm,length(y),1);
    % calculate Sj
    sj=sum((FM(y,:)-wm1).^2);
    Sw=sj+Sw;
end

Sw

%% Step 2.b   Sb=between class scatter matrix
Sb=0;
% between class mean
M=mean(FM);
for c=1:cn
    wm=0;
    y=0;
    
    y=find(x(:,end)==c);
    wm=mean(FM(y,:));
    
   s1=length(y).*((wm-M).^2);
   Sb=Sb+s1;
end

%% Step 3 eignvector and eigen values

% [Eigenvector_NotSorted, EigValMat]=eig(Sb,Sw);
Sw
Sb
[Eigenvector_NotSorted, EigValMat]=eig(Sw'*Sb);

% diagonals value in matrix "k" is our eigenvalues
% diagonals value in matrix "k" is our eigenvalues
%% Step 4 sort eigenvalue and eigen vectors
Eigenvalue_NotSorted=diag(EigValMat);
[Eigenvalue, Order]=sort(Eigenvalue_NotSorted,'descend');
% 
% Eigenvector=Eigenvector_NotSorted(:,Order);
% 
% %% Step 5 
% % multiply old feature matrix in Eigenvector to get 
% % new feature matrix which is sorted from high to low discremination
% 
% % FDR is new feature matrix
% FDA=x*Eigenvector;

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
