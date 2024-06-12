function MSE=MS_SampEn(x,MAXscale,dim,tau,treshold)

L=length(x);
if nargin <5
    treshold=0.2*std(x);
end
%%
k=0;
for scale=1:MAXscale
k=k+1;
    j=fix(L/scale);

ys=reshape(x(1:scale*j),scale,j)';
my=mean(ys,2);
if length(my)<2
    break
end
y=my';
%% now you should calculate ENTROPY of each rows of "y"sepreatly
z= SampEn(y,dim,tau,treshold);
% z=PermEn(y,2);
% [z ~]=ShannonEn(y,1,10);

MSE(k)=z;

end
if nargout==0
plot(MSE,'--rs','LineWidth',2,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',5)
            title('MS SampEn')
            xlabel('Scales')
            ylabel('Sample Entropy')
end
end