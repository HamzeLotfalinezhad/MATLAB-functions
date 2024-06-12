function MSE=MS_FuzzyEn(x,MAXscale,dim,tau,treshold,n)

% D = exp(-(dist.^n)/r);     %% (dist < r);

L=length(x);
if nargin <6
    n=2;
end
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
z = FuzzyEn( y,dim, tau,treshold, n );

MSE(k)=z;

end
if nargout==0
plot(MSE,'--rs','LineWidth',2,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',5)
            title('MS FuzzyEn')
            xlabel('Scales')
            ylabel('Fuzzy Entropy')
end
end