clc; close all; clear all; warning off
addpath(genpath(pwd));

K     = 10; 
M     = 10; 
N     = 100;
alpha = 0.1;
s     = ceil(alpha*N);

b     = 5;
c     = sqrt(2*b/chi2inv((1-s/N)^(1/M),K));
xi    = zeros(K,M,N);     

if  M  ==1  
    xi    = randn(K,M,N);        % iid data
else
    E     = repmat((1:K)'/K,1,M);
    C     = 0.5*ones(M,M)+0.5*eye(M); 
    for n = 1:N
        xi(:,:,n) = mvnrnd(E,C);  % non-iid data
    end 
end
lam        = 1/c/2; 
A          = reshape(xi.*xi,K,M*N);      
Funcf      = @(x)FuncfNOP(x,lam);
FuncG      = @(x,W,J)FuncGNOP(x,W,J,A,b,K,M,N); 
P          = 50-25*(alpha==0.01);    
pars.tau0  = logspace(log10(0.5),log10(1.75),P); 
out        = SNSCO(K,M,N,s,Funcf,FuncG,pars);
            