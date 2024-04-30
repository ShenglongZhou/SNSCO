clc; close all; clear all; warning off
addpath(genpath(pwd));

M     = 1;
N     = 100;
K     = 10;  

xi    = randn(K,M,N); 
A     = reshape(xi.*xi,K,M*N);    
b     = 5;
s0    = ceil([0.01 0.05 0.1]*N);

out    = cell(3,3);
out1   = cell(3,3);
mx     = zeros(3,3);
X0     = [ones(K,1) 1*rand(K,1)  0*rand(K,1)]; 
for j  = 1:size(X0,2)
    pars.x0 = X0(:,j);  
    for i   = 1:nnz(s0)
        s   = s0(i);              
        c         = sqrt(2*b/chi2inv((1-s/N)^(1/M),K));
        lam       = 1/c/2;
        FuncG     = @(x,W,J)FuncGNOP(x,W,J,A,b,K,M,N);
        Funcf     = @(x)FuncfNOP(x,lam);
        res       = SNSCO(K,M,N,s, Funcf, FuncG,pars); 
        out{j,i}  = res.Obj(1:res.iter);
        out1{j,i} = res.Error(1:res.iter);
        mx(j,i)   = res.iter;
     end
end
 
% plot results ------------------------------------------------------------  
close all;
Norows  = 1;
Nocols  = 3;
x0      = 0.025; % margin of the bottom left corner to the left edge
y0      = 0.16;  % margin of the bottom left corner to the bottom edge
gapx    = 0.02;  % gap between two subplots along x axis
gapy    = 0.07;  % gap between two subplots along y axis

posFig = [820 300 log(1+Nocols)*420 log(1+Norows)*320];
lx     = (1-2.95*x0-(Nocols-1)*gapx)/Nocols ;
ly     = (1-1.6*y0-(Norows-1)*gapy)/Norows ;
[X,Y]  = meshgrid(1:Nocols,1:Norows);
Y      = Y(end:-1:1,:);
posX   = x0 + (X-1)*(lx+gapx);
posY   = y0 + (Y-1)*(ly+gapy);

colors = {'#3d8c95','#225675','#e6873c'}; 
lgd    = {'\alpha=0.01','\alpha=0.05','\alpha=0.10'};
lsy    = { '-','-','-'};
tit    = { 'x^0=1','x^0=\zeta','x^0=0'};
figure('Renderer', 'painters', 'Position',posFig)
set(0,'DefaultAxesTitleFontWeight','normal');
for row = 1 : Norows   
    for col    = 1 : Nocols     
        sub    = subplot(Norows,Nocols,(row-1)*Nocols+col);       
        for i  = 1 : nnz(s0)
            y  = out1{i,col}(2:end);
           %  fg = loglog(max(1,length(y)-99):length(y),y(max(1,end-99):end)); hold on
           fg = semilogy(1:length(y),y); hold on
           %fg = plot(1:length(y),y); hold on
            fg.LineWidth = 1.5;
            fg.Color     = colors{i};
            fg.LineStyle = lsy{i};  
        end
        grid on; legend(tit,'Location','NorthEast');
        if  col == 1  
            ylabel('Error'); set(gca,'YTickLabel',[]);
        elseif col==2  
            set(gca,'YTickLabel',[]);
        else
            ax = gca;  ax.YAxisLocation = 'right'; 
            yticks([1e-5 1e-3 1e-1  10 100]);
            yticklabels({'10^{-5}','10^{-3}','10^{-1}','10^{1}','10^{2}'})
        end   
        axis([1 ceil(max(mx(:))/10)*10 1e-6 1e3]);
        xticks([10:10:ceil(max(mx(:))/10)*10]); 
        title(lgd{col},'FontWeight','Normal');
        xlabel('Iteration'); 
        set(sub, 'Position',[posX(row,col),posY(row,col),lx,ly] );
        set(0,'DefaultAxesTitleFontWeight','normal');

     end
end

 
figure('Renderer', 'painters', 'Position',posFig)
set(0,'DefaultAxesTitleFontWeight','normal');  
for row = 1 : Norows   
    for col    = 1 : Nocols     
        sub    = subplot(Norows,Nocols,(row-1)*Nocols+col);       
        for i  = 1 : nnz(s0)
            y  = out{i,col}(2:end);
            fg = plot(1:length(y),y); hold on
            fg.LineWidth = 1.5;
            fg.Color     = colors{i};
            fg.LineStyle = lsy{i};             
        end
        grid on; legend(tit,'Location','NorthEast');
        if  col == 1  
            ylabel('Objective'); set(gca,'YTickLabel',[]);
        elseif col==2  
            set(gca,'YTickLabel',[]);
        else
            ax = gca;  ax.YAxisLocation = 'right'; 
            yticks([-8:2:-2]);
        end   
        axis([1 ceil(max(mx(:))/10)*10 -9 0]);
        xticks([10:10:ceil(max(mx(:))/10)*10]); 
        title(lgd{col},'FontWeight','Normal');
        xlabel('Iteration'); 
        set(sub, 'Position',[posX(row,col),posY(row,col),lx,ly] );
        set(0,'DefaultAxesTitleFontWeight','normal');
     end
end
