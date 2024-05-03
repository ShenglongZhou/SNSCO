function out = SNSCO(K,M,N,s,Funcf,FuncG,pars)
% This solver solves 0/1 constrained optimization in the following form:
%
%         min_{x\in\R^K} f(x),  s.t. \| G(x) \|^+_0<=s, x>=0 
%
% where 
%      f(x) : \R^K --> \R
%      G(x) : \R^K --> \R^{M-by-N}
%      s << N 
%      \|Z\|^+_0 counts the number of columns with positive maximal values
% =========================================================================
% Inputs:
%   K      : Dimnesion of variable x                             (REQUIRED)
%   M      : Row number of G(x)                                  (REQUIRED)
%   N      : Column number of G(x)                               (REQUIRED)
%   s      : An integer in [1,N), typical choice ceil(0.01*N)    (REQUIRED)
%   Funcf  : Function handle of f(x)                             (REQUIRED)
%   FuncG  : Function handle of G(x)                             (REQUIRED)
%   pars   : All parameters are OPTIONAL  
%            pars.x0      -- Initial point (default:ones(K,1)) 
%            pars.tau0    -- A vector containing a number of parameter \tau (default:1)
%                            e.g., pars.tau0 =logspace(log10(0.5),log10(1.75),50);
%            pars.tol     -- Tolerance of the halting condition (default:1e-6*M*N)
%            pars.maxit   -- Maximum number of iterations (default: 2000) 
%            pars.display -- Display results or not for each iteration (default:1)
% =========================================================================
% Outputs:
%     out.x:      Solution x
%     out.obj:    Objective function value f(x)
%     out.G:      Function value of G(x) 
%     out.time:   CPU time
%     out.iter:   Number of iterations 
%     out.error:  Error
%     out.Error:  Error of every iteration
% =========================================================================
% Written by Shenglong Zhou on 30/04/2024 based on the algorithm proposed in
%     Shenglong Zhou, Lili Pan, Naihua Xiu, and Geoffrey Ye Li, 
%     0/1 constrained optimization solving sample average approximation 
%     for chance constrained programming, arXiv:2210.11889, 2024    	
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
% =========================================================================
 warning off; 

if nargin < 7; pars = []; end

[gamma0,mu0,I0,r0,maxit,tau0,x0,W0,tol,thd,display] ...
        = GetParameters(K,M,N,s,pars);
mk      = length(tau0);   
iteron  = (mk==1);
out.obj = Inf;
out.obj1= Inf;
out.time= 0;

if  display
    if  mk==1
        fprintf('   Iter     Time(sec)      Error      Objective    Level\n')
    else
        
        fprintf('   tau      Time(sec)      Error      Objective    Level\n')
    end
    fprintf(' -------------------------------------------------------\n')
end

for i   = 1:mk    
 
    outi = SNSCOsingle(x0,W0,s,Funcf,FuncG,tau0(i),mu0,gamma0,I0,r0,ceil(maxit/mk),tol,thd,iteron);
    out.time  = out.time + outi.time;
    if  display && mk>1
        fprintf('  %6.4f     %6.4f      %.2e    %8.4f      %3d\n',...
        tau0(i), out.time,outi.error,outi.obj,outi.voil) ;
    end 
    
    if  mk==1
        out     = outi;
        out.tau = tau0(i);
    else
        if  outi.obj  < out.obj && outi.voil<=s 
            out.obj   = outi.obj;
            out.x     = outi.x;
            out.G     = outi.G;
            out.voil  = outi.voil;
            out.error = outi.error;
            out.mark  = outi.mark;
            out.timeb = outi.time;
            out.tau   = tau0(i);
        end     
    end  
end
 
fprintf(' -------------------------------------------------------\n')
fprintf(' Objective:     %10.4f\n',out.obj);
fprintf(' Voilations:    %10d\n',out.voil);
fprintf(' Time:          %6.3f sec\n',out.time);
fprintf(' Best tau:      %10.4f\n',out.tau);
fprintf(' -------------------------------------------------------\n')
end


%--------------------------------------------------------------------------
function [gamma0,mu0,i0,r0,maxit,tau0,x0,W,tol,thd,disp] = GetParameters(K,M,N,s,pars)    
    tau0   = 1; 
    mu0    = 0.125;  
    x0     = ones(K,1);
    W      = 50*ones(M,N); 
    tol    = 1e-6*M*N; 
    gamma0 = 1 + (2-2*(s<=2))/s;
    r0     = 0.5;
    i0     = 6 + (M>1); 
    disp   = 1;       
    thd    = 1e-4; 
    if isfield(pars,'gamma0');  gamma0 = pars.gamma0;   end
    if isfield(pars,'tau0');    tau0   = pars.tau0;     end
    if isfield(pars,'tol');     tol    = pars.tol;      end
    if isfield(pars,'x0');      x0     = pars.x0;       end
    if isfield(pars,'W0');      W      = pars.W0;       end
    if isfield(pars,'mu0');     mu0    = pars.mu0;      end
    if isfield(pars,'i0');      i0     = pars.i0;       end
    if isfield(pars,'r0');      r0     = pars.r0;       end
    if isfield(pars,'disp');    disp   = pars.disp;     end
    mk    = length(tau0); 
    maxit = 2e3*(mk==1) + 250*mk*(mk>1); 
    if isfield(pars,'maxit');  maxit = pars.maxit;      end
end

function out = SNSCOsingle(x,W,s,Funcf,FuncG,tau,mu,gamma,I0,r0,maxit,tol,thd,display)

t0    = tic;

K     = length(x);
[M,N] = size(W);
bestf = Inf; 
besti = 1;
bestE = 1;  
Err   = zeros(maxit,1);
fail  = zeros(maxit,1);
obj   = zeros(maxit,1);
sol   = zeros(K,maxit);
T     = 1:K;  
FwV   = zeros(K,1);
d     = zeros(K,1);  
JJ    = cell(maxit,1);
Fnorm = @(var)norm(var,'fro'); 
TtauI = @(Lambda)Ttau(Lambda,M,s);   
funG  = @(v)FuncG(v,[],[]);
Ind   = TtauI(funG(x)+tau*W); 
fvoil = @(v)nnz(max(v,[],1)>thd);  
for iter  = 1:maxit
    
    [f,gf,hf]      = Funcf(x); 
    [G,gG,gGW,hGW] = FuncG(x,W,Ind);  
    gfG            = gf+gGW;
    z              = x - tau*gfG;  
    Tp             = find(z>= 0); 
    nTp            = length(Tp);  
     
   if nTp==K  || nTp ==0
        FTp    = -gfG; 
        FwV    = tau*FTp;
       Tp      = T;
       nTp     = K;
   else
        Tn      = T;
        Tn(Tp)  = []; 
        FTp     = -gfG(Tp);  
        FTn     = -x(Tn);
        FwV(Tp) = tau*FTp;  
        FwV(Tn) = FTn; 
   end

    GV          = -reshape(G(Ind),[],1); 
    Error       = Fnorm([FwV; GV]);      
    voils       = fvoil(G); 
    Err(iter)   = Error;
    fail(iter)  = voils; 
    sol(:,iter) = x;
    obj(iter)   = f; 

    if  voils <= s &&   bestf > f - 1e-4      
        bestf  = f;  
        bestx  = x;
        bestG  = G;
        bestW  = W;
        besti  = iter; 
        bestv  = voils;
    end  

    if  (mod(iter,1)==0 || iter<10) && display 
        fprintf('  %4d       %6.4f      %.2e    %8.4f      %3d\n' , iter,toc(t0),Error,f,voils) ;
    end      
   
    stop  = Error < tol && voils <=s;  
    stop0 = iter  > 9  && std(obj(iter-9:iter))<1e-6 && voils <=s;

    if  (stop  || stop0) && voils>0
        if  ~(mod(iter,10)==0 || iter<10)  && display 
            fprintf('  %4d       %6.4f      %.2e    %8.4f      %3d\n' , iter,toc(t0),Error,f,voils) ;
        end
        break; 
    end 

    nGV = size(gG,2);  
    if isempty(Ind) 
       if  nTp    == K
           dir     = hf\FTp; 
       else
           dir     = zeros(K,1); 
           dir(Tp) = hf(Tp,Tp)\FTp;  
           dir(Tn) = FTn; 
       end
    else 
        if  nTp  ~= K 
            GV    = GV-(FTn'*gG(Tn,:))'; 
            gG    = gG(Tp,:);
        end
        if  nTp <= 500 && nGV<1e5  
            temp =  hGW(Tp,Tp)+ hf(Tp,Tp);  
            gGt  = gG';  
            tnan = 0;
            if  nGV   <= nTp*0.25
                rhs    = gGt*(temp\FTp)- GV; 
                tnan   = (max(isnan(rhs))==1);  
                if ~tnan
                    D      = gGt*(temp\gG);
                    dV     = D\rhs;
                    if norm(dV)>1e3*nGV+1e5*(voils==N) 
                       dV  = ( D + (1e-2/iter)*eye(nGV))\rhs; 
                    end
                    dTp    = temp\(FTp-gG*dV); 
                end                
            end
            
            if  nGV    > nTp*0.25 || tnan                 
                D      = gG*gGt+ mu*temp; 
                if  tnan 
                    D  = D + (1e-2/iter)*eye(nTp); 
                end
                rhs    = mu*FTp + gG*GV;  
                dTp    = D\rhs;  
                dV     = (gGt*dTp-GV)/mu;        
            end
                
        else   
            rsh   = mu*FTp + gG*GV;
            fx    = @(v) (mu*( hGW(Tp,Tp)+ hf(Tp,Tp) ) + 1e-4/iter).*v+gG*(v'*gG)'; 
            dTp   = my_cg(fx,rsh,1e-6,10,zeros(nTp,1));
            dV    = ((dTp'*gG)'-GV)/mu; 
        end

        if nTp   == K 
            dir   = [dTp; dV]; 
        else
            d(Tp) = dTp;  
            d(Tn) = FTn;
            dir   = [d; dV]; 
        end
    end

    if  max(isnan(dir))==1 || Fnorm(dir) > (1e8*(M==1)+1e8*(M>1))  
        dir  = [FwV; GV]; 
        dir  = dir/max(abs(dir));  
    end

    step        = 1;
    xold        = x; 
    NG          = N*ones(I0+1,1); 
    ii          = max(5,I0-2) + 2*(s/N<0.1);     
    for i       = 1:I0 
        x       = xold + step*dir(1:K);   
        G       = funG(x);
        NG(i+1) = fvoil(G); 
        if abs(NG(i+1)-s) <= (gamma-1)*s || ... 
           (NG(i+1) ~= N && i > ii && nnz(NG((i-ii):i)-NG(i+1))==0 )
            break; 
        end
        step = step*r0;
    end
 
    if  i   == I0 
        mNG  = min(abs(NG(NG>0)-s)); 
        t    = find(abs(s-NG)==mNG); 
         if  mNG == N-s && tnan 
             step =  r0^t(end); 
         else
            step = r0^(max(0,t(1)-2)); 
         end
        x    = xold + step*dir(1:K); 
        G    = funG(x);      
    end

    Wold       = W;
    W          = zeros(M,N);
    dirW       = dir(K+1:end);
    if M       == 1
       dirW    = dirW'; 
    end
    if  step   == 1  
        W(Ind) = Wold(Ind) + dirW; 
    else 
        W(Ind) = Wold(Ind) + step*dirW;    
    end
    
    Ind = TtauI(G+tau*W);  

    if  mod(iter,10)==0 
        gamma = max(1,gamma-1/s);  
        mu    = max(1e-6, min(0.9995*mu, 1e2*N*Error) ); 
    end    
    JJ{iter} = Ind; 
    rep      = 0;  
    if iter  > 20  
       rep   = nnz(abs(fail(iter-10:iter)-voils)<1);  
    end
    
    if (iter > 99+100*(M==1)) || (rep  > 10)  
       if  rep > 0 || isempty(Ind) || iter>1e3
           for j    = (iter-2):(iter-1)
               Jc   = union(Ind,JJ{j});
           end 
           Ind      = reshape(unique(Jc),[],1);  
           JJ{iter} = Ind;  
       end     
    end
     
end
 
if  bestf == Inf || (abs(bestf-f)<1e-4 && voils<=s && bestE > Error)
    bestf = f;  
    bestx = x;
    bestG = G; 
    bestW = W; 
    besti = iter;
    bestv = voils;
end  

out.time  = toc(t0);
out.mark  = bestf == f; 
out.x     = bestx;
out.W     = bestW;
out.obj   = bestf;
out.G     = bestG; 
out.bit   = besti; 
out.voil  = bestv;
out.iter  = iter;
out.error = Error; 
out.Error = Err;
out.Obj   = obj;  
end

% calculate J -------------------------------------------------------------
function J = Ttau(Lambda,M,s)
    t = 1e-8;
    z = max(Lambda,[],1);   
    T = find(abs(z)<=t);
    if nnz(z>t)>s
        Tp     = find(z>t);  
        zTp    = sum(max(Lambda(:,Tp),0).^2,1); 
        [~,Js] = maxk(zTp,s);  
        Tp(Js) = []; 
        T      = sort([T Tp]');   
    end 
    J = [];
    if ~isempty(T)
        [idx,idy] = find(Lambda(:,T)>=-t);
        J = (reshape(T(idy),[],1)-1)*M + reshape(idx,[],1);
    end
     J = sort(J);
end

% conjugate gradient-------------------------------------------------------
function x = my_cg(fx,b,cgtol,cgit,x)
    if ~isa(fx,'function_handle'); fx = @(v)fx*v; end
    r = b;
    if nnz(x)>0; r = b - fx(x);  end
    e = norm(r,'fro')^2;
    t = e;
    p = r;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        w  = fx(p);
        pw = p.*w;
        a  = e/sum(pw(:));
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = norm(r,'fro')^2;
        p  = r + (e/e0)*p;
    end   
end
