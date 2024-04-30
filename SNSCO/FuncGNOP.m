function [G,gradG,gradGW, hessGW] = FuncGNOP(x,W,Ind,A,b,K,M,N)
    G   = reshape((x.*x/2)'*A-b,M,N); 
    if  nargout > 1
        if  isempty(Ind) 
            gradG   = [];
            gradGW  = zeros(K,1); 
            hessGW  = zeros(K,1);
        else 
            AInd    = A(:,Ind);    
            gradG   = x.*AInd; 
            hessGW  = AInd*reshape(W(Ind),length(Ind),1);   
            gradGW  = x.*hessGW;  
            hessGW  = diag(hessGW); 
        end
    end  
end

