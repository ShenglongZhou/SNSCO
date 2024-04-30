function [objef, gradf, hessf] = FuncfNOP(x,lambda)
 
    objef = (lambda/2)*norm(x)^2-sum(x);
    if  nargout>1
        gradf = lambda*x-1;
        hessf = lambda*eye(length(x));  
    end

end
