function [lambda,x,values,vectors, residuals,its, x0] = shroff_extrap_dynamic(A,varargin)
%Using the extrapolation on the Shifted Symmetric Higher Order Power
%Method (SSHOPM). We can use this for dynamic and regular extrapolation.
%This needs to be used in conjunction with the tensor toolbox.
%
%% Error checking on A
P = ndims(A);
N = size(A,1);

if ~issymmetric(A)
    error('Tensor must be symmetric.')
end

%% Check inputs
p = inputParser;
p.addParamValue('Shift', 'adaptive');
p.addParamValue('MaxIts', 1000, @(x) isscalar(x) && (x > 0));
p.addParamValue('Start', [], @(x) isequal(size(x),[N 1]));
p.addParamValue('Tol', 1.0e-15, @(x) isscalar(x) && (x > 0));
p.addParamValue('Display', -1, @isscalar);
p.addParamValue('Concave', false, @islogical);
p.addParamValue('Margin', 1e-6, @(x) isscalar(x) && (x > 0));
p.addParamValue('Gamma', 'adaptive_gamma'); % Default is adaptive with safeguarding
p.parse(varargin{:});

%% Copy inputs
maxits = p.Results.MaxIts;
x0 = p.Results.Start;
shift = p.Results.Shift;
tol = p.Results.Tol;
display = p.Results.Display;
concave = p.Results.Concave;
margin = p.Results.Margin;
gamma = p.Results.Gamma;

%% Check shift
if ~isnumeric(shift)
    adaptive = true;
    shift = 0;
else
    adaptive = false;
end

%% Check Gamma
if ~isnumeric(gamma)
    adaptive_gamma = true;
    gamma = 0;
else
    adaptive_gamma = false;
end

%% Check starting vector
if isempty(x0)
    x0 = 2*rand(N,1)-1;
end

if norm(x0) < eps
    error('Zero starting vector');
end

%% Check concavity
if shift ~= 0
    concave = (shift < 0);
end        

%% Execute extrapolated power method
if (display >= 0)
    fprintf('TENSOR SHIFTED POWER METHOD WITH SIMPLE EXTRAPOLATION: ');
    if concave
        fprintf('Concave ');
    else
        fprintf('Convex  ');
    end
    fprintf('\n');
    if isnumeric(shift)
    fprintf('Shift : %4d', shift);
    end
    fprintf(', Initial Vector : [');
    fprintf('%g ', x0/norm(x0));
    fprintf(']');
end

flag = -2; ldif = 1; j = 0;
x = x0/norm(x0);
lambda = x'*ttsv(A,x,-1);
vectors = zeros(1,N);
if adaptive
    shift = adapt_shift(A,x,margin,concave);
end
plam = lambda;
v = ttsv(A,x,-1) + shift * x;
if (concave)
    v = -v;
end
 
nv = norm(v);
if nv < eps 
    flag = -1; 
    return;
end
x = v / nv;    
lambda = x'* ttsv(A,x,-1);
values(1) = lambda; vectors(1, :) = x';
ldif = abs(lambda - plam); value_diff(1) = ldif; 
residuals(1) = norm(ttsv(A,x,-1) - lambda*x); 

% ---- Doing it again for another x value----------------------------------
if adaptive
    shift = adapt_shift(A,x,margin,concave);
end
plam = lambda;
v = ttsv(A,x,-1) + shift * x;
if (concave)
    v = -v;
end
 
nv = norm(v);
if nv < eps 
    flag = -1; 
    return;
end
x = v / nv;    
lambda = x'* ttsv(A,x,-1);
values(2) = lambda; vectors(2, :) = x';
ldif = abs(lambda - plam); value_diff(2) = ldif; %Not returning this for now.
residuals(2) = norm(ttsv(A,x,-1) - lambda*x);

if norm(abs(lambda-plam)) < tol
    flag = 0;
end

if flag == 0
    return;
end

%---We start the extrapolation from here-----------------------------------
%---It makes sense to use sshopm for the previous two---------
%---iterations. --------------------------

j = 2; 
if adaptive
    shift = adapt_shift(A,x,margin,concave);
end

while ldif > tol && j < maxits
  j = j + 1; 
  x0 = x; %to capture the previous x
  v0 = v; %To capture the previous v
  plam = lambda;   
  v = ttsv(A,x,-1) + shift * x;
    if (concave)
        v = -v;
    end
  nv = norm(v);
    if nv < eps 
        flag = -1; 
        break;
    end
    if adaptive_gamma
        [gamma, rho(j-2)] = adapt_g(A, x, shift, lambda);
    end
  u = (1-gamma)*v + gamma*v0;  
  x = u/norm(u);    
  x_gamma = (1-gamma)*x + gamma*x0;
  x_gamma = x_gamma/norm(x_gamma);
  lambda = x_gamma'*ttsv(A,x_gamma,-1);
  values(j) = lambda;vectors(j, :) = x_gamma';
  ldif = abs(lambda - plam); value_diff(j) = ldif;
  residuals(j) = norm(ttsv(A,x_gamma,-1) - lambda*x_gamma);

  if adaptive_gamma  % Incase we need gamma values later. Not used currently
      gamma_values(j-2)= gamma;
  end
   
    if adaptive
        newshift = adapt_shift(A,x,margin,concave);        
    else
        newshift = shift;
    end
    
    if norm(abs(lambda-plam)) < tol
        flag = 0;
    end
   
    shift = newshift;
    
    if flag == 0
        break;
    end
end
its = j;

%% Display Results ------------------------------------------

if (display > 0) && ((flag == 0) || (mod(its,display) == 0))
        fprintf('\n');
        fprintf('----  --------- ----- ------------ ----- ------\n');
        fprintf('Iter  Lambda    Diff  |newx-x|     Shift Gamma \n');
        fprintf('----  --------- ----- ------------ ----- ------\n');
        fprintf('%4d  ', its);
        % Lambda
        fprintf('%9.6f ', lambda);
        d = lambda-plam;
        if (d ~= 0)
            if (d < 0), c = '-'; else c = '+'; end
            fprintf('%ce%+03d ', c, round(log10(abs(d))));
        else
            fprintf('      ');
        end
        % Change in X
        fprintf('%8.6e ', norm(vectors(end, :)-vectors(end-1, :)));          
        
        % Shift
        fprintf('%5.2f', shift);

        % Gamma
        fprintf('%6.2f', gamma);
end

%% Check results
if (display >=0)
    switch(flag)
        case 0
            fprintf('\n');
            fprintf('Successful Convergence');
        case -1 
            fprintf('\n');
            fprintf('Converged to Zero Vector');
        case -2
            fprintf('\n');
            fprintf('Exceeded Maximum Iterations');
        otherwise
            fprintf('\n');
            fprintf('Unrecognized Exit Flag');
    end
    fprintf('\n');
end


%% ----------------------------------------------------
%% --- Will look into this later ----------------------

function alpha = adapt_shift(A,x,tau,concave)

m = ndims(A);
Y = ttsv(A,x,-2);
e = eig(Y);

if concave
    if max(e) <= -tau/(m^2-m)
        alpha = 0;
    else
        alpha = -(tau/m) - ((m-1)*max(e));
    end
else
    if min(e) >= tau/(m^2-m)
        alpha = 0;
    else
        alpha = (tau/m) - ((m-1)*min(e));
    end
end

function [optimal_g, rho] = adapt_g(A, x, shift, lambda)

m = ndims(A);
J = ((m-1)*(ttsv(A, x, -2) - lambda*(x*x')) + shift*(eye(size(A,1))- x*x'))/(lambda + shift);
e = eig(J);
rho = max(e);
if rho < 1    
    optimal_g = ((rho - 2) + 2* sqrt(1 - rho))/rho;
else
    optimal_g = ((rho - 2) + 2* real(sqrt(1 - rho)))/rho;
end