format short

syms x f(x)

f(x) = atan(x);
root = 0; % Known

r(x) = diff(f(x), 2)/(2*diff(f(x), 1));
convergence_point = r(root); % Eval: 0
% This confirms that the point of convergence is zero.


Secant(f(x), [1, .75], 20)
%    -0.3345
%     0.0563
%    -0.0023
%    2.3021e-06
%   -3.9138e-12
%    6.9139e-24
%      0
%    NaN

% Notice the error proportion decreases each iter. As the root converges,
% the error decreases superlinearly. Near the end, The numbers are so small
% that a division by zero occurs due to machine roundoff.

function root = Secant(funct, pts, n)
    error = [.5,.6,.7];
    syms x
    
    div = eval(subs(funct, x, pts(1)) - subs(funct, x, pts(2)));
    i = 0;
    
    % Continue until convergence or iterations n.
    while (i <= n && div ~= 0)
        
        % Evaluate the next point, shift the other points down in the array
        pts = [pts(1) - eval(subs(funct, x, pts(1))) * ((pts(1) - pts(2)) / div), pts(1)];
        
        % f(x_1) - f(x_2)
        div = eval(subs(funct, x, pts(1)) - subs(funct, x, pts(2)));
        
        % Evaluate the next error, shift the other errors down in the array
        error = [eval(-0.5 * subs(diff(funct, 2), x, pts(1)) / subs(diff(funct), x, pts(1)) * prod(error(1:2))), error(1), error(2)];
        disp(error(1) / (error(2)*error(3)));
        
        i = i + 1;
    end
    root = pts(1);
end