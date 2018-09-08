% demo for https://en.wikipedia.org/wiki/Loss_of_significance

digits(4)
% Start at pi/4, and move closer to zero each iteration
x = pi/4;
for i = 0:15
    fprintf('x = %d \n', x);
    
    % Compute absolute error for old and new function
    error_new = double((vpa(sin(x)) - vpa(2*sin(x/2)^2)) - (sin(x) - 2*sin(x/2)^2));
    error_old = double((vpa(sin(x) + cos(x)) - 1) - (sin(x) + cos(x) - 1));
    
    fprintf('error_new = %d \n', error_new);
    fprintf('error_old = %d \n', error_old);

    x = x/4;
end

f_precise(0)    % Output: 0
f_precise(pi/8) % Output: 0.306562964876377
f_precise(pi/4) % Output: 0.414213562373095
f_precise(pi/2) % Output: 0.306562964876377

% f_precise evaluates the function sinx + cosx - 1 to full precision
function eval = f_precise(x)
    % Outside of the neighborhood where x approaches zero
    if mod(abs(x), 2 * pi) > pi/4
        eval = sin(x) + cos(x) - 1;
        
    % Within the neighborhood where x approaches zero
    else
        eval = sin(x) - 2*sin(x/2)^2; % By problem 1.4.19
    end
end