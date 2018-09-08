format('long')
% demo A
Secant(@demo_A, [Newton(@demo_A, @demo_A_prime, 2, 1), 2], 5)
Newton(@demo_A, @demo_A_prime, 2, 5)
% After five iterations, Newton's method is slightly more accurate.
% Secant method: 1.532088886237958
% Newton method: 1.532088886237968

% demo B
Secant(@demo_B, [Newton(@demo_B, @demo_B_prime, 2, 1), 2], 5)
Newton(@demo_B, @demo_B_prime, 2, 5)
% After five iterations, Newton's method is again slightly more accurate.
% Secant method: 1.236183928207312
% Newton method: 1.236183928518088

function root = Newton(funct, funct_prime, x, n)
    for i = 1:n
        x = x - funct(x) / funct_prime(x);
    end
    root = x;
end

function root = Secant(funct, x, n)
    div = funct(x(1)) - funct(x(2));
    i = 0;
    % Due to the possible division by zero, this method has an additional
    % escape condition div.
    while (i <= n && div ~= 0)
        div = funct(x(1)) - funct(x(2));
        x = [x(1) - funct(x(1)) * ((x(1) - x(2)) / div), x(1)];
        i = i + 1;
    end
    root = x(1);
end


function y = demo_A(x)
    y = x^3 - 3*x + 1;
end

function y = demo_A_prime(x)
    y = 3 * x^2 - 3;
end

function y = demo_B(x)
    y = x^3 - 2 * sin(x);
end

function y = demo_B_prime(x)
    y = 3 * x^2 - 2 * cos(x);
end