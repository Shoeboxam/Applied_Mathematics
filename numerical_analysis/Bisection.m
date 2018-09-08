disp(bisect(@demo_A, 0, 1, 10))
% Output:   0.3333

disp(bisect(@demo_B, 4, 5, 10))
% Output:   4.4933
% The upper bound continued to correct itself until it became positive.
% Once the upper bound was corrected, the root was found with bisection.

disp(bisect(@demo_B, 1, 2, 10))
% Output:   1.5706
% This is a false zero- while the sign does change, the output of the
% function diverges to negative and positive infinity.

function root = bisect(funct, a, b, n)
    a_sign = funct(a) > 0;
    b_sign = funct(b) > 0;
    
    % In the case where the root exists and both sides of the interval are
    % the same, decrease the upper bound until a sign is flipped.
    iter = 0;
    while a_sign == b_sign && iter < n
        b = b - (a - b)/n;
        b_sign = funct(b);
    end
    
    % Actual bisection method with a single function call
    for i = 0:n
        c = (a + b) / 2;
        if funct(c) > 0 == a_sign
            a = c;
        else
            b = c;
        end
    end
    % Return midpoint of interval
    root = (a + b) / 2;
end

function y = demo_A(x)
    y = 9*x^4 + 18*x^3 + 38*x^2 - 57*x + 14;
end

function y = demo_B(x)
    y = tan(x) - x;
end