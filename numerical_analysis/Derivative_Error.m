n = 30;
h = 1;
x = 0.5;
ermin = 1;
 
for i = 1:n
    h = 0.25*h;
    
    % Use the definition of the derivative to approximate
    y = (sin(x + h) - sin(x)) / h;
    
    % Find relative error against the exact value of the derivative
    error = abs(cos(x) - y);
    
    disp(strcat('Iter: ', num2str(i), '  H: ', num2str(h), ' Error: ', num2str(error)))
    % As the difference factor h gets smaller, the approximation for the
    % derivative of sin becomes more accurate. This is shown in the error
    % calculation via a relative difference. 
    
    % h becomes so small that the addition
    % of h in the evaluation of the derivative is truncated, causing a loss
    % of precision that increases in magnitude every iteration.
    
    % Iteration 14 is the optimal tradeoff between
    % h-difference and floating point precision.
    
    if (error < ermin)
        ermin = error;
        imin = i;
    end
end
disp(strcat('imin: ',num2str(imin)))
disp(strcat('ermin:', num2str(ermin)))
 
% Iter:1  H:0.25 Error:0.06873
% Iter:2  H:0.0625 Error:0.015548
% Iter:3  H:0.015625 Error:0.0037811
% Iter:4  H:0.0039063 Error:0.00093861
% Iter:5  H:0.00097656 Error:0.00023423
% Iter:6  H:0.00024414 Error:5.8532e-05
% Iter:7  H:6.1035e-05 Error:1.4631e-05
% Iter:8  H:1.5259e-05 Error:3.6578e-06
% Iter:9  H:3.8147e-06 Error:9.1444e-07
% Iter:10  H:9.5367e-07 Error:2.2861e-07
% Iter:11  H:2.3842e-07 Error:5.7244e-08
% Iter:12  H:5.9605e-08 Error:1.4636e-08
% Iter:13  H:1.4901e-08 Error:4.391e-09
% Iter:14  H:3.7253e-09 Error:3.0596e-09
% Iter:15  H:9.3132e-10 Error:1.1842e-08
% Iter:16  H:2.3283e-10 Error:1.1842e-08
% Iter:17  H:5.8208e-11 Error:1.1842e-08
% Iter:18  H:1.4552e-11 Error:1.1842e-08
% Iter:19  H:3.638e-12 Error:3.8265e-06
% Iter:20  H:9.0949e-13 Error:1.9085e-05
% Iter:21  H:2.2737e-13 Error:0.00014116
% Iter:22  H:5.6843e-14 Error:0.00034713
% Iter:23  H:1.4211e-14 Error:0.0013237
% Iter:24  H:3.5527e-15 Error:0.0025826
% Iter:25  H:8.8818e-16 Error:0.0025826
% Iter:26  H:2.2204e-16 Error:0.12758
% Iter:27  H:5.5511e-17 Error:0.87758
% Iter:28  H:1.3878e-17 Error:0.87758
% Iter:29  H:3.4694e-18 Error:0.87758
% Iter:30  H:8.6736e-19 Error:0.87758
% imin:14
% ermin:3.0596e-09
