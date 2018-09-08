x = 3; % evaluate at point
coefficients = [2 -5 7 -4 1];

n = length(coefficients);
p = 0; % accumulator
for i = n:-1:1
    p = coefficients(int64(i))+  x * p;
end
disp('Evaluation')
disp(p)


% 	Horners synthetic division
r = 2;
divis = zeros(n-1, 1);
for i = n:-1:2
    divis(int64(i - 1)) = coefficients(i);
end
 
for i = n-1:-1:2
    divis(int64(i-1)) = coefficients(i) + r * divis(int64(i));
end
disp('Remainder coefficients')
disp(divis)


%   Horners polynomial factorization
r = 3;

for k = 1:n
    for j = n-1:-1:k
        coefficients(int64(j)) = coefficients(j) + r * coefficients(int64(j+1));
    end
end
disp('Factored coefficients')
disp(coefficients)