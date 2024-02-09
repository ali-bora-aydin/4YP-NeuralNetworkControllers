function out = reluu(A)

fun = @(x) max(0,x);
out = arrayfun(fun, A);

end