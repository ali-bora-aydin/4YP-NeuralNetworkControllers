% Try to solve systems with ODE45 and make sure the output is same 

% Define system as state space matrices
A = [ 0 1; -1 0];
B = [0 ; 1];
C = [1 0 ];
D = 0;

% LQR control parameters
Q = 0.5 * [1 0; 0 0];
R = 1/16;

% Solve optimal control problem
[K, S, P] = lqr(A,B,Q,R);

% Let's just try to solve this basic system

% x_dot = (A+B*K)x + B*r(t)   where u(t) = K*x(t) + r(t)
% y = C*x
func = @(t,x) (A-B*K)*x + B*stepFunc(t); % add reference signal stepFunc
tspan = [0 20];
x0 = [0 ; 0];
[t,x] = ode45(func, tspan, x0);
% We've got the evolution of states now. Let's get the output y too.
y = C * x';
plot(t)
hold on

% Compare with control toolbox solver
sys = ss(A-B*K, B, C, D);
step(sys);
legend("ODE45","ControlSolver")
hold off



 


function out = stepFunc(t)
arguments 
    t = 0 % by default where the step func increases
end
if t >= 0
    out = 1;
else
    out = 0;
end
end