% ODE45 verification for NN controller
A = [ 0 1; -1 0];
B = [0 ; 1];
C = [1 0 ];
D = 0;
K = [-2 -2]; % result of algebraic Riccatti for LQR

load tanh30secs10.mat % variable name netTrained

net = netTrained;
% x_dot = A*x + B * NN(x) + B*r(t)   where u(t) = K*x(t) + r(t)
tstart = 1; % step function start time
func = @(t,x)  A*x + B*sim(net, x) + B*stepFunc(t,tstart); % add reference signal stepFunc
tspan = [0 20];
x0 = [0 ; 0];
[t,x] = ode45(func, tspan, x0);
% We've got the evolution of states now. Let's get the output y too.
y = C * x';

plot(t,y)
hold on


% Now transmit data from Simulink to make sure the results are the same

% First of all make sure the Simulink network and loaded one are the same
% Copied the values from the Simulink model
outputLayerWeights = [-0.32100397976799210919551796905579976737499237060546875;0.0341053754801734221313580519563402049243450164794921875;0.136015743648133680121503630289225839078426361083984375;-0.339144292311621686764055993990041315555572509765625;-0.077733991682308600790207719910540618002414703369140625;0.056169843776350626696203249821337522007524967193603515625;0.8722541475835383639747533379704691469669342041015625;0.250538345056389111231709421190316788852214813232421875;0.2255502850244358870046568199541070498526096343994140625;-0.420930547645180064275649556293501518666744232177734375];
loadedWeights = net.LW{2,1}';

if ~all(outputLayerWeights == loadedWeights)
    warning("Neural networks are not identical")
end

tSim = out.simulinkY.Time;
ySim = out.simulinkY.Data;

% but the timeseries object also has a convenient plot method
out.simulinkY.plot()
hold off

legend("ODE45", "Simulink")



function out = stepFunc(t, tstart)
arguments 
    t
    tstart = 0;
end
if t - tstart >= 0
    out = 1;
else
    out = 0;
end
end