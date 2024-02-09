% NOTES ABOUT SANITY CHECK OBSERVATIONS due to mistakes:

% 1. Using wrong state-space matrices makes the problem infeasible
% 2. Using wrong activation function makes the problem infeasible


% Define state space matrices
m = 3;
k = 3;
b = 6;
A = [ 0, 1; -k/m, -b/m];
B = [0 ; 1/m];
C = [1 0];
D = 0;
cont_ss = ss(A,B,C,D);
% Convert continous time matrices to discrete-time
dt = 0.01;
disc_ss = c2d(cont_ss, dt, 'zoh');
A = disc_ss.A;
B = disc_ss.B;
C = disc_ss.C;
D = disc_ss.D;


ng = size(A,2); % number of states

% Load the network
load("C:\Users\Bora\OneDrive - Nexus365\Desktop\4YP MATLAB\SDP\sdpReLU.mat" , "net")

% Load the weights and biases
W1 = double(net.Layers(2).Weights);
W2 = double(net.Layers(4).Weights);
W3 = double(net.Layers(6).Weights);
% biases are vectors of zeros actually
b1 = double(net.Layers(2).Bias);
b2 = double(net.Layers(4).Bias);
b3 = double(net.Layers(6).Bias);

% Get dimensions
n1 = size(W1,1);
n2 = size(W2,2);
n3 = size(W3,3);
nphi = n1 + n2;

% Create N matrix: N_ux, N_uw, (no N_ub etc since biases are zero)
% N = blkdiag(W1,W2,W3); % since biases are zero

N = [zeros(n3, size(W1,2) + size(W2,2)) , W3; ...
     W1, zeros( n1, size(W2,2)+size(W3,2) ) ; ...
     zeros(n2, size(W1,2)), W2, zeros( n2, size(W3,2))];

% NOTE: I implemented the ordering of N differently
Nux = N(1:n3, 1:size(W1,2));
Nuw = N(1:n3, size(W1,2)+1 : end);
Nvx = N(n3+1:end, 1:size(W1,2) );
Nvw = N(n3+1:end, size(W1,2)+1 : end );


% Compute equilibrium values
x_eq = [0.0 ; 0.0];
v1_eq = W1*x_eq + b1;
w1_eq = reluu(v1_eq);
v2_eq = W2*w1_eq + b2;
w2_eq = reluu(v2_eq);
v3_eq = W3*w2_eq + b3;
% u_eq = v3_eq

%% Compute lower and upper bounds:
% I will do this very simply now leading to conservative estimate
% Just set all alphas to be 0, and betas to be
% The simplest sector that will cover all ReLU space
% Also pretty bad
% The computation of bounds are redundant now but keep for future


% 1. Select bounds for eq. value at first NN layer for v1: symmetric
%delta_v1 = 0.5;
delta_v1 = 0.001; % I chose smaller to see maybe if it solves the problem
v1_ub = v1_eq + delta_v1;
v1_lb = v1_eq - delta_v1;

alpha1 = zeros(size(v1_eq));
% 2. Use selected bounds on v1 to compute intervals of w1:
w1_lb = reluu(v1_lb);
w1_ub = reluu(v1_ub);

% 3. Use w1 bounds to compute bounds for next input v2: Solution to optim.
c = 0.5 * (w1_lb + w1_ub);
r = 0.5 * (w1_ub - w1_lb);
v2_ub = W2*c + b2 + abs(W2)*abs(r);
v2_lb = W2*c + b2 - abs(W2)*abs(r);

alpha2 = zeros(size(v2_eq));

w2_lb = reluu(v2_lb);
w2_ub = reluu(v2_ub);

Alpha = blkdiag(diag(alpha1), diag(alpha2));
% note: nphi = n1 + n2
beta = 1;
Beta = beta * eye(nphi);


%% Convex Optimization - compute ROA
% to add cvx to path run below
% addpath("C:\Users\Bora\OneDrive - Nexus365\Desktop\C20\cvx")
cvx_clear
cvx_begin sdp quiet
    cvx_solver mosek

    % Variables
    variable P(ng,ng) symmetric;
    % This is diag(lambda) in M_phi
    variable Lambda(nphi,nphi) diagonal;
    
    % P positive definite constraint
    P >= 10^-8 * eye(ng);

    % Lambdas need to be all non-negative: S-procedure
    Lambda(:) >= 0;

    %%% AFTER THIS SECTION: now these are my attempts
    % Eq 18 setup
    R_v = [eye(ng), zeros(ng, nphi); Nux, Nuw];
    R_phi = [Nvx, Nvw; zeros(nphi, ng), eye(nphi)];
    
    M_phi = [zeros(nphi), Lambda; Lambda, zeros(nphi)];
    Psi_phi = [Beta, -eye(nphi); -Alpha, eye(nphi)];
    
    % Eq 18 itself:
    R_v'*[A'*P*A - P , A'*P*B ; B'*P*A, B'*P*B]*R_v + R_phi'*Psi_phi'*M_phi*Psi_phi*R_phi <= 1e-10 * eye(nphi+ng);
    
    % Eq 19 for loop: i = 1,...,n1
    for i = 1:n1
        [delta_v1^2 , W1(i,:) ; W1(i,:)', P] >= 0;
    end

    cvx_end

    % If we could find a P that is not NaN (also lambda but it has weird
    % data structure)
    if all(~isnan(P), "all")
        disp("The SDP is feasible")
    else
        disp(":(")
    end

