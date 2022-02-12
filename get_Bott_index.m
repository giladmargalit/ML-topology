% ======================================================================
% This function takes a BdG Hamiltonian H defined on a lattice of Nx by Ny
% points (so size(H) = (2*Nx*Ny),(2*Nx*Ny)) and returns its Chern number,
% calculated using the Bott index formula.
%
% !! NOTE: the current version of this code ONLY works for BdG
% Hamiltonians, as the projector operator P takes all negative energy
% states (lower half of all states). DO NOT use it on a non-BdG
% Hamiltonian. In the absence of particle-hole symmetry a different version
% of this function must be written, taking into account the chemical
% potential (or simply the band one is interested in).
%
% (*) Compared to the OLD version (OLD_get_Bott_index), the change is the
% basis: in the old version we work in the position basis where the x_op
% and y_op are trivially diagonal, but the projection operator P is
% complicated. Here we work in the eigenbasis of H, so P is trivial (unity
% for negative states, zero for positive states) and we have to transform
% the position operators in the following way:
% (Psi') * H * (Psi) is diagonal ==> the transformation of the operators
% must be exp_x_op -> (Psi') * exp_x_op * (Psi) and the same for y.
% This basis change makes the algorithm about 8x faster, since it
% eliminated the need for a for loop iterating over the states.
% (*) I also got rid of the foor loop in the generation of x_op and y_op by
% simply using meshgrid (this is consistent with the way my lattice sites
% are labeled, be cautious if conventions change).
%
% REFERENCES:
% 1) The original paper that suggested this algorithm is "Disordered
% topological insulators via C*-algebras" by Loring and Hastings:
% http://iopscience.iop.org/article/10.1209/0295-5075/92/67004/pdf
% 2) T.A. Loring himself summarized the method in a (relatively) short
% response to a question in physics stack exchange: 
% https://physics.stackexchange.com/questions/378804/how-to-calculate-chern-number-of-a-band-numerically
% 3) The paper "Topological Photonic Quasicrystals: Fractal Topological
% Spectrum and Protected Transport" by Bandres, Rechtsman, and Segev used
% this method and they put a short explanation in page 6:
% https://journals.aps.org/prx/pdf/10.1103/PhysRevX.6.011016
%
% ======================================================================


function chern = get_Bott_index(H,Nx,Ny)

[Psi,~] = eig(H,'vector');

% build normalized position operators
[X,Y] = meshgrid(1:Nx,1:Ny);
x_op = [X(:);X(:)]/Nx;
y_op = [Y(:);Y(:)]/Ny;
exp_x_op = (Psi') * diag(exp(1i*2*pi*x_op)) * Psi;
exp_y_op = (Psi') * diag(exp(1i*2*pi*y_op)) * Psi;

% build projector P
P = [eye(size(H)/2) zeros(size(H)/2);
    zeros(size(H,1)/2,size(H,2))];

% build U,V operators
U = P * exp_x_op * P + (eye(size(P))-P);
V = P * exp_y_op * P + (eye(size(P))-P);

% get Bott index
M = V * U * (V') * (U');
chern = -imag(sum(log(eig(M)))) / (2*pi);

end