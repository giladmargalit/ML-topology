function H = get_pip_hopping_hamiltonian_anisotropy(Nx,Ny,tx,ty,mu)

% Creates an Nx*Ny by Nx*Ny Hamiltonian for the lattice, where Nx and Ny
% are width and height and tx, ty, and mu are lattice parameters. tx, ty,
% and mu are all vectors of shape (1,Nx*Ny) which contain all disorder in
% their channels.

% This function only outputs the particle-particle block of the full BdG
% Hamiltonian, so pairing information is not included.

N_sites = Nx*Ny;
H = zeros(N_sites);

for j=1:N_sites
    
    % chemical potential
    H(j,j) = H(j,j) - 0.5*mu(j);

    % hop down
    j_down = j + 1;
    if (mod(j,Ny) == 0)
        j_down = j_down - Ny;
    end
    H(j,j_down) = H(j,j_down) + ty(j);

    % hop right
    j_right = j + Ny;
    if (j > Ny*(Nx-1))
        j_right = j - Ny*(Nx-1);
    end
    H(j,j_right) = H(j,j_right) + tx(j);
        
end

H = H + H';

end