function H = get_general_pairing_hamiltonian(Nx,Ny,Delta_x,Delta_y)

% Creates a full disordered BdG Hamiltonian including pairing for an Nx by
% Ny lattice. Delta_x and Delta_y are vectors of size (1,Nx*Ny) which
% include all disorder information in the pairing channel.

N_sites = Nx*Ny;
H = zeros(N_sites);

for j=1:N_sites

    % pair down
    j_down = j + 1;
    if (mod(j,Ny) == 0)
        j_down = j_down - Ny;
    end
    H(j,j_down) = H(j,j_down) + Delta_y(j);

    % pair right
    j_right = j + Ny;
    if (j > Ny*(Nx-1))
        j_right = j - Ny*(Nx-1);
    end
    H(j,j_right) = H(j,j_right) + Delta_x(j);
        
end

H = 0.5*(H - transpose(H));

end