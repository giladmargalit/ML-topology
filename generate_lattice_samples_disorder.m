clearvars;

%% SETTINGS

Nx = 4; Ny = Nx;
biased_set = 1;
find_gap = 0;
find_clean_chern = 0;
num_iter = 1e6;
chunks = 100; % must divide num_iter
out_dir = 'path/to/output';

%% LATTICE PARAMETERS

% scale factor
base_Nx = 4;
base_Ny = 4;
scale_factor = ( (Nx/base_Nx)*(Ny/base_Ny) ) ^ (0.5);

% variance of fluctuating parameters
mu_var = 0.3 * scale_factor;
t_minus_var = 0.2;
Delta_var = 0.05;

% fixed parameters
t_plus = 1;

% parameters ranges
min_t_minus = -2;
max_t_minus = 2;
min_Delta = 0.1;
max_Delta = 0.2;
min_mu = -4;
max_mu = 4;

num_sites = Nx*Ny;


%% GENERATE DISORDER REALIZATIONS

    
if biased_set
    % generate 2D probability distribution
    mu_vec = linspace(-2, 2, 1001);
    t_minus_vec = linspace(-2, 2, 1001);
    dist = zeros(length(mu_vec), length(t_minus_vec));
    for mu_idx=1:length(mu_vec)
        for t_idx=1:length(t_minus_vec)
            mu = mu_vec(mu_idx);
            t_minus = t_minus_vec(t_idx);
            dist(mu_idx, t_idx) = min([abs(mu-1),abs(mu+1),...
                abs(mu-t_minus)/sqrt(2),abs(mu+t_minus)/sqrt(2)]);
        end
    end
    prob = (1 ./ (0.03 + dist.^1))';
end

samples_per_chunk = num_iter/chunks;
chern_vec = zeros(1,samples_per_chunk);
gap_vec = zeros(1,samples_per_chunk);
clean_chern_vec = zeros(1,samples_per_chunk);

tic;
parfor idx=0:chunks-1

    data_file = fopen(sprintf('%s\\data_%09d.txt',out_dir,idx), 'at');
    labels_file = fopen(sprintf('%s\\labels_%09d.txt',out_dir,idx), 'at');
    if find_gap
        gaps_file = fopen(sprintf('%s\\gaps_%09d.txt',out_dir,idx), 'at');
    end
    if find_clean_chern
        clean_file = fopen(sprintf('%s\\clean_%09d.txt',out_dir,idx), 'at');
    end

    chern_vec = zeros(1,samples_per_chunk);
    gap_vec = zeros(1,samples_per_chunk);
    clean_chern_vec = zeros(1,samples_per_chunk);

    for idx2=1:samples_per_chunk

        % generate random global values (in biased distribution if needed)
        if biased_set
            [mu_0, t_minus_0] = pinky(mu_vec, t_minus_vec, prob);
            mu_0 = 2*mu_0;
            Delta_0 = min_Delta + (max_Delta - min_Delta)*rand();
        else
            mu_0 = min_mu + (max_mu - min_mu)*rand();
            Delta_0 = min_Delta + (max_Delta - min_Delta)*rand();
            t_minus_0 = min_t_minus + (max_t_minus - min_t_minus)*rand();
        end

        % add disorder to mu
        mu_dis = mu_var*(rand(1,num_sites)-0.5);
        mu_dis = mu_dis - mean(mu_dis);
        mu = t_plus * (mu_0 + mu_dis);

        % add disorder to Delta
        Delta_dis = Delta_var*(rand(1,num_sites)-0.5);
        Delta_dis = Delta_dis - mean(Delta_dis);
        Delta = t_plus * (Delta_0 + Delta_dis);

        % add disorder to t_minus
        t_minus_dis = t_minus_var*(rand(1,num_sites)-0.5);
        t_minus_dis = t_minus_dis - mean(t_minus_dis);
        t_minus = t_plus * (t_minus_0 + t_minus_dis);

        % set actual values for the Hamultonian
        tx = (t_plus + t_minus) / 2;
        ty = (t_plus - t_minus) / 2;
        Delta_x = Delta;
        Delta_y = 1i*Delta;

        H_hop = get_pip_hopping_hamiltonian_anisotropy(Nx,Ny,tx,ty,mu);
        H_pair = get_general_pairing_hamiltonian(Nx,Ny,Delta_x,Delta_y);
        H_BdG = [H_hop H_pair;H_pair' -H_hop];
        
        if find_gap
            [chern_vec(idx2), gap_vec(idx2)] = get_Bott_and_gap(H_BdG,Nx,Ny);
            chern_vec(idx2) = -chern_vec(idx2);
        else
            chern_vec(idx2) = -get_Bott_index(H_BdG,Nx,Ny);
        end
        
        if find_clean_chern
            clean_chern = pip_anisotropy_get_chern(mean(tx),mean(ty),mean(mu));
            clean_chern_vec(idx2) = clean_chern;
        end

        H_input = [mu,t_minus,Delta];

        fprintf(data_file,'%.03f,',H_input);
        fprintf(data_file,'\n');
        fprintf(labels_file,'%d\n',round(chern_vec(idx2)));
        if find_gap
            fprintf(gaps_file,'%.03f\n',gap_vec(idx2));
        end
        if find_clean_chern
            fprintf(clean_file,'%d\n',round(clean_chern_vec(idx2)));
        end
    end
    fclose(data_file);
    fclose(labels_file);
    if find_gap
        fclose(gaps_file);
    end
    if find_clean_chern
        fclose(clean_file);
    end
end

toc;
