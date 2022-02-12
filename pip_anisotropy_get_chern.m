function C = pip_anisotropy_get_chern(tx,ty,mu)
% Finds the Chern number for a clean system

transition_points = 2*[tx+ty,tx-ty,-tx+ty,-tx-ty];
transition_points = sort(transition_points);

if (mu <= transition_points(1) || mu >= transition_points(4))
    C = 0;
else
    if (mu <= transition_points(2))
        C = 1 - 2*logical(abs(tx - ty) > abs(tx + ty));
    else
        if (mu <= transition_points(3))
            C = 0;
        else
            C = -1 + 2*logical(abs(tx - ty) > abs(tx + ty));
        end
    end
end

end