%findings:
%running [2,2] ML on single channel image takes 41s
%running it on 4ch image takes 81s, that's half time!

f_names = "20190823162315_20190823162606.tif";
dir_path = 'C:\Users\tensorflow\projects\expanded-dataset\CAPELLA_ARL_SM_SLC_*';
out_path = 'C:\Users\tensorflow\projects\processed\';

pol = ["HH", "HV", "VH", "VV"];

%measure time
% t_process = tic;

%collect filenames for all 4 polarimetry
pol_f_names = dir(append(dir_path, f_names));

%read HH and get the width and height for pre-allocate
f_path = append(pol_f_names(1).folder, '\', pol_f_names(1).name);
[slc, r] = readgeoraster(f_path);
slc_size = size(slc);

%pre-allocate for quad pol processed slc
slc_pol = zeros(slc_size(1), slc_size(2), 4, 'uint8');

%append 1st image
slc_pol(:,:,1) = slc;

% read and append other 3
for ip = 2:4
    f_path = append(pol_f_names(ip).folder, '\', pol_f_names(ip).name);
    [slc, r] = readgeoraster(f_path);
    slc_pol(:,:,ip) = slc;
end

% fprintf('Total processing time: %f\n', toc(t_process))