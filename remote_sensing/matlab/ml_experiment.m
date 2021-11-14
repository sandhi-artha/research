%findings:
%running [2,2] ML on single channel image takes 41s
%running it on 4ch image takes 81s, that's half time!
%turns out it's INCORRECT, bcz results have same value in all channel

f_names = "20190823162315_20190823162606.tif";
dir_path = 'C:\Users\tensorflow\projects\expanded-dataset\CAPELLA_ARL_SM_SLC_*';
out_path = 'C:\Users\tensorflow\projects\processed\';

pol = ["HH", "HV", "VH", "VV"];




%collect filenames for all 4 polarimetry
pol_f_names = dir(append(dir_path, f_names));

% collevt 4 pol then ML on all
t_process = tic;
slc_pol = get_quad_pol_slc(pol_f_names);
fprintf('time collect quad pol: %f\n', toc(t_process))

t_process = tic;
img_pol = sarimg_multilook2d(slc_pol, [2,2], 'none');
fprintf('time process quad pol: %f\n', toc(t_process))




% usual way
t_process = tic;

% read 1 pol to determine image size
f_path = append(pol_f_names(1).folder, '\', pol_f_names(1).name);
[slc, r] = readgeoraster(f_path);
slc_size = size(slc);

%pre-allocate for quad pol processed slc
img_pol_trad = zeros(slc_size(1), slc_size(2), 4, 'double');

for ip = 1:4
    f_path = append(pol_f_names(ip).folder, '\', pol_f_names(ip).name);
    [slc, r] = readgeoraster(f_path);
    %perform multi_look
    img = sarimg_multilook2d(slc, [2,2], 'none');
    img_pol_trad(:,:,ip) = img;  %append image  
end

fprintf('time process quad pol traditional: %f\n', toc(t_process))


%eval, in img_pol_sub1 and 2 will have same value
%its incorrect, something wrong with the ML algo only using 1st ch
img_pol_sub1 = img_pol(21001:21100,1001:1100,1);
img_pol_sub2 = img_pol(21001:21100,1001:1100,2);
img_pol_trad_sub1 = img_pol_trad(21001:21100,1001:1100,1);
img_pol_trad_sub2 = img_pol_trad(21001:21100,1001:1100,2);


function slc_pol = get_quad_pol_slc(pol_f_names)
    %read HH and get the width and height for pre-allocate
    f_path = append(pol_f_names(1).folder, '\', pol_f_names(1).name);
    [slc, r] = readgeoraster(f_path);
    slc_size = size(slc);

    %pre-allocate for quad pol slc
    slc_pol = zeros(slc_size(1), slc_size(2), 4, 'double');

    %append 1st image
    slc_pol(:,:,1) = slc;

    % read and append other 3
    for ip = 2:4
        f_path = append(pol_f_names(ip).folder, '\', pol_f_names(ip).name);
        [slc, r] = readgeoraster(f_path);
        slc_pol(:,:,ip) = slc;
    end
end