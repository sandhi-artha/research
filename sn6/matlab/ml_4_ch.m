% read all 4 channels of each acquisition date see if they're the same size
% only VH has 203 .tif, the others 204, so we'll use VH to list name files

% f_paths = dir('C:\Users\tensorflow\projects\expanded-dataset\*VH*.tif');
% f_names = strings(length(f_paths),1);
% 
% for i = 1:length(f_paths)
%     f_name_cell = split(f_paths(i).name, '_');
%     f_name = append(f_name_cell(end-1),'_',f_name_cell(end));
%     f_names(i) = append(f_name);
% end


%Error notes:
%out of 203 stripes, only
%CAPELLA_ARL_SM_SLC_VH_20190823081648_20190823081959.tif raised error due
%to different shapes in each pol

%for 2x2 ML, each stripe took 3-4m to process, so all finished in ~12h
%for 4x4 ML, 6-8m per stripe, total time ~26h (93185.76s)
%for 8x8 ML, 20-22m per stripe

%open the saved filenames
load('slc_filenames.mat', 'f_names');
dir_path = 'C:\Users\tensorflow\projects\expanded-dataset\CAPELLA_ARL_SM_SLC_*';
out_path = 'C:\Users\tensorflow\projects\processed\';

%test pipeline for 1 acq_date
% f_names = "20190823162315_20190823162606.tif";

%start logging output from command window and append to log.txt
%to check if diary is still on: get(0, 'Diary')
diary('log.txt')

%get current date and time
tnow = now;
dnt = datetime(tnow,'ConvertFrom','datenum');
fprintf('Start job: %s\n', dnt);

filt = [6];
pol = ["HH", "HV", "VH", "VV"];

err = ["Missing filenames"];
stats = struct('f',{}, 'pol',{}, 'pre_norm_min',{}, 'pre_norm_max',{});

%measure time
t_process = tic;

%for all acquistion_date
for ia = 1:length(f_names)
    fprintf('stripe %d from %d\n', ia, length(f_names))
    
    %collect filenames for all 4 polarimetry
    pol_f_names = dir(append(dir_path, f_names(ia)));
    
    % read 1 pol to determine image size
    f_path = append(pol_f_names(1).folder, '\', pol_f_names(1).name);
    [slc, r] = readgeoraster(f_path);
    slc_size = size(slc);
    
    is = 1;  %reset counter for stats item
    
    %for all filters
    for ifl = 1:length(filt)
        fprintf('f%d..\n', filt(ifl))
        %pre-allocate for quad pol processed slc
        slc_pol = zeros(slc_size(1), slc_size(2), 4, 'uint8');
        
        %for all pol
        for ip = 1:4
            f_path = append(pol_f_names(ip).folder, '\', pol_f_names(ip).name);
            fprintf('%s.. ', pol(ip))
            [slc, r] = readgeoraster(f_path);

            %error handling, skip this acq_date if pol not same shape
            if not(isequal(size(slc), slc_size))
                err = [err pol_f_names(ip).name];
                break
            end
        
            %perform multi_look
            img = sarimg_multilook2d(slc, [filt(ifl), filt(ifl)], 'none');
            
            %save into stats
            stats(is).f = filt(ifl);
            stats(is).pol = pol(ip);
            stats(is).pre_norm_min = min(img, [], 'all');
            stats(is).pre_norm_max = max(img, [], 'all');
            is = is+1;
            
            img = norm(img);  %normalize image
            slc_pol(:,:,ip) = img;  %append image  
        end
        
        if isequal(size(slc), slc_size)
            %if no error, save the image
            fprintf('saving image..\n')
            %add to the end _f*.tif
            ext = append('_f', string(filt(ifl)), '.tif');
            %save to out_path and replace the ending
            fn = append(out_path, 'SLC_POL_', f_names(ia));
            fn = strrep(fn, '.tif', ext);
            imwrite(slc_pol, fn);
        end
    end
    
    fprintf('saving stats..\n');
    fn = append(out_path, 'SLC_POL_', f_names(ia));
    fn = strrep(fn, '.tif', '.csv');
    if isfile(fn)
        %if file exist, append to it
        writetable(struct2table(stats), fn, 'WriteMode','Append',...
            'WriteVariableNames',false,'WriteRowNames',true) 
    else
        %if not, create a new file
        writetable(struct2table(stats), fn)
    end
end

fprintf('Total processing time: %f\n', toc(t_process))
%stop logging
diary off


function img = norm(img_in)
    %replace 0 values with 10^-5 so it doesn't go to inf when converted
    img_in(img_in==0) = 1e-5;
    %convert to log scale (dB) ydb = 20*log_10 (y)
    img = mag2db(img_in);
    
    %global norm -100dB to +100dB
%     max_value = 100;  %100dB == magnitude of 10^5
%     img = img/max_value;  % but what to do with negative values?
        
    %local norm
    img = img - min(img, [], 'all');  %shift min val to 0
    img = img / max(img, [], 'all');  %scale max val to 1
    
    %convert to int
    img = img*255;
    img = uint8(img);
end
