



function image_generating(img_num,shot_num)
    %% load distribution
    load('NoiseDist.mat');

    Dist = Dist21298;
    
    cluster_num = randi([80000 100000]);%  randi([40000 80000]);
    
   
    
    %% generating source
    c0 = rand() * 0.1 + 0.07;  %%%%%
    c1 = rand() * 0.4 - 0.2;   c1 = c1/2;%%%%%%%%
    c2 = rand() * 0.92 - 0.46;  c2 = c2/2;
    c3 = rand() * 0.2 - 0.1;   c3 = c3/2;
    c4 = rand() * 0.42 - 0.21;   c4 = c4/2;
    x2 = rand() * 0.2 - 0.1;   x2 = x2/2;
    y2 = rand() * 0.2 - 0.1;   y2 = y2/2;
    peak_intensity = rand()*2000 + 4000;
    
    %%
    GetSource(c0,c1,c2,c3,c4,x2,y2,peak_intensity)    
    
    %% read the clean sim image
    
    sim = imread(['NISreal_19/c0=',num2str(c0,'%.2f'),'_c1=', num2str(c1,'%.2f') ,'_c2=', num2str(c2,'%.2f') ,'_c3=', num2str(c3,'%.2f') ,'_c4=', num2str(c4,'%.2f') ,'_x2=', num2str(x2,'%.2f') ,'_y2=', num2str(y2,'%.2f'),'_source.tif']);
    sim = double(sim);
    sim = sim+500;
    
    
    %% produce kernel noise
    
     %%%%%%%should be a random number

    load('2D_kernel.mat');
    
    PSF = s23075;
    
    noise = randn(512);   
    
    kernel_scale = std2(noise)/std2(conv2(noise,PSF,'same'));
    
    %scalar = sqrt(sim)/2.2*kernel_scale;
    scalar = sqrt(sim)/1.1*kernel_scale;
    
    kernel_noise = conv2(noise.*scalar,PSF,'same');   
    
    
    bkgroundvalue = 60; 
    
    N = 512;

    

    image = kernel_noise - min(min(kernel_noise));
    intensity_corr_ratio = -min(min(kernel_noise))/bkgroundvalue;

    
    cleanbkg = image;


    N = 512;

    %%
    %%%%%%%%%%%%%threshold_factor: used to do threshold segmentation for
    %%%%%%%%%%%%%clusters
    threshold_factor = 3.0;

    [IMmean,yindex] = find(Dist==max(max(Dist)));
    IMmean = IMmean/threshold_factor;


    [Ygrid,Xgrid] = size(Dist);
    
    Xin = 1:1:Xgrid;
    Yin = 1:1:Ygrid;
    

    %count = [];

    for i = 1:cluster_num

        
        % Random intensity for the speckle
        [speckle_size,intensity_abs] = pinky(Xin,Yin,Dist);
        speckle_size_cor_factor = 2.0;
        speckle_size = ceil(speckle_size/speckle_size_cor_factor);

        %intensity = intensity_abs/avgbkg;  %%%%%%%%%%%%there should be a random number
        intensity = intensity_abs/IMmean/intensity_corr_ratio/5.0;
        
        % Create a blank image for the speckle
        speckle = zeros(N);
    
        % Random initial position for the random walk
        x = randi([2, N-1]);
        y = randi([2, N-1]);
        x0 = x;
        y0 = y;
            
        

        %count(i) = intensity_abs;


    
        % Choose whether this speckle is a straight line
        is_straight_line = rand() < 0.5;

        if is_straight_line
    
            %peckle_size = randi([6, 15]);
            % Random angle for the straight line
            angle = rand() * 2 * pi;
    
            % Perform the random walk in a straight line
            for j = 1:speckle_size
                % Round the position to the nearest integer
                xi = round(x);
                yi = round(y);
                
                tail_intensity_ratio = randi([2 8]);
                % Calculate the decayed intensity
                decayed_intensity = intensity/j;

                %{
                if j == 1
                    decayed_intensity = intensity;
                else 
                    decayed_intensity = intensity/tail_intensity_ratio;
                end
                %}
    
                % Add the current position to the speckle
                if xi >= 1 && xi <= N && yi >= 1 && yi <= N
                    speckle(xi, yi) = decayed_intensity;
                end
    
                % Move to the next position
                x = x + cos(angle);
                y = y + sin(angle);
            end
       
        
        else
            % Perform the random walk
            for j = 1:speckle_size
            % Add the current position to the image
                decayed_intensity = intensity * (1 - (j-1) / speckle_size);
                speckle(x, y) = decayed_intensity;
    
            % Randomly move to a neighboring pixel (up, down, left, or right)
                direction = randi([1, 4]);
                switch direction
                    case 1  % up
                        x = max(x - 1, 1);
                    case 2  % down
                        x = min(x + 1, N);
                    case 3  % left
                        y = max(y - 1, 1);
                    case 4  % right
                        y = min(y + 1, N);
                    end
            end
        end
    
        % Apply a small Gaussian filter to the speckle


        
        
  
            
        
        
        %%%%%%%halo part
        halo_sigma = power(speckle_size,1/8.0)*power(intensity,1/8.0);
        bkgspeckle = imgaussfilt(speckle,halo_sigma);
        % those two factor decide the gradient and apperance of the cluster, how blur it is. import to tune
        bkgspeckle_factor = 0.8;
        
        

        % keep speckle size unchanged after the filtering, look for the
        % biggest speckle_size elements and keep them
        

        

        if is_straight_line
            sigma = rand()*0.5 + 0.3;
            speckle_kernel = imgaussfilt(speckle,sigma); 
            ratio = intensity / max(max(speckle_kernel));
            %final_kernel = speckle_kernel*(ratio - 1/bkgspeckle_factor);
            final_kernel = speckle_kernel*ratio;
        else
            sigma = rand()*0.5 + 2.5;
            speckle_kernel = imgaussfilt(speckle,sigma); 
            ratio = intensity / max(max(speckle_kernel));
            [vals, index] = sort(speckle_kernel(:), 'descend');
            selected_index = index(1:ceil(speckle_size));
            final_kernel = 0 * speckle_kernel;   %same size and type but all 0
            %final_kernel(selected_index) = speckle_kernel(selected_index)*(ratio - 1/bkgspeckle_factor);
            final_kernel(selected_index) = speckle_kernel(selected_index)*ratio;
        end

       

        
        %final_speckle_kernel = final_kernel + 1 + bkgspeckle/(bkgspeckle_factor) ;  %%%%%%factor for bkgspeckle should be a random number
        final_speckle_kernel = final_kernel + 1;

        image = image.* final_speckle_kernel;

        %size(small_image)
        

        disp([num2str(i),'/',num2str(cluster_num),'   speckle size:',num2str(speckle_size),'   intensity:',num2str(intensity),'   halosigma:',num2str(halo_sigma)]);
    end
    

    


    
    %%%electronic noise%%%%
    E_noise = randn(512) * 5;
    
    finalsim = E_noise + sim + image;    
    final_save = finalsim - 500 + min(min(kernel_noise)) + bkgroundvalue;
    
    final_save(final_save<0) = 0;
    
    
    % 在 fun_s21297.m 中修改第239行代码
    shot_num_str = shot_num; 

    % 构建路径
    folder_path = fullfile('simulated_images', shot_num_str, 'sim', 'noisy');
    filename = ['noisy_', shot_num_str, '_', num2str(img_num), 'nclst_', num2str(cluster_num), '.tif'];
    full_path = fullfile(folder_path, filename);

    % 创建目录（如果不存在）
    if ~exist(folder_path, 'dir')
        mkdir(folder_path);
    end

    % 保存图像
    imwrite(uint16(final_save), full_path);
    
    cleansim = E_noise + sim + cleanbkg;
    clean_save = cleansim - 500 + min(min(kernel_noise)) + bkgroundvalue;
    clean_save(clean_save<0) = 0;
    % 原代码（第257行）替换为以下内容：
    shot_num = 's21297';  % 确保 shot_num 是字符串
    folder_path = fullfile('simulated_images', shot_num, 'sim', 'clean');
    filename = ['clean_', shot_num, '_', num2str(img_num), 'nclst_', num2str(cluster_num), '.tif'];
    full_path = fullfile(folder_path, filename);

    % 创建目录（如果不存在）
    if ~exist(folder_path, 'dir')
        mkdir(folder_path);
    end

    % 写入图像
    imwrite(uint16(clean_save), full_path);

    
    cleanorg = sim;
    cleanorg_save = cleanorg - 500  + bkgroundvalue;
    % 定义路径参数（假设 shot_num 是字符串，如 's21297'）
    folder_path = fullfile('simulated_images', shot_num, 'sim', 'cleanorg');
    filename = ['cleanorg_', shot_num, '_', num2str(img_num), 'nclst_', num2str(cluster_num), '.tif'];
    full_path = fullfile(folder_path, filename);

    % 创建目录（如果不存在）
    if ~exist(folder_path, 'dir')
        mkdir(folder_path);
    end

    % 保存图像
    imwrite(uint16(cleanorg_save), full_path);
end

   