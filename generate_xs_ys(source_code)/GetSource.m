
function Iclean = GetSource(c0,c1,c2,c3,c4,x2,y2,peak_intensity)


    array_source_1_size = 512;
    
    % declare the variables to be used
    %{
    c0 = 0.3;
    c1 = 0.2;
    c2 = 0.46;
    c3 = 0.1;
    c4 = -0.21;
    x2 = 0.10;
    y2 = -0.10;
    %}
    
    %{
    c0 = 0.12;
    c1 = 0.0;
    c2 = 0.0;
    c3 = 0.0;
    c4 = 0.0;
    x2 = 0.0;
    y2 = 0.0;
    %}
    
    c0 = c0;
    c1 = c1;
    c2 = c2;
    c3 = c3;
    c4 = c4;
    x2 = x2;
    y2 = y2;
    peak_intensity = peak_intensity;
    
    Array_source_polar = zeros(1000, 1000); % initialize array for storing polar source
    Array_source_theta = zeros(1000, 1); % initialize array for storing thetas
    rho2 = linspace(0, 2, 1000); % create a linearly spaced vector for rho2
    
    % loop to fill the polar source array and theta array
    for i = 1:1000
        theta = 2 * pi * (i - 1) / 1000; % calculate theta
        rho1 = get_value_legendre(cos(theta), c0, c1, c2, c3, c4); % get the value of rho1
        crood_x = rho1 * cos(theta); % get the x coordinate
        crood_y = rho1 * sin(theta); % get the y coordinate
        Array_source_theta(i) = calculate_theta(crood_x - x2, crood_y - y2); % store the calculated theta
        sigma_square = law_cosines(crood_x, crood_y, x2, y2); % calculate sigma_square using law of cosines
        Array_source_polar(i, :) = exp(log(0.17) * (rho2 .^ 2) / sigma_square); % store the calculated polar source
    end
    
    % initialize the 2D array source
    Array_source_1 = zeros(array_source_1_size, array_source_1_size);
    
    % loop to fill the 2D array source
    for a = 1:array_source_1_size
        x_a = -1.0 + 2.0 / array_source_1_size * (a - 1); % calculate x_a
        for b = 1:array_source_1_size
            y_b = -1.0 + 2.0 / array_source_1_size * (b - 1); % calculate y_b
            rho3 = sqrt(x_a ^ 2 + y_b ^ 2); % calculate rho3
            if rho3 < 0.01 % condition for rho3
                Array_source_1(a, b) = 1; % if condition is true, set array value to 1
            else
                theta_ab = calculate_theta(x_a, y_b); % calculate theta_ab
                index = get_suit_theta(theta_ab, Array_source_theta); % get suitable index for theta_ab
                Array_source_1(a, b) = Array_source_polar(index, floor(int32(sqrt(x_a ^ 2 + y_b ^ 2) / (2 / 1000)))); % set value in the array
            end
        end
    end
    
    Array_source_1 = max(Array_source_1, 0); % get the maximum between array values and 0
    rootname = ['c0=' num2str(c0, '%.2f') ... % create a rootname string based on the values of variables
               '_c1=' num2str(c1, '%.2f') ...
               '_c2=' num2str(c2, '%.2f') ...
               '_c3=' num2str(c3, '%.2f') ...
               '_c4=' num2str(c4, '%.2f') ...
               '_x2=' num2str(x2, '%.2f') ...
               '_y2=' num2str(y2, '%.2f')];
    
    FirstFlodername = 'NISreal_19'; % set the name of the first folder
    SecondFlodername = fullfile(FirstFlodername ); % set the name of the second folder
    tifname_source = [rootname '_source.tif']; % set the name of the tif file
    if ~exist(SecondFlodername, 'dir')
       mkdir(SecondFlodername) % create the second folder if it doesn't exist
    end
    imwrite(uint16(Array_source_1 * peak_intensity), fullfile(SecondFlodername, tifname_source)); % write the array as an image to the tif file
    
    %{
    cnt_array = Array_source_1 == 0; % count the number of zeros in the array
    len_array_source = array_source_1_size^2 - sum(cnt_array, 'all'); % get the length of the array source
    Array_source = zeros(len_array_source, 4); % initialize the array source
    k = 1; % initialize counter
    for i = 1:array_source_1_size
        for j = 1:array_source_1_size
            if Array_source_1(i, j) % condition for array value
                Array_source(k, 1) = -0.1 + 0.002 * (i - 1); % calculate and set value in the array source
                Array_source(k, 2) = -0.1 + 0.002 * (j - 1); % calculate and set value in the array source
                Array_source(k, 3) = 0; % set value in the array source
                Array_source(k, 4) = Array_source_1(i, j); % set value in the array source
                k = k + 1; % increment counter
            end
        end
    end
    %}
    %%

end





% define a function to calculate the angle theta using x and y
function theta = calculate_theta(x, y)
    theta = acos(x / sqrt(x ^ 2 + y ^ 2)); % calculate theta
    if y < 0
        theta = 2 * pi - theta; % adjust theta value if y < 0
    end
end

% define a function to calculate square of c, using law of cosines
function c_square = law_cosines(x1, y1, x2, y2)
    c_square = x1 ^ 2 + y1 ^ 2 + x2 ^ 2 + y2 ^ 2 - 2 * (x1 * x2 + y1 * y2);
end

% define a function to get the index of the suitable theta from the array of thetas
function index = get_suit_theta(theta1, array_theta)
    [~, index] = min(abs(theta1 - array_theta)); % find the minimum difference
end

% define a function to get the value of legendre polynomials with coefficients
function num_legrd = get_value_legendre(x, c0, c1, c2, c3, c4)
P0 = legendre(0, x); % calculate 0th order Legendre Polynomial
P1 = legendre(1, x); % calculate 1st order Legendre Polynomial
P2 = legendre(2, x); % calculate 2nd order Legendre Polynomial
P3 = legendre(3, x); % calculate 3rd order Legendre Polynomial
P4 = legendre(4, x); % calculate 4th order Legendre Polynomial

num_legrd = c4 * c0 * P4(1,:) ... % calculate the linear combination of Legendre Polynomials
            + c2 * c0 * P2(1,:) ...
            + c0 * P0(1,:) ...
            + c1 * c0 * P1(1,:) ...
            + c3 * c0 * P3(1,:);
end




