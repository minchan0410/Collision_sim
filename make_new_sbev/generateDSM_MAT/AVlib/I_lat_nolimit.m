function [I_lat_out, I_long, TTC, x_p, d_br, d_w,  DLC, TLC]...
    = I_lat_nolimit(rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, Va,Vb,heading_angle, target_width, target_length, ego_length, distance_to_leftlane, distance_to_rightlane, I_LAT_PARAM)
% I_lat Calculates lateral collision index.
%
% I_lat_out = I_lat(rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, heading_angle,...
%                   target_width, target_length, ego_length, distance_to_leftlane, distance_to_rightlane, I_LAT_PARAM)
% rel_pos_x {double}             : Relative longitudinal position (m)
% rel_pos_y {double}             : Relative lateral position (m)
% rel_vel_x {double}             : Relative longitudinal velocity (m/s)
% rel_vel_y {double}             : Relative lateral velocity (m/s)
% heading_angle {double}         : Relative heading angle (rad)
% target_width {double}          : width of target vehicle (m)
% target_length {double}         : length of target vehicle (m)
% ego_length {double}            : length of ego vehicle (m)
% distance_to_leftlane {double}  : distance between ego vehicle center to left lane (m)
% distance_to_rightlane {double} : distance between ego vehicle center to right lane (m)
%
% I_LAT_PARAM {struct}           : Parameters for calculation of lateral collision index
%                                   I_LAT_PARAM.TLC_THRESHOLD : minimum relative lateral position of ROI
%                                   I_LAT_PARAM.T_THINKING : the delay in human response between recognition and manipulation (s)
%                                   I_LAT_PARAM.T_BRAKE : the system delay (s)
%                                   I_LAT_PARAM.A_X_MAX : maximum deceleration of vehicle(negative number) (m/s^2)
%                                   I_LAT_PARAM.X_THRESHOLD : if warning index below threshold, it indicates current situation is dangerous
%                                   I_LAT_PARAM.TTC_INVERSE_THRESHOLD : if TTC inverse beyond threshold threshold, it indecates current situation is dangerous
%                                   I_LAT_PARAM.P_LONG_DEFAULT_VALUE : default value for initialization
%                                   I_LAT_PARAM.TTC_MAX : default value for exception
%                                   I_LAT_PARAM.TLC_MAX : default value for exception
%
% Reference : K. Kim, ¡°Predicted potential risk-based vehicle motion control of automated vehicles for integrated risk management,¡± Doctoral Dissertation, Seoul National University Graduate, 2016.



%--------------------------------------------------------------------------
% Parameter
%--------------------------------------------------------------------------
% parameter in algorithm
TLC_THRESHOLD                = I_LAT_PARAM.TLC_THRESHOLD;
T_THINKING                   = I_LAT_PARAM.T_THINKING;
T_BRAKE                      = I_LAT_PARAM.T_BRAKE;
A_X_MAX                      = I_LAT_PARAM.A_X_MAX;
X_MAX                        = I_LAT_PARAM.X_MAX;
X_THRESHOLD                  = I_LAT_PARAM.X_THRESHOLD;
% X_THRESHOLD=10;
TTC_INVERSE_THRESHOLD        = I_LAT_PARAM.TTC_INVERSE_THRESHOLD;

% parameter for initialization/exception
P_LONG_DEFAULT_VALUE         = I_LAT_PARAM.P_LONG_DEFAULT_VALUE;
TTC_MAX                      = I_LAT_PARAM.TTC_MAX;
TLC_MAX                      = I_LAT_PARAM.TLC_MAX;


%--------------------------------------------------------------------------
% initialization
%--------------------------------------------------------------------------
I_lat_out = 0;
p_long = P_LONG_DEFAULT_VALUE;

%--------------------------------------------------------------------------
% longitudinal collision index
%--------------------------------------------------------------------------
% calculate the clearance between two vehicles(p_long)
if rel_pos_x >= 0
    p_long = rel_pos_x;
else % exception for adjacent and rear vehicle
    p_long = rel_pos_x + ego_length + target_length; %ignore the rear collision(?)
end

% calculate TTC(time to collision) : TTC positive = dangerous, TTC negative = safe
if p_long ~= P_LONG_DEFAULT_VALUE
    if rel_vel_x < 0
        if rel_pos_x >= 0 && p_long >= 0% TTC for front vehicle || TTC for rear vehicle
            TTC = p_long/(-rel_vel_x);
        elseif rel_pos_x <= 0 && p_long <= 0 % rear far
            TTC = TTC_MAX;
        elseif rel_pos_x <= 0 && p_long > 0 % exception : TTC for adjacent vehicle
            TTC = 0.5;
        end
    else % prevent infinite TTC
        TTC = TTC_MAX;
        
        if rel_pos_x < 0 && p_long < 0 % exception : RT_LFL2R_IN rear collision
            TTC = p_long/(-rel_vel_x);
          
        end
        
        if rel_pos_x <= 0 && p_long > 0 % exception : TTC for adjacent vehicle PSD-22-13(1020) LK_CIR_MER_data_2
            TTC = 0.5;
        end
    end
else
    TTC = TTC_MAX;
end
if TTC>TTC_MAX
    TTC=TTC_MAX;
end

% calculate warning braking critical distance(d_br)
d_br = (-rel_vel_x)*T_BRAKE - ((Va)^2-(Vb)^2)/(2*A_X_MAX);

% calculate warning critical distance(d_w)
d_w = (-rel_vel_x)*T_THINKING + (-rel_vel_x)*T_BRAKE - ((Va)^2-(Vb)^2)/(2*A_X_MAX);
if Vb<0
    d_br = (-rel_vel_x)*T_BRAKE + ((Vb)^2+(Va)^2)/(2*A_X_MAX);
    d_w = (-rel_vel_x)*T_THINKING + (-rel_vel_x)*T_BRAKE + ((Vb)^2+(Va)^2)/(2*A_X_MAX);
end
% calculate warning index(x_p)
x_p = (p_long - d_br)/(d_w - d_br);
if p_long<17.5
    p_long=p_long;
end
if x_p > X_MAX ||Va-Vb<0
    x_p = X_MAX;
end
if  x_p < 0
    x_p=0;
end
% calculate collision risk index(I_long)
I_long = max((X_MAX - x_p)/(X_MAX + X_THRESHOLD), abs(1/TTC)/abs(TTC_INVERSE_THRESHOLD));
% if I_long>1
%     I_long=1;
% end

%--------------------------------------------------------------------------
% lateral collision index
%--------------------------------------------------------------------------
% calculate DLC
if rel_pos_y >= distance_to_leftlane % outside of leftlane : calculate DLC, DLC > 0
    DLC = rel_pos_y - (target_length/2*sin(sign(heading_angle)*heading_angle) + target_width/2*cos(heading_angle));%-distance_to_leftlane;
    
    if DLC < 0
        DLC = 0;
    end
elseif rel_pos_y <= distance_to_rightlane % outside of rightlane : calculate DLC, DLC < 0
    DLC = rel_pos_y + (target_length/2*sin(sign(heading_angle)*heading_angle) + target_width/2*cos(heading_angle));%-distance_to_rightlane;
    if DLC > 0
        DLC = 0;
    end
else % in-lane : do not calculate DLC
    DLC = 0;
end

% calculate TLC
if rel_vel_y ~= 0
    if abs(rel_vel_y) > 0.01
        if DLC ~= 0

            TLC =  -(DLC)/(rel_vel_y);
            if TLC<0
                TLC=TLC_MAX;
            end

        else % exception : In case of DLC = 0, 1/TLC is -inf, then min(TLC_THRESHOLD/TLC,1) is -inf
            TLC = 0;
        end
    else % exception : In case of small value of rel vel y, TLC has too big value
        TLC = TLC_MAX;
        if DLC == 0
            TLC = 0;
        end
    end
elseif DLC==0
    TLC = 0;
else % prevent infinite TLC
    TLC = TLC_MAX;
end
% out of FOV
if abs(DLC)<2
    DLC=DLC;
end
% if abs(rel_pos_y)>2
%     TLC=TLC_MAX;
% end
% if abs(DLC)>2||rel_vel_x > -0.2
%     I_long=0;
% end
% calculate lateral collision index
if TLC < 0 % safe
    I_lat_out = 0;
else
    if TLC<TLC_THRESHOLD
        TLC=TLC_THRESHOLD;
    end
    I_lat_out = min(I_long,1) * min(TLC_THRESHOLD/TLC);
end

if rel_vel_x<-10
    I_lat_out=I_lat_out;
end
if I_lat_out>10000
    I_lat_out=I_lat_out;
end



end