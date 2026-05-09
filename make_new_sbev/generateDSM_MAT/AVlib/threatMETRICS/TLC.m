function [TLC_out, DLC_out] = TLC(rel_pos_y, rel_vel_y, heading_angle, target_width, target_length, distance_to_leftlane, distance_to_rightlane, TLC_PARAM)
% TLC Calculates Time to Line Crossing.
%
% [TLC_out, DLC_out] = TLC(rel_pos_y, rel_vel_y, heading_angle, target_width,...
%                           target_length, distance_to_leftlane, distance_to_rightlane, TLC_PARAM)
% rel_pos_y {double}             : Relative lateral position (m)
% rel_vel_y {double}             : Relative lateral velocity (m/s)
% heading_angle {double}         : Relative heading angle (rad)
% target_width {double}          : width of target vehicle (m)
% target_length {double}         : length of target vehicle (m)
% distance_to_leftlane {double}  : distance between ego vehicle center to left lane (m), positive number
% distance_to_rightlane {double} : distance between ego vehicle center to right lane (m), negative number
%
% TLC_PARAM {struct}           : Parameters for calculation of time to line crossing
%                                   TLC_PARAM.TLC_MAX : default value for exception
%
% Reference : Mammar, Said, S?bastien Glaser, and Mariana Netto. "Time to line crossing for lane departure avoidance: A theoretical study and an experimental setting." IEEE Transactions on intelligent transportation systems 7.2 (2006): 226-241.



%--------------------------------------------------------------------------
% Parameter
%--------------------------------------------------------------------------
% parameter in algorithm

% parameter for initialization/exception
TLC_MAX                      = TLC_PARAM.TLC_MAX;
TLC_Y_MIN                    = TLC_PARAM.TLC_Y_MIN;
%--------------------------------------------------------------------------
% initialization
%--------------------------------------------------------------------------
TLC_out = 0;
DLC_out = 0;

% calculate DLC
if rel_pos_y >= distance_to_leftlane % outside of leftlane : calculate DLC( DLC > 0 )
    DLC_out = rel_pos_y - (target_length/2*sin(sign(heading_angle)*heading_angle) + target_width/2*cos(heading_angle) + distance_to_leftlane );
    
    if DLC_out < 0
        DLC_out = 0;
    end
elseif rel_pos_y <= distance_to_rightlane % outside of rightlane : calculate DLC( DLC < 0 )
    DLC_out = rel_pos_y + (target_length/2*sin(sign(heading_angle)*heading_angle) + target_width/2*cos(heading_angle) - distance_to_rightlane );
    if DLC_out > 0
        DLC_out = 0;
    end
else % in-lane : do not calculate DLC
    DLC_out = 0;
end

% calculate TLC
if rel_vel_y ~= 0
    if abs(rel_vel_y) > 0.01
        if DLC_out ~= 0
            TLC_out = - DLC_out/rel_vel_y;
        else % exception : In case of DLC = 0, 1/TLC is -inf
            TLC_out = 0;
        end
    else % exception : In case of small value of rel vel y, TLC has too big value
        TLC_out = TLC_MAX;
    end

else % prevent infinite TLC
    if DLC_out == 0
        TLC_out = 0;
    else
        TLC_out = TLC_MAX;
    end
    
end

if TLC_out<0
    TLC_out = TLC_MAX;
end

if abs(rel_pos_y) < TLC_Y_MIN
    TLC_out=0;
end
end