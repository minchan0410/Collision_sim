function [HONDA_w, HONDA_br] = HONDA(rel_pos_x, rel_pos_y, rel_vel_x,Va,Vb, HONDA_PARAM)
%  HONDA calculates Honda's time to collision algorithm in ROI.
%
% HONDA_out = HONDA(rel_pos_x, rel_pos_y, rel_vel_x, ROI)
% rel_pos_x {double} : Relative longitudinal position (m)
% rel_pos_y {double} : Relative lateral position (m)
% rel_vel_x {double} : Relative longitudinal velocity (m/s)
% HONDA_PARAM {struct} : Parameters for calculation of Honda algorithm
%                       HONDA_PARAM.ROI.Y_MIN : minimum relative lateral position of ROI
%                       HONDA_PARAM.ROI.Y_MAX : maximum relative lateral position of ROI
%                       HONDA_PARAM.ROI.X_MIN : minimum relative longitudinal position of ROI
%                       HONDA_PARAM.ROI.X_MAX : maximum relative longitudinal position of ROI
%                       HONDA_PARAM.HONDA_MIN   : default value for exception
TAU1=0.1;
TAU2=0.2;
ALPHA1=10;
ALPHA2=10;
TTC_THRESHOLD = 2.2;
SAFE_MARGIN = 6.2;
VELOCITY_THRESHOLD = 11.67;
if rel_pos_y >= HONDA_PARAM.ROI.Y_MIN && rel_pos_y <= HONDA_PARAM.ROI.Y_MAX && ...
        rel_pos_x >= HONDA_PARAM.ROI.X_MIN && rel_pos_x <= HONDA_PARAM.ROI.X_MAX
    HONDA_w =-TTC_THRESHOLD*rel_vel_x+SAFE_MARGIN;
else
    HONDA_w = HONDA_PARAM.HONDA_MIN;
end
if Va>=VELOCITY_THRESHOLD
    HONDA_br=TAU2*(-rel_vel_x)-TAU1*TAU2*ALPHA1-0.5*ALPHA1*(TAU1)^2;
else
    HONDA_br=TAU2*(Va)-0.5*ALPHA1*(TAU2-TAU1)^2-(Vb)^2/(2*ALPHA2);
end
    if Vb<-.1
    HONDA_br=TAU2*(Va)-0.5*ALPHA1*(TAU2-TAU1)^2+(Vb)^2/(2*ALPHA2);
    end
if HONDA_br<0
    HONDA_br=0;
end
end   
