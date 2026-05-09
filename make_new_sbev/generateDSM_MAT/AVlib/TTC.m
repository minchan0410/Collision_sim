function TTC_out = TTC(rel_pos_x, rel_pos_y, rel_vel_x, TTC_PARAM)
% TTC Calculates time to collision in ROI.
%
% TTC_out = TTC(rel_pos_x, rel_pos_y, rel_vel_x, ROI)
% rel_pos_x {double} : Relative longitudinal position (m)
% rel_pos_y {double} : Relative lateral position (m)
% rel_vel_x {double} : Relative longitudinal velocity (m/s)
% TTC_PARAM {struct} : Parameters for calculation of TTC
%                       TTC_PARAM.ROI.Y_MIN : minimum relative lateral position of ROI
%                       TTC_PARAM.ROI.Y_MAX : maximum relative lateral position of ROI
%                       TTC_PARAM.ROI.X_MIN : minimum relative longitudinal position of ROI
%                       TTC_PARAM.ROI.X_MAX : maximum relative longitudinal position of ROI
%                       TTC_PARAM.TTC_MAX   : default value for exception

if abs(rel_pos_y) <= TTC_PARAM.ROI.Y_MAX
    TTC_out = - rel_pos_x/rel_vel_x;
    
else
    TTC_out = TTC_PARAM.TTC_MAX;
end
if TTC_out<0
    TTC_out = TTC_PARAM.TTC_MAX;
end

end