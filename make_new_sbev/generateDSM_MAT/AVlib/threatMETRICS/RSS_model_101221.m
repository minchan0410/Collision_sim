function [RSS_x, RSS_y] = RSS_model_101221(Rel_x, Rel_y, vxA, vxB, vyA, vyB, classB, RSS_Param, SHAPE, EGO_VEHICLE)
% Responsibility-Sensitive Safety(RSS) model
%
% RSS_model - Calulate the safe longitudinal/lateral distance.
% [RSS_x, RSS_y] = RSS_model(Rel_x,Rel_y,vxA,vxB,vyA,vyB,RSS_Param)
%
% RSS_model Input
%
% Rel_x : Relative longitudinal position (m)
% Rel_y : Relative lateral position (m)
% vxA : Longitudinal velocity of ego vehicle (m/s)
% vxB : Longitudinal velocity of target (m/s)
% vyA : Lateral velocity of ego vehicle (m/s)
% vyB : Lateral velocity of target (m/s)
% classB : shape(classification) of target
% RSS_Param {struct} : Parameters of RSS_model
%                      AMAX_XA  : Maximum long acc of ego veh.
%                      AMIN_XA  : Minimum long decc of ego veh.
%                      AMAX_XB  : Maximum long acc of target.
%                      RHO_X    : Response time of ego veh.
%                      AMAX_YA  : Maximum lat acc of ego veh.
%                      AMIN_YA  : Minimum lat decc of ego veh.
%                      AMAX_YB  : Maximum lat acc of target.
%                      AMIN_YB  : Minimum lat decc of target.
%                      RHO_YA   : Response time of ego veh.
%                      RHO_YB   : Response time of target.
%                      F        : Fluctutation margin.
% SHAPE {struct} : Parameters of shape in sensor fusion
%                      OBSTACLE
%                      ROAD_BARRIER
%                      VEHICLE_CANDIDATE
%                      VEHICLE_CONFIRMED
%                      PEDESTRIAN_CANDIDATE
%                      PEDESTRIAN_CONFIRMED
%                      MOTOR_BIKE_CANDIDATE
%                      MOTOR_BIKE_CONFIRMED
%                      BICYCLE_CANDIDATE
%                      BICYCLE_CONFIRMED
%                      E_SCOOTER_CANDIDATE
%                      E_SCOOTER_CONFIRMED
% EGO_VEHICLE {struct} : Parameters of shape in sensor fusion
%                      EGO_WIDTH : width of ego veh
%                      EGO_LENGTH : length of ego veh
%
%
% RSS_model output
% RSS_x : Safe longitudinal distance between ego veh and target veh.
% RSS_y : Safe lateral distance between ego veh and target veh.

% Reference : Shai Shalev-Shwartz et al, ¡°On a Formal Model of Safe and Scalable Self-driving Cars,¡±
% https://arxiv.org/abs/1708.06374, Mobileye, 2017.
%%%%%%%%%%%%%%%%%%%%%% Parameter %%%%%%%%%%%%%%%%%%%%%%%%
if classB == SHAPE.VEHICLE_CONFIRMED
    AMAX_XA  =  RSS_Param.VEHICLE.AMAX_XA;
    AMIN_XA  =  RSS_Param.VEHICLE.AMIN_XA;
    AMAX_XB  =  RSS_Param.VEHICLE.AMAX_XB;
    RHO_X    =  RSS_Param.VEHICLE.RHO_X;
    AMAX_YA  =  RSS_Param.VEHICLE.AMAX_YA;
    AMIN_YA  =  RSS_Param.VEHICLE.AMIN_YA;
    AMAX_YB  =  RSS_Param.VEHICLE.AMAX_YB;
    AMIN_YB  =  RSS_Param.VEHICLE.AMIN_YB;
    RHO_YA   =  RSS_Param.VEHICLE.RHO_YA;
    RHO_YB   =  RSS_Param.VEHICLE.RHO_YB;
    F        =  RSS_Param.VEHICLE.F;
    
elseif classB == SHAPE.PEDESTRIAN_CONFIRMED
    AMAX_XA  =  RSS_Param.PEDESTRIAN.AMAX_XA;
    AMIN_XA  =  RSS_Param.PEDESTRIAN.AMIN_XA;
    AMAX_XB  =  RSS_Param.PEDESTRIAN.AMAX_XB;
    RHO_X    =  RSS_Param.PEDESTRIAN.RHO_X;
    AMAX_YA  =  RSS_Param.PEDESTRIAN.AMAX_YA;
    AMIN_YA  =  RSS_Param.PEDESTRIAN.AMIN_YA;
    AMAX_YB  =  RSS_Param.PEDESTRIAN.AMAX_YB;
    AMIN_YB  =  RSS_Param.PEDESTRIAN.AMIN_YB;
    RHO_YA   =  RSS_Param.PEDESTRIAN.RHO_YA;
    RHO_YB   =  RSS_Param.PEDESTRIAN.RHO_YB;
    F        =  RSS_Param.PEDESTRIAN.F;
    
elseif classB == SHAPE.BICYCLE_CONFIRMED
    AMAX_XA  =  RSS_Param.CYCLIST.AMAX_XA;
    AMIN_XA  =  RSS_Param.CYCLIST.AMIN_XA;
    AMAX_XB  =  RSS_Param.CYCLIST.AMAX_XB;
    RHO_X    =  RSS_Param.CYCLIST.RHO_X;
    AMAX_YA  =  RSS_Param.CYCLIST.AMAX_YA;
    AMIN_YA  =  RSS_Param.CYCLIST.AMIN_YA;
    AMAX_YB  =  RSS_Param.CYCLIST.AMAX_YB;
    AMIN_YB  =  RSS_Param.CYCLIST.AMIN_YB;
    RHO_YA   =  RSS_Param.CYCLIST.RHO_YA;
    RHO_YB   =  RSS_Param.CYCLIST.RHO_YB;
    F        =  RSS_Param.CYCLIST.F;
    
elseif classB == SHAPE.E_SCOOTER_CONFIRMED
    AMAX_XA  =  RSS_Param.E_SCOOTER.AMAX_XA;
    AMIN_XA  =  RSS_Param.E_SCOOTER.AMIN_XA;
    AMAX_XB  =  RSS_Param.E_SCOOTER.AMAX_XB;
    RHO_X    =  RSS_Param.E_SCOOTER.RHO_X;
    AMAX_YA  =  RSS_Param.E_SCOOTER.AMAX_YA;
    AMIN_YA  =  RSS_Param.E_SCOOTER.AMIN_YA;
    AMAX_YB  =  RSS_Param.E_SCOOTER.AMAX_YB;
    AMIN_YB  =  RSS_Param.E_SCOOTER.AMIN_YB;
    RHO_YA    =  RSS_Param.E_SCOOTER.RHO_YA;
    RHO_YB    =  RSS_Param.E_SCOOTER.RHO_YB;
    F        =  RSS_Param.E_SCOOTER.F;
end

W  =  EGO_VEHICLE.EGO_WIDTH;


%%%%%%%%%%%%%%%%%%%%%% long model %%%%%%%%%%%%%%%%%%%%%%%%
d0 = Rel_x;
% RSS_x = d0 - (vxB.^2./(2*AMAX_XB)) + (vxA.*RHO_X + 0.5*AMAX_XA.*RHO_X.^2 + ((vxA + RHO_X*AMAX_XA).^2./(2*AMIN_XA)));

if vxB >= 0
    RSS_x = d0 + (vxB.^2./(2*AMAX_XB)) - (vxA.*RHO_X + 0.5*AMAX_XA.*RHO_X.^2 + ((vxA + RHO_X*AMAX_XA).^2./(2*AMIN_XA)));
else
    RSS_x = d0 - (vxB.^2./(2*AMAX_XB)) - (vxA.*RHO_X + 0.5*AMAX_XA.*RHO_X.^2 + ((vxA + RHO_X*AMAX_XA).^2./(2*AMIN_XA)));
end
if RSS_x < -10
    RSS_x = -10;
end

if RSS_x > 20
    RSS_x = 20;
end
%%%%%%%%%%%%%%%%%%%%%% lat model %%%%%%%%%%%%%%%%%%%%%%%%%
% if Rel_y>=0
%     vyB=-vyB;
% end
d0_y = Rel_y - (W);
if Rel_y < 0
 d0_y = -Rel_y - (W);   
end
if d0_y < 0
    d0_y = 0;
end

if Rel_y >= 0
    term1 = RHO_YA.*((vyA + (vyA + RHO_YA*AMAX_YA))./2);
    term2 = (vyA + RHO_YA*AMAX_YA).^2./(2*AMIN_YA);
    term3 = RHO_YB.*((vyB + (vyB - RHO_YB*AMAX_YB))./2);
    term4 = (vyB - RHO_YB*AMIN_YB).^2./(2*AMIN_YB);
    
    RSS_y = d0_y - (F + (term1 + term2)) + (term3 - term4);
else
    term1 = RHO_YA.*((vyA + (vyA - RHO_YA*AMAX_YA))./2);
    term2 = (vyA - RHO_YA*AMAX_YA).^2./(2*AMIN_YA);
    term3 = RHO_YB.*((vyB + (vyB + RHO_YB*AMAX_YB))./2);
    term4 = (vyB + RHO_YB*AMIN_YB).^2./(2*AMIN_YB);

    RSS_y = d0_y - (F - (term1 - term2)) - (term3 + term4);
end




if RSS_y < -10
    RSS_y = -10;
end

if RSS_y > 10
    RSS_y = 10;
end