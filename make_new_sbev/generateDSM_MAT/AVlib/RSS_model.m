function [RSS_x, RSS_y] = RSS_model(Rel_x,Rel_y,vxA,vxB,vyA,vyB,RSS_Param)
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
% vxB : Longitudinal velocity of target vehicle (m/s)
% vyA : Lateral velocity of ego vehicle (m/s)
% vyB : Lateral velocity of target vehicle (m/s)
%
% RSS_Param {struct} : Parameters of RSS_model
%                      AMAX_XA  : Maximum long acc of ego veh.
%                      AMIN_XA  : Minimum long decc of ego veh.
%                      AMAX_XB  : Maximum long acc of target veh.
%                      RHO_X    : Response time.
%                      AMAX_Y   : Maximum lat acc of veh.
%                      AMIN_Y   : Minimum lat decc of veh.
%                      F        : Fluctutation margin.
%                      RHO_Y    : Response time.
%                      L_F      : Distance of the front tire from the c.g. of the veh.
%                      W        : Width of ego veh.
%
% RSS_model output
% RSS_x : Safe longitudinal distance between ego veh and target veh.
% RSS_y : Safe lateral distance between ego veh and target veh.

% Reference : Shai Shalev-Shwartz et al, ¡°On a Formal Model of Safe and Scalable Self-driving Cars,¡±
% https://arxiv.org/abs/1708.06374, Mobileye, 2017.
%%%%%%%%%%%%%%%%%%%%%% Parameter %%%%%%%%%%%%%%%%%%%%%%%%
AMAX_XA  =  RSS_Param(1).AMAX_XA;
AMIN_XA  =  RSS_Param(1).AMIN_XA;
AMAX_XB  =  RSS_Param(1).AMAX_XB;
RHO_X    =  RSS_Param(1).RHO_X;
AMAX_Y   =  RSS_Param(1).AMAX_Y;
AMIN_Y   =  RSS_Param(1).AMIN_Y;
RHO_Y    =  RSS_Param(1).RHO_Y;
L_F      =  RSS_Param(1).L_F;
W        =  RSS_Param(1).W;
F        =  RSS_Param(1).F;

%%%%%%%%%%%%%%%%%%%%%% long model %%%%%%%%%%%%%%%%%%%%%%%%
d0=Rel_x;
RSS_x=d0-(vxB.^2./(2*AMAX_XB))+(vxA.*RHO_X+0.5*AMAX_XA.*RHO_X.^2+((vxA+RHO_X*AMAX_XA).^2./(2*AMIN_XA)));
if vxB<-0.1
    RSS_x=d0+(vxB.^2./(2*AMAX_XB))+(vxA.*RHO_X+0.5*AMAX_XA.*RHO_X.^2+((vxA+RHO_X*AMAX_XA).^2./(2*AMIN_XA)));
end
if RSS_x<-10
    RSS_x=-10;
end

if RSS_x > 20
    RSS_x=20;
end
%%%%%%%%%%%%%%%%%%%%%% lat model %%%%%%%%%%%%%%%%%%%%%%%%%
% if Rel_y>=0
%     vyB=-vyB;
% end
d0_y=Rel_y-(W);
if Rel_y<0
 d0_y=-Rel_y-(W);   
end
if d0_y<0
    d0_y=0;
end
term1=RHO_Y.*((vyA+vyA+RHO_Y*AMAX_Y)./2);
term2=(vyA+vyA+RHO_Y*AMAX_Y).^2./(2*AMIN_Y);
term3=RHO_Y.*((vyB+vyB-RHO_Y*AMAX_Y)./2);
term4=(-2*vyB-RHO_Y*AMAX_Y).^2./(2*AMIN_Y);
RSS_y=d0_y-(F+term1+term2)+(term3-term4);


if RSS_y < -10
    RSS_y= -10;
end

if RSS_y > 10
    RSS_y = 10;
end