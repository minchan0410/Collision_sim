% Calibration by JMLee

ON=1;
OFF=0;
%%
TTC_PARAM.ROI.Y_MIN = -2;
TTC_PARAM.ROI.Y_MAX = 2;
TTC_PARAM.TTC_MAX = 11;

TTC_INVERSE_PARAM.TTC_INVERSE_MAX = 100;
TLC_INVERSE_PARAM.TLC_INVERSE_MAX = 100;

TLC_PARAM.TLC_MAX = 11;
TLC_PARAM.TLC_Y_MIN = 2;

I_LAT_PARAM.TLC_THRESHOLD = 0.5;
I_LAT_PARAM.T_THINKING = 0.1;
I_LAT_PARAM.T_BRAKE = 0.1;
I_LAT_PARAM.A_X_MAX = -10;
% I_LAT_PARAM.A_X_MAX = -7;
I_LAT_PARAM.X_MAX = 4.5;
I_LAT_PARAM.X_THRESHOLD = 6;
I_LAT_PARAM.TTC_INVERSE_THRESHOLD = 2;
I_LAT_PARAM.P_LONG_DEFAULT_VALUE = 300;
I_LAT_PARAM.TTC_MAX = 11;
I_LAT_PARAM.TLC_MAX = 11;


RSS_Param.VEHICLE.AMAX_XA = 3;    % 자차 최대가속도
RSS_Param.VEHICLE.AMIN_XA = 10;   % 자차 최소감속도
RSS_Param.VEHICLE.AMAX_XB = 10;   % 상대차 최대감속도
RSS_Param.VEHICLE.RHO_X   = 1;    % response time of ego veh for braking
RSS_Param.VEHICLE.AMAX_YA = 4;    % 자차 횡방향 최대가속도
RSS_Param.VEHICLE.AMIN_YA = 4;    % 자차 횡방향 최소감속도
RSS_Param.VEHICLE.AMAX_YB = 4;    % 상대차 횡방향 최대가속도
RSS_Param.VEHICLE.AMIN_YB = 4;    % 상대차 횡방향 최소감속도
RSS_Param.VEHICLE.RHO_YA = 1;     % response time of ego veh
RSS_Param.VEHICLE.RHO_YB = 1;     % response time of target
RSS_Param.VEHICLE.F = 0.1;        % 횡방향 마진 (fluctuation margin)



RSS_Param.PEDESTRIAN.AMAX_XA = 3;    % 자차 최대가속도
RSS_Param.PEDESTRIAN.AMIN_XA = 10;   % 자차 최소감속도
RSS_Param.PEDESTRIAN.AMAX_XB = 2;    % 상대 보행자 최대감속도
RSS_Param.PEDESTRIAN.RHO_X = 1;     % response time of ego veh for braking
RSS_Param.PEDESTRIAN.AMAX_YA = 4;    % 자차 횡방향 최대가속도
RSS_Param.PEDESTRIAN.AMIN_YA = 4;    % 자차 횡방향 최소감속도
RSS_Param.PEDESTRIAN.AMAX_YB = 2;    % 상대 횡방향 최대가속도
RSS_Param.PEDESTRIAN.AMIN_YB = 2;    % 상대 횡방향 최소감속도
RSS_Param.PEDESTRIAN.RHO_YA = 1;     % response time of ego veh
RSS_Param.PEDESTRIAN.RHO_YB = 0.5;   % response time of target
RSS_Param.PEDESTRIAN.F = 0.1;        % 횡방향 마진 (fluctuation margin)




RSS_Param.CYCLIST.AMAX_XA = 3;    % 자차 최대가속도
RSS_Param.CYCLIST.AMIN_XA = 10;   % 자차 최소감속도
RSS_Param.CYCLIST.AMAX_XB = 3;    % 상대 자전거 최대감속도
RSS_Param.CYCLIST.RHO_X = 1;      % response time of ego veh for braking
RSS_Param.CYCLIST.AMAX_YA = 0;    % 자차 횡방향 최대가속도
RSS_Param.CYCLIST.AMIN_YA = 4;    % 자차 횡방향 최소감속도
RSS_Param.CYCLIST.AMAX_YB = 1;    % 상대 횡방향 최대가속도
RSS_Param.CYCLIST.AMIN_YB = 1;    % 상대 횡방향 최소감속도
RSS_Param.CYCLIST.RHO_YA = 1;     % response time of ego veh
RSS_Param.CYCLIST.RHO_YB = 0.4;   % response time of target
RSS_Param.CYCLIST.F = 0.1;        % 횡방향 마진 (fluctuation margin)




RSS_Param.E_SCOOTER.AMAX_XA = 3;    % 자차 최대가속도
RSS_Param.E_SCOOTER.AMIN_XA = 10;   % 자차 최소감속도
RSS_Param.E_SCOOTER.AMAX_XB = 2.5;  % 상대 e scooter 최대감속도
RSS_Param.E_SCOOTER.RHO_X = 1;      % response time of ego veh for braking
RSS_Param.E_SCOOTER.AMAX_YA = 0;    % 자차 횡방향 최대가속도
RSS_Param.E_SCOOTER.AMIN_YA = 4;    % 자차 횡방향 최소감속도
RSS_Param.E_SCOOTER.AMAX_YB = 1;    % 상대 횡방향 최대가속도
RSS_Param.E_SCOOTER.AMIN_YB = 1;    % 상대 횡방향 최소감속도
RSS_Param.E_SCOOTER.RHO_YA = 1;     % response time of ego veh
RSS_Param.E_SCOOTER.RHO_YB = 0.7;   % response time of target
RSS_Param.E_SCOOTER.F = 0.1;        % 횡방향 마진 (fluctuation margin)



HONDA_PARAM.ROI.Y_MIN = -2;
HONDA_PARAM.ROI.Y_MAX = 2;
HONDA_PARAM.ROI.X_MIN = -1;
HONDA_PARAM.ROI.X_MAX = 120;
HONDA_PARAM.HONDA_MIN = 1;

% veh_CG_to_front=2.1;
% BWD_POS_Y_CORRECT=0.9;