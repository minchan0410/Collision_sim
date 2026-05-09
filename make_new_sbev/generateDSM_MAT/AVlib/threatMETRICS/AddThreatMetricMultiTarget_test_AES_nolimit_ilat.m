orig_state = warning('off','MATLAB:sqrtm:SingularMatrix');
% orig_state = warning('off','sqrtm:SingularMatrix');
%% Parameter

IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE                     = 1;  %      [rad]              Global heading angle
IN_VEHICLE_SENSOR.MEASURE.GLO_POS_X                             = 2;  %      [m]                Global longitudinal position
IN_VEHICLE_SENSOR.MEASURE.GLO_POS_Y                             = 3;  %      [m]                Global lateral position
IN_VEHICLE_SENSOR.MEASURE.VEHICLE_SPEED                         = 4;  %      [m/s]              absolute velocity
IN_VEHICLE_SENSOR.MEASURE.LONG_ACC                              = 5;  %      [m/s^2]
IN_VEHICLE_SENSOR.MEASURE.LONG_VEL                              = 6;  %     [m/s]
IN_VEHICLE_SENSOR.MEASURE.LAT_VEL                               = 7;  %     [m/s]

IN_VEHICLE_SENSOR.STATE_NUMBER                                  = length(fieldnames(IN_VEHICLE_SENSOR.MEASURE));


% CLASS B
CLASS_B.MEASURE.GLO_HEADING_ANGLE                           = 1;  %      [rad]                                            global 좌표계에서의 heading angle
CLASS_B.MEASURE.GLO_POS_Y                                   = 2;  %      [m]                                             position = 뒷범퍼 중심
CLASS_B.MEASURE.GLO_POS_X                                   = 3;  %      [m]
CLASS_B.MEASURE.GLO_VEL_Y                                   = 4;  %      [m/s]
CLASS_B.MEASURE.GLO_VEL_X                                   = 5;  %      [m/s]
CLASS_B.MEASURE.WIDTH                                       = 6;  %      [m]
CLASS_B.MEASURE.LENGTH                                      = 7;  %      [m]
CLASS_B.MEASURE.CLASSIFICATION                              = 8;

CLASS_B.PREPROCESSING.REL_POS_Y                             = 8;  %     [m]
CLASS_B.PREPROCESSING.REL_POS_X                             = 9;  %     [m]
CLASS_B.PREPROCESSING.REL_VEL_Y                             = 10;  %     [m/s]
CLASS_B.PREPROCESSING.REL_VEL_X                             = 11;  %     [m/s]
CLASS_B.PREPROCESSING.HEADING_ANGLE                         = 12;  %     [rad]

CLASS_B.MEASURE.STATE_NUMBER                                = length(fieldnames(CLASS_B.MEASURE)); %                      Class_B 에서 출력되는 최대 state 개수
CLASS_B.PREPROCESSING.STATE_NUMBER                          = length(fieldnames(CLASS_B.PREPROCESSING)); %                       Preprocessing 에서 추가될 state 개수
CLASS_B.STATE_NUMBER                                        = CLASS_B.MEASURE.STATE_NUMBER + CLASS_B.PREPROCESSING.STATE_NUMBER;
CLASS_B.TRACK_NUMBER                                        = 8;

% Description
CLASS_B.DESCRIPTION_CLASSIFICATION.UNDECIDED               = 0;
CLASS_B.DESCRIPTION_CLASSIFICATION.CAR                     = 1;
CLASS_B.DESCRIPTION_CLASSIFICATION.PEDESTRIAN              = 2;
CLASS_B.DESCRIPTION_CLASSIFICATION.BICYCLE                 = 3;
CLASS_B.DESCRIPTION_CLASSIFICATION.MOTOR_BIKE              = 4;

% Road
ROAD.MEASURE.TOTAL_LANE_NUMBER                        = 1;
ROAD.MEASURE.WIDTH                                    = 2;   %     [m]
ROAD.MEASURE.SHOULDER_EXIST                           = 3;
ROAD.MEASURE.SHOULDER_WIDTH                           = 4;
ROAD.MEASURE.CURVATURE                                = 5;   %     [1/m]
ROAD.MEASURE.ROAD_SLOPE                               = 6;   %     [rad]
ROAD.MEASURE.DISTANCE_TO_LEFTLANE                     = 7;   %     [m]
ROAD.MEASURE.DISTANCE_TO_RIGHTLANE                    = 8;   %     [m]

% ROAD.PREPROCESSING.STATE_NUMBER                    = length(fieldnames(ROAD.PREPROCESSING));
ROAD.PREPROCESSING.STATE_NUMBER                    = 0;

ROAD.MEASURE.STATE_NUMBER                          = length(fieldnames(ROAD.MEASURE));
ROAD.STATE_NUMBER                                  = ROAD.MEASURE.STATE_NUMBER + ROAD.PREPROCESSING.STATE_NUMBER;

% Line
LINE.LEFT                                             = 1;
LINE.RIGHT                                            = 2;

LINE.MEASURE.CURVATURE_RATE                           = 1;   %     [1/m^2]
LINE.MEASURE.CURVATURE                                = 2;   %     [1/m]
LINE.MEASURE.ROAD_SLOPE                               = 3;   %     [rad]
LINE.MEASURE.DISTANCE_TO_LINE                         = 4;   %     [m]
LINE.MEASURE.LINE_NUMBER                              = 5;

LINE.MEASURE.STATE_NUMBER                             = length(fieldnames(LINE.MEASURE));

LINE.PREPROCESSING.CURVATURE_RATE                     = 6;   %     [1/m^2]
LINE.PREPROCESSING.CURVATURE                          = 7;   %     [1/m]

LINE.PREPROCESSING.STATE_NUMBER                       = length(fieldnames(LINE.PREPROCESSING));

LINE.STATE_NUMBER                                  = LINE.MEASURE.STATE_NUMBER + LINE.PREPROCESSING.STATE_NUMBER;

%% Initialization
sim_time = data.Time.data;

STATE_LENGTH                                    = length(fieldnames(TRAINING));
Training_data                                   = zeros(length(sim_time), STATE_LENGTH, CLASS_B.TRACK_NUMBER);
I_lat_out                                       = zeros(length(sim_time),1);
I_lat_out2                                      = zeros(length(sim_time),1);
I_long_out                                      = zeros(length(sim_time),1);
I_long_out2                                      = zeros(length(sim_time),1);

DLC_out                                         = zeros(length(sim_time),1);
TTC_out                                         = zeros(length(sim_time),1);
TLC_out                                         = zeros(length(sim_time),1);
TTC_inverse_out                                 = zeros(length(sim_time),1);
TLC_inverse_out                                 = zeros(length(sim_time),1);
HONDA_w                                         = zeros(length(sim_time),1);
HONDA_br                                        = zeros(length(sim_time),1);
THM                                             = zeros(length(sim_time),1);
RSS_x                                           = zeros(length(sim_time),1);
RSS_y                                           = zeros(length(sim_time),1);

In_Vehicle_Sensor = zeros(IN_VEHICLE_SENSOR.STATE_NUMBER, 1, length(sim_time));
Class_B = zeros(CLASS_B.MEASURE.STATE_NUMBER, CLASS_B.TRACK_NUMBER, length(sim_time));

%% Threat Metric Parameters
run('.\Threat_Parameter')

%% Vehicle Parameters
Search_Width = char(regexp(Vehicle_File,'[^\n]*CarGen.Vehicle.Width =[^\n]*','match'));
EGO_WIDTH = str2double(Search_Width(strfind(Search_Width,'=')+2:end-1))*1/1000;

Search_Length = char(regexp(Vehicle_File,'[^\n]*CarGen.Vehicle.Length =[^\n]*','match'));
EGO_LENGTH = str2double(Search_Length(strfind(Search_Length,'=')+2:end-1))*1/1000;

Search_Ego_CG2Rear_Bumper = strtrim(char(regexp(Vehicle_File,'[^\n]*Body.pos =[^\n]*','match')));
eval(['tmp_Ego_CG2Rear_Bumper = [' Search_Ego_CG2Rear_Bumper(strfind(Search_Ego_CG2Rear_Bumper,'=')+2:end) '];']);
EGO_CG2_REAR_BUMPER = tmp_Ego_CG2Rear_Bumper(1,1);
EGO_CG2_FRONT_BUMPER = EGO_LENGTH - EGO_CG2_REAR_BUMPER;

EGO_VEHICLE.EGO_WIDTH = EGO_WIDTH;
EGO_VEHICLE.EGO_LENGTH = EGO_LENGTH;

%% Traffic and Sensor Parameters

Search_Traffic_Num = strtrim(char(regexp(Scenario_File,'[^\n]*Traffic.N =[^\n]*','match')));
Traffic_Num = str2double(Search_Traffic_Num(strfind(Search_Traffic_Num,'=')+2:end));

Search_Sensor_Num = strtrim(char(regexp(Vehicle_File,'[^\n]*Sensor.Object.N =[^\n]*','match')));
Sensor_Num = str2double(Search_Sensor_Num(strfind(Search_Sensor_Num,'=')+2:end));

if Sensor_Num == 0
    disp('Carmaker 시나리오 파일에서 Object 센서가 없습니다.');
else
    Sensor_Name_Cell = cell(1,Sensor_Num);
    
    for i = 1:Sensor_Num
        Search_Sensor_name_char = strtrim(char(regexp(Vehicle_File,['[^\n]*Sensor.Object.' num2str(i-1) '.name =[^\n]*'],'match')));
        tmp_Sensor_name_char = Search_Sensor_name_char(strfind(Search_Sensor_name_char,'=')+2:end);
        
        Sensor_Name_Cell(1,i) = cellstr(tmp_Sensor_name_char);
        
    end
    
    Traffic_Name_Cell = cell(1,Traffic_Num);
    
    for i = 1:Traffic_Num
        
        Search_Traffic_name_char = strtrim(char(regexp(Scenario_File,['[^\n]*Traffic.' num2str(i-1) '.Name =[^\n]*'],'match')));
        tmp_Traffic_name_char = Search_Traffic_name_char(strfind(Search_Traffic_name_char,'=')+2:end);
        
        Traffic_Name_Cell(1,i) = cellstr(tmp_Traffic_name_char);
        
    end
end

if length(Traffic_Name_Cell(1,:)) == 1
    tmp_Traffic_Name = char(Traffic_Name_Cell(1,1));
    
    Search_Traffic_Dimension = strtrim(char(regexp(Scenario_File,['[^\n]*Traffic.0.Basics.Dimension =[^\n]*'],'match')));
    eval(['tmp_Traffic_Dimension = [' Search_Traffic_Dimension(strfind(Search_Traffic_Dimension,'=')+2:end) '];']);
    
    Search_Traffic_CG2Rear_Bumper = strtrim(char(regexp(Scenario_File,['[^\n]*Traffic.0.Basics.Fr12CoM =[^\n]*'],'match')));
    eval(['TARGET_CG2_REAR_BUMPER = ' Search_Traffic_CG2Rear_Bumper(strfind(Search_Traffic_CG2Rear_Bumper,'=')+2:end) ';']);
    
    tmp_Traffic_Length = tmp_Traffic_Dimension(1,1);
    TARGET_WIDTH = tmp_Traffic_Dimension(1,2);
    TARGET_CG2_FRONT_BUMPER = tmp_Traffic_Length - TARGET_CG2_REAR_BUMPER;
    TARGET_TRAFFIC_Name = tmp_Traffic_Name;
    TARGET_LENGTH = TARGET_CG2_FRONT_BUMPER + TARGET_CG2_REAR_BUMPER;
    
    Class_B(CLASS_B.MEASURE.WIDTH,1, :) = TARGET_WIDTH;
    Class_B(CLASS_B.MEASURE.LENGTH,1, :) = TARGET_LENGTH;
    
    % 추후 환경 파일 이용해서 shape 정보 추가하기
    Class_B(CLASS_B.MEASURE.CLASSIFICATION,1, :) = CLASS_B.DESCRIPTION_CLASSIFICATION.CAR;
    
else
    for SIG_Num = 1:length(Traffic_Name_Cell(1,:))
        tmp_Traffic_Name = char(Traffic_Name_Cell(1,SIG_Num));
        
        Search_Traffic_Dimension = strtrim(char(regexp(Scenario_File,['[^\n]*Traffic.' num2str(SIG_Num-1) '.Basics.Dimension =[^\n]*'],'match')));
        eval(['tmp_Traffic_Dimension = [' Search_Traffic_Dimension(strfind(Search_Traffic_Dimension,'=')+2:end) '];']);
        
        Search_Traffic_CG2Rear_Bumper = strtrim(char(regexp(Scenario_File,['[^\n]*Traffic.' num2str(SIG_Num-1) '.Basics.Fr12CoM =[^\n]*'],'match')));
        eval(['TARGET_CG2_REAR_BUMPER = ' Search_Traffic_CG2Rear_Bumper(strfind(Search_Traffic_CG2Rear_Bumper,'=')+2:end) ';']);
        
        tmp_Traffic_Length = tmp_Traffic_Dimension(1,1);
        TARGET_WIDTH = tmp_Traffic_Dimension(1,2);
        TARGET_CG2_FRONT_BUMPER = tmp_Traffic_Length - TARGET_CG2_REAR_BUMPER;
        TARGET_TRAFFIC_Name = tmp_Traffic_Name;
        TARGET_LENGTH = TARGET_CG2_FRONT_BUMPER + TARGET_CG2_REAR_BUMPER;
        
        eval(['Class_B(CLASS_B.MEASURE.WIDTH,' num2str(SIG_Num) ', :) = TARGET_WIDTH;']);
        eval(['Class_B(CLASS_B.MEASURE.LENGTH,' num2str(SIG_Num) ', :) = TARGET_LENGTH;']);
        eval(['Class_B(CLASS_B.MEASURE.CLASSIFICATION,' num2str(SIG_Num) ', :) = CLASS_B.DESCRIPTION_CLASSIFICATION.CAR;']);
    end
end


%% Preprocessing - Coordinate Transform

Traffic_Number = Traffic_Num;
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE,:) = data.Car_Yaw.data';
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_POS_Y,:) = data.Car_ty.data';  % Fr0(global)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_POS_X,:) = data.Car_tx.data';  % Fr0(global)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.VEHICLE_SPEED,:) = data.Car_v.data'; % wheel velocity
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_ACC,:) = data.Car_ax.data'; % Fr1(body fixed)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,:) = data.Car_vx.data'; % Fr1(body fixed)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LAT_VEL,:) = data.Car_vy.data'; % Fr1(body fixed)

% if Traffic_Number > 2
%     Traffic_Number = 2;
% end

if strcmp(Cur_Scenario_Selection, 'LK_PCSL_ST')||strcmp(Cur_Scenario_Selection, 'LK_PCSL_STP_ST')||...
        strcmp(Cur_Scenario_Selection, 'LK_PCSR_ST')||strcmp(Cur_Scenario_Selection, 'LK_PCSR_STP_ST')||...
        strcmp(Cur_Scenario_Selection, 'LK_POCL_ST')||strcmp(Cur_Scenario_Selection, 'LK_POCR_ST')||...
        strcmp(Cur_Scenario_Selection, 'LK_PSTP_ST')||strcmp(Cur_Scenario_Selection, 'LK_PWAL_ST')||strcmp(Cur_Scenario_Selection, 'LK_PWAR_ST')
        for SIG_Num = 1:Traffic_Number
        VAR_Num = SIG_Num;        
        if VAR_Num == 1
            eval(['Class_B(CLASS_B.MEASURE.GLO_POS_Y,' num2str(SIG_Num) ', :) = data.Traffic_P00_ty.data;']); % Fr0 (global)
            eval(['Class_B(CLASS_B.MEASURE.GLO_POS_X,' num2str(SIG_Num) ', :) = data.Traffic_P00_tx.data;']); % Fr0 (global)
            eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_Y,' num2str(SIG_Num) ', :) = data.Traffic_P00_v_0_y.data;']); % Fr0 (global)
            eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_X,' num2str(SIG_Num) ', :) = data.Traffic_P00_v_0_x.data;']); % Fr0 (global)
%             eval(['Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE,' num2str(SIG_Num) ', :) = data.Traffic_P00_rz.data-pi;']);
            eval(['Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE,' num2str(SIG_Num) ', :) = -1.57;']);

        else
            if SIG_Num < 10
                STR_SIG_Num = ['0' num2str(SIG_Num-1)];
            end            
            eval(['Class_B(CLASS_B.MEASURE.GLO_POS_Y,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_ty.data;']); % Fr0 (global)
            eval(['Class_B(CLASS_B.MEASURE.GLO_POS_X,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_tx.data;']); % Fr0 (global)
            eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_Y,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_v_0_y.data;']); % Fr0 (global)
            eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_X,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_v_0_x.data;']); % Fr0 (global)
%             eval(['Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_rz.data-pi;']);            eval(['Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_rz.data-pi;']);
            eval(['Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE,' num2str(SIG_Num) ', :) = 0;']);

        end
        end
else
    
for SIG_Num = 1:Traffic_Number
    VAR_Num = SIG_Num;
    
    if VAR_Num == 1
        eval(['Class_B(CLASS_B.MEASURE.GLO_POS_Y,' num2str(SIG_Num) ', :) = data.Traffic_RV_ty.data;']); % Fr0 (global)
        eval(['Class_B(CLASS_B.MEASURE.GLO_POS_X,' num2str(SIG_Num) ', :) = data.Traffic_RV_tx.data;']); % Fr0 (global)
        eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_Y,' num2str(SIG_Num) ', :) = data.Traffic_RV_v_0_y.data;']); % Fr0 (global)
        eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_X,' num2str(SIG_Num) ', :) = data.Traffic_RV_v_0_x.data;']); % Fr0 (global)
        eval(['Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE,' num2str(SIG_Num) ', :) = data.Traffic_RV_rz.data;']);
    else
        if SIG_Num < 10
            STR_SIG_Num = ['0' num2str(SIG_Num-1)];
        end
        
        eval(['Class_B(CLASS_B.MEASURE.GLO_POS_Y,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_ty.data;']); % Fr0 (global)
        eval(['Class_B(CLASS_B.MEASURE.GLO_POS_X,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_tx.data;']); % Fr0 (global)
        eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_Y,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_v_0_y.data;']); % Fr0 (global)
        eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_X,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_v_0_x.data;']); % Fr0 (global)
        eval(['Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_rz.data;']);
    end
end
%     for SIG_Num = 1:Traffic_Number
%         VAR_Num = SIG_Num;        
%         if VAR_Num == 1
%             eval(['Class_B(CLASS_B.MEASURE.GLO_POS_Y,' num2str(SIG_Num) ', :) = data.Traffic_C00_ty.data;']); % Fr0 (global)
%             eval(['Class_B(CLASS_B.MEASURE.GLO_POS_X,' num2str(SIG_Num) ', :) = data.Traffic_C00_tx.data;']); % Fr0 (global)
%             eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_Y,' num2str(SIG_Num) ', :) = data.Traffic_C00_v_0_y.data;']); % Fr0 (global)
%             eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_X,' num2str(SIG_Num) ', :) = data.Traffic_C00_v_0_x.data;']); % Fr0 (global)
%             eval(['Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE,' num2str(SIG_Num) ', :) = data.Traffic_C00_rz.data;']);
%         else
%             if SIG_Num < 10
%                 STR_SIG_Num = ['0' num2str(SIG_Num-1)];
%             end            
%             eval(['Class_B(CLASS_B.MEASURE.GLO_POS_Y,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_ty.data;']); % Fr0 (global)
%             eval(['Class_B(CLASS_B.MEASURE.GLO_POS_X,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_tx.data;']); % Fr0 (global)
%             eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_Y,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_v_0_y.data;']); % Fr0 (global)
%             eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_X,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_v_0_x.data;']); % Fr0 (global)
%             eval(['Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE,' num2str(SIG_Num) ', :) = data.Traffic_T' STR_SIG_Num '_rz.data;']);
%         end
%     end
end

for track_number = 1:Traffic_Number
    
    Class_B(CLASS_B.PREPROCESSING.HEADING_ANGLE, track_number, :) = (Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE, track_number, :) -In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :));
    
    X_FrontCenter_A = EGO_CG2_FRONT_BUMPER.*cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_POS_X, 1, :);
    Y_FrontCenter_A = EGO_CG2_FRONT_BUMPER.*sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_POS_Y, 1, :);
    
    X_AB = Class_B(CLASS_B.MEASURE.GLO_POS_X, track_number, :) - X_FrontCenter_A;
    Y_AB = Class_B(CLASS_B.MEASURE.GLO_POS_Y, track_number, :) - Y_FrontCenter_A;
    
    x_AB = X_AB .* cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + Y_AB .* sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :));
    y_AB = -X_AB .* sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + Y_AB .* cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :));
    
    Class_B(CLASS_B.PREPROCESSING.REL_POS_Y, track_number, :) = y_AB;
    Class_B(CLASS_B.PREPROCESSING.REL_POS_X, track_number, :) = x_AB;
    
    Class_B(CLASS_B.PREPROCESSING.REL_VEL_X, track_number, :) = Class_B(CLASS_B.MEASURE.GLO_VEL_X, track_number, :) .* cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + Class_B(CLASS_B.MEASURE.GLO_VEL_Y, track_number, :).*sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL, 1, :);
    Class_B(CLASS_B.PREPROCESSING.REL_VEL_Y, track_number, :) = -Class_B(CLASS_B.MEASURE.GLO_VEL_X, track_number, :) .* sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + Class_B(CLASS_B.MEASURE.GLO_VEL_Y, track_number, :).*cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LAT_VEL, 1, :);
    
end


%%  Road
Search_Ego_Route                                                                = strtrim(char(regexp(Scenario_File,'[^\n]*Road.VhclRoute =[^\n]*','match')));
Ego_Route_number                                                                = str2double(Search_Ego_Route(end));

Search_Ego_RouteID                                                              = strtrim(char(regexp(Scenario_File,'[^\n]*Road.RouteId =[^\n]*','match')));
Ego_Route_ID                                                                    = str2double(Search_Ego_RouteID(end));

Search_Ego_LanePath                                                             = strtrim(char(regexp(Road_File,['[^\n]*Route.' num2str(Ego_Route_number) '.DrvPath:[^\n][^\t][^\n]*'],'match')));

tmp_expression = '[\n][^.]\d*';
tmp_Num_LanePath = (char(regexp(Search_Ego_LanePath,tmp_expression,'match')));
tmp_Num_LanePath = str2double(char(regexp(Search_Ego_LanePath,tmp_expression,'match')));


Ego_LanePath_ID                                                                 = tmp_Num_LanePath;

Search_total_link                                                               = char(regexp(Road_File,'[^\n]*nLinks =[^\n]*','match'));
Search_total_link_num                                                           = str2double(Search_total_link(strfind(Search_total_link,'=')+2:end-1));

Search_LanePath                                                                 = char(regexp(Road_File,'[^\n]*LanePath[^\n]*','match'));

total_LanePath = 0;

if ~isempty(Search_LanePath)
    for tmp_lanepath_index = 1:length(Search_LanePath(:,1))
        
        tmp_lanepath_char = Search_LanePath(tmp_lanepath_index,:);
        equal_index = strfind(tmp_lanepath_char,'=');
        
        eval(['tmp_LanePath_array = [' strtrim(tmp_lanepath_char(1,equal_index+1:end)) '];']);
        
        if tmp_LanePath_array(1,3) ~= 2 % Path Width Limit
            continue;
        else
            total_LanePath = total_LanePath + 1;
            
            num_expression = '\d* =\d*';
            tmp_Num_LanePath = char(regexp(tmp_lanepath_char,num_expression,'match'));
            tmp_Num_LanePath = str2double(tmp_Num_LanePath(1:end-1));
            
            if tmp_Num_LanePath < 10
                eval(['simParams.' strtrim(tmp_lanepath_char(1,1:equal_index-4)) num2str(total_LanePath) '.array = [' strtrim(tmp_lanepath_char(1,equal_index+1:end)) '];']);
            else
                eval(['simParams.' strtrim(tmp_lanepath_char(1,1:equal_index-5)) num2str(total_LanePath) '.array = [' strtrim(tmp_lanepath_char(1,equal_index+1:end)) '];']);
            end
            
            if tmp_LanePath_array(1,1) == Ego_LanePath_ID
                Ego_Lane_ID = tmp_LanePath_array(1,2);
                
                simParams.EGO_LANE = total_LanePath;
                
                if strcmp(Cur_Scenario_Selection, 'LK_CIR_MER')
                    simParams.EGO_LANE = total_LanePath + 1; % 가운데 주행 못하는 차로 추가(가드레일로 막혀있음)
                elseif strcmp(Cur_Scenario_Selection, 'LK_CIL_CU')
                    simParams.EGO_LANE = 4;
                elseif strcmp(Cur_Scenario_Selection, 'LK_CIR_ST')
                    simParams.EGO_LANE = 5;
                elseif strcmp(Cur_Scenario_Selection, 'LK_CIR_CU')
                    simParams.EGO_LANE = 3;
                elseif strcmp(Cur_Scenario_Selection, 'LK_COR_STP_CU')
                    simParams.EGO_LANE = 3;
                elseif strcmp(Cur_Scenario_Selection, 'LK_OVE_CU')
                    simParams.EGO_LANE = 3; % 디버깅 필요
                elseif strcmp(Cur_Scenario_Selection, 'LK_STP_CU')
                    simParams.EGO_LANE = 3; % 디버깅 필요
                end
            end
        end
    end
end

if Search_total_link_num ~= 0
    for tmp_link_number = 1:Search_total_link_num
        tmp_linkL_char = ['Link.' num2str(tmp_link_number-1) '.LaneSection.0.LaneL'];
        Search_LinkL = char(regexp(Road_File,['[^\n]*' tmp_linkL_char '[^\n]*'],'match'));
        
        LinkL_number = str2double(Search_LinkL(end,length('Link.0.LaneSection.0.LaneL.')+1:length('Link.0.LaneSection.0.LaneL.')+1))+1;
        
        tmp_linkR_char = ['Link.' num2str(tmp_link_number-1) '.LaneSection.0.LaneR'];
        Search_LinkR = char(regexp(Road_File,['[^\n]*' tmp_linkR_char '[^\n]*'],'match'));
        
        LinkR_number = str2double(Search_LinkR(end,length('Link.0.LaneSection.0.LaneR.')+1:length('Link.0.LaneSection.0.LaneR.')+1))+1;
    end
end

no_driving_laneL = 0;
driving_laneL = 0;

if LinkL_number ~= 0
    for tmp_LinkL_index = 1:LinkL_number
        
        tmp_char = ['[^\n]*Link.0.LaneSection.0.LaneL.' num2str(tmp_LinkL_index-1) ' =[^\n]*'];
        Search_LinkL_char = char(regexp(Road_File,tmp_char,'match'));
        
        equal_index = strfind(Search_LinkL_char,'=');
        
        eval(['tmp_LaneL_Width_array = [' strtrim(Search_LinkL_char(1,equal_index+1:end)) '];']);
        if tmp_LaneL_Width_array(1,2) < 3.5
            no_driving_laneL = no_driving_laneL + 1;
            continue;
        else
            
            driving_laneL = driving_laneL + 1;
            LaneType = tmp_LaneL_Width_array(1,4);
            
            if LaneType == 4
                eval(['simParams.LaneL' num2str(tmp_LinkL_index-1) '.TYPE = ''Road_border'';']);
            elseif LaneType == 5
                eval(['simParams.LaneL' num2str(tmp_LinkL_index-1) '.TYPE = ''Road_side'';']);
            elseif LaneType == 0
                eval(['simParams.LaneL' num2str(tmp_LinkL_index-1) '.TYPE = ''Driving_lane'';']);
            end
            
            tmp_id_char = ['[^\n]*Link.0.LaneSection.0.LaneL.' num2str(tmp_LinkL_index-1) '.ID =[^\n]*'];
            Search_LinkL_id_char = char(regexp(Road_File,tmp_id_char,'match'));
            equal_id_index = strfind(Search_LinkL_id_char,'=');
            
            eval(['tmp_LaneL_ID = ' strtrim(Search_LinkL_id_char(1,equal_id_index+1:end)) ';']);
            
            
            eval(['simParams.LaneL' num2str(tmp_LinkL_index-1) '.WIDTH = ' num2str(tmp_LaneL_Width_array(1,2)) ';']);
            eval(['simParams.LaneL' num2str(tmp_LinkL_index-1) '.ID = tmp_LaneL_ID;']);
            
            if tmp_LaneL_ID == Ego_Lane_ID
                Ego_Lane = ['LaneL' num2str(tmp_LinkL_index-1)];
            end
        end
    end
end

no_driving_laneR = 0;
driving_laneR = 0;

if LinkR_number ~= 0
    for tmp_LinkR_index = 1:LinkR_number
        
        tmp_char = ['[^\n]*Link.0.LaneSection.0.LaneR.' num2str(tmp_LinkR_index-1) ' =[^\n]*'];
        Search_LinkR_char = char(regexp(Road_File,tmp_char,'match'));
        
        equal_index = strfind(Search_LinkR_char,'=');
        
        eval(['tmp_LaneR_Width_array = [' strtrim(Search_LinkR_char(1,equal_index+1:end)) '];']);
        if tmp_LaneR_Width_array(1,2) < 3.5
            no_driving_laneR = no_driving_laneR + 1;
            continue;
        else
            driving_laneR = driving_laneR + 1;
            LaneType = tmp_LaneR_Width_array(1,4);
            
            if LaneType == 4
                eval(['simParams.LaneR' num2str(tmp_LinkR_index-1) '.TYPE = ''Road_border'';']);
            elseif LaneType == 5
                eval(['simParams.LaneR' num2str(tmp_LinkR_index-1) '.TYPE = ''Road_side'';']);
            elseif LaneType == 0
                eval(['simParams.LaneR' num2str(tmp_LinkR_index-1) '.TYPE = ''Driving_lane'';']);
            end
            
            tmp_id_char = ['[^\n]*Link.0.LaneSection.0.LaneR.' num2str(tmp_LinkR_index-1) '.ID =[^\n]*'];
            Search_LinkR_id_char = char(regexp(Road_File,tmp_id_char,'match'));
            equal_id_index = strfind(Search_LinkR_id_char,'=');
            
            eval(['tmp_LaneR_ID = ' strtrim(Search_LinkR_id_char(1,equal_id_index+1:end)) ';']);
            
            eval(['simParams.LaneR' num2str(tmp_LinkR_index-1) '.WIDTH = ' num2str(tmp_LaneR_Width_array(1,2)) ';']);
            eval(['simParams.LaneR' num2str(tmp_LinkR_index-1) '.ID = tmp_LaneR_ID;']);
            
            if tmp_LaneR_ID == Ego_Lane_ID
                Ego_Lane = ['LaneR' num2str(tmp_LinkR_index-1)];
            end
        end
    end
end

for i = 1:driving_laneL + driving_laneR
    eval(['simParams.LanePath' num2str(i) ' = [];']);
end

Search_Ref_Line                                                                 = strtrim(char(regexp(Road_File,'[^\n]*Link.0.LaneSection.0.ID =[^\n]*','match')));
Ref_Line_ID = str2double(Search_Ref_Line(end));

Search_RoadMarking                                                              = char(regexp(Road_File,'[^\n]*RL.1.RoadMarking.[^\n]*','match'));
if ~strcmp(Cur_Scenario_Selection, 'LK_OVE_CU') && ~strcmp(Cur_Scenario_Selection, 'LK_OVE_ST') &&  ~strcmp(Cur_Scenario_Selection, 'LK_STP_ST')
    if ~isempty(Search_RoadMarking)
        for tmp_RoadMarking_index = 1:length(Search_RoadMarking(:,1))/3
            
            tmp_RoadMarking_ID_char = ['[^\n]*RL.1.RoadMarking.' num2str(tmp_RoadMarking_index-1) '.ID =[^\n]*'];
            Search_RoadMarking_ID_char = char(regexp(Road_File,tmp_RoadMarking_ID_char,'match'));
            
            equal_index = strfind(Search_RoadMarking_ID_char,'=');
            
            eval(['tmp_RoadMarking_ID_array = [' strtrim(Search_RoadMarking_ID_char(1,equal_index+1:end)) '];']);
            
            tmp_RoadMarking_Lane_ID = tmp_RoadMarking_ID_array(1,2);
            
            for tmp_Line_index = 1:driving_laneR+driving_laneL+1
                if tmp_Line_index <= driving_laneL
                    %                 tmp_L_save = tmp_Lane_index - driving_laneL + driving_laneR - 1;
                    tmp_L = -tmp_Line_index + driving_laneL;
                    
                    field_name = fieldnames(simParams);
                    
                    tmp_check_var = 0;
                    
                    for check_index = 1:length(field_name(:,1))
                        if strcmp(char(field_name(check_index,1)),['LaneL' num2str(tmp_L)])
                            tmp_check_var = 1;
                        end
                    end
                    
                    if tmp_check_var == 0
                        continue;
                    end
                    
                    eval(['tmp_LaneL_ID = simParams.LaneL' num2str(tmp_L) '.ID;']);
                    
                    if tmp_LaneL_ID == tmp_RoadMarking_Lane_ID
                        
                        tmp_RoadMarking_char = ['[^\n]*RL.1.RoadMarking.' num2str(tmp_RoadMarking_index-1) ' =[^\n]*'];
                        Search_RoadMarking_char = char(regexp(Road_File,tmp_RoadMarking_char,'match'));
                        
                        equal_index = strfind(Search_RoadMarking_char,'=');
                        
                        eval(['tmp_RoadMarking_array = [' strtrim(Search_RoadMarking_char(1,equal_index+1:end-3)) '];']);
                        
                        if tmp_RoadMarking_array(1,9) == 1 || tmp_RoadMarking_array(1,9) == 4
                            tmp_RoadMarking_array_char = 'Single line';
                        elseif tmp_RoadMarking_array(1,9) == 2
                            tmp_RoadMarking_array_char = 'Broken line';
                        end
                        
                        eval(['simParams.Line_' num2str(driving_laneL-tmp_L) '.ID = tmp_RoadMarking_ID_array(1,1);' ]);
                        eval(['simParams.Line_' num2str(driving_laneL-tmp_L) '.LANE_MARKER_TYPE = tmp_RoadMarking_array_char;']);
                        
                        eval(['simParams.LanePath' num2str(driving_laneL-tmp_L) '.ID = simParams.LaneL' num2str(tmp_L) '.ID;' ]);
                        eval(['simParams.LanePath' num2str(driving_laneL-tmp_L) '.TYPE = simParams.LaneL' num2str(tmp_L) '.TYPE;' ]);
                        eval(['simParams.LanePath' num2str(driving_laneL-tmp_L) '.WIDTH = simParams.LaneL' num2str(tmp_L) '.WIDTH;' ]);
                        
                        %                     eval(['simParams.LaneL' num2str(tmp_L) ' = [];']);
                        
                    end
                elseif tmp_Line_index == driving_laneL + 1
                    if Ref_Line_ID == tmp_RoadMarking_Lane_ID
                        
                        tmp_RoadMarking_char = ['[^\n]*RL.1.RoadMarking.' num2str(tmp_RoadMarking_index-1) ' =[^\n]*'];
                        Search_RoadMarking_char = char(regexp(Road_File,tmp_RoadMarking_char,'match'));
                        
                        equal_index = strfind(Search_RoadMarking_char,'=');
                        
                        eval(['tmp_RoadMarking_array = [' strtrim(Search_RoadMarking_char(1,equal_index+1:end-3)) '];']);
                        
                        if tmp_RoadMarking_array(1,9) == 1 || tmp_RoadMarking_array(1,9) == 4
                            tmp_RoadMarking_array_char = 'Single line';
                        elseif tmp_RoadMarking_array(1,9) == 2
                            tmp_RoadMarking_array_char = 'Broken line';
                        end
                        
                        eval(['simParams.Line_' num2str(tmp_Line_index) '.ID = tmp_RoadMarking_ID_array(1,1);' ]);
                        eval(['simParams.Line_' num2str(tmp_Line_index) '.LANE_MARKER_TYPE = tmp_RoadMarking_array_char;']);
                        
                    end
                else
                    tmp_R = tmp_Line_index - (driving_laneL) - 2;
                    
                    eval(['tmp_LaneR_ID = simParams.LaneR' num2str(tmp_R) '.ID;']);
                    
                    if tmp_LaneR_ID == tmp_RoadMarking_Lane_ID
                        
                        tmp_RoadMarking_char = ['[^\n]*RL.1.RoadMarking.' num2str(tmp_RoadMarking_index-1) ' =[^\n]*'];
                        Search_RoadMarking_char = char(regexp(Road_File,tmp_RoadMarking_char,'match'));
                        
                        equal_index = strfind(Search_RoadMarking_char,'=');
                        
                        eval(['tmp_RoadMarking_array = [' strtrim(Search_RoadMarking_char(1,equal_index+1:end-3)) '];']);
                        
                        if tmp_RoadMarking_array(1,9) == 1 || tmp_RoadMarking_array(1,9) == 4
                            tmp_RoadMarking_array_char = 'Single line';
                        elseif tmp_RoadMarking_array(1,9) == 2
                            tmp_RoadMarking_array_char = 'Broken line';
                        end
                        
                        eval(['simParams.Line_' num2str(tmp_Line_index) '.ID = tmp_RoadMarking_ID_array(1,1);' ]);
                        eval(['simParams.Line_' num2str(tmp_Line_index) '.LANE_MARKER_TYPE = tmp_RoadMarking_array_char;']);
                        
                        eval(['simParams.LanePath' num2str(tmp_Line_index-1) '.ID = simParams.LaneR' num2str(tmp_R) '.ID;' ]);
                        eval(['simParams.LanePath' num2str(tmp_Line_index-1) '.TYPE = simParams.LaneR' num2str(tmp_R) '.TYPE;' ]);
                        eval(['simParams.LanePath' num2str(tmp_Line_index-1) '.WIDTH = simParams.LaneR' num2str(tmp_R) '.WIDTH;' ]);
                        
                        %                     eval(['simParams.LaneR' num2str(tmp_R) ' = [];']);
                        
                    end
                end
            end
        end
    end
end

% if strcmp(Scenario_Selection, 'LK_OVE_CU')
%     simParams.TOTAL_LANE = 3;
% elseif strcmp(Scenario_Selection, 'LK_OVE_ST')
%     simParams.TOTAL_LANE = 3;
% elseif strcmp(Scenario_Selection, 'LK_STP_ST')
%     simParams.TOTAL_LANE = 3;
% elseif strcmp(Scenario_Selection, 'LK_CIR_MER')
%     simParams.TOTAL_LANE = driving_laneL + driving_laneR + 1; % 가운데 주행 못하는 차로 추가(가드레일로 막혀있음)
% % elseif
% else
%     simParams.TOTAL_LANE = driving_laneL + driving_laneR;
% end

if strcmp(Cur_Scenario_Selection, 'LK_CIR_MER')
    simParams.TOTAL_LANE = driving_laneL + driving_laneR + 1; % 가운데 주행 못하는 차로 추가(가드레일로 막혀있음)
    % elseif strcmp(Scenario_Selection, 'LK_STP_ST')
    %     simParams.TOTAL_LANE = 6;
else
    simParams.TOTAL_LANE = driving_laneL + driving_laneR;
end


TOTAL_LINE_NUM = driving_laneL + driving_laneR + 1;

invalid_Left_Line_number = simParams.EGO_LANE;
invalid_Right_Line_number = TOTAL_LINE_NUM - simParams.EGO_LANE;


tmp_data_name = fieldnames(data);

for tmp_k = 1:length(tmp_data_name)
    tmp_one_data_name = cell2mat(tmp_data_name(tmp_k));
    
    if ~isempty(char(regexp(tmp_one_data_name,'[^\n]*Sensor_Road_.*Route_[^\n]*','match')))
        tmp_Road_Name = char(regexp(tmp_one_data_name,'_Road_.*_Route_','match'));
        tmp_underbar_index = regexp(tmp_Road_Name,'_');
        Road_Name = tmp_Road_Name(tmp_underbar_index(2)+1:tmp_underbar_index(end-1)-1);
        break
    end
end

Road = zeros(ROAD.MEASURE.STATE_NUMBER, 1, length(sim_time));

Road(ROAD.MEASURE.TOTAL_LANE_NUMBER, :) = TOTAL_LINE_NUM;
Road(ROAD.MEASURE.WIDTH, :) = 3.5 * ones(1,length(sim_time));

Road(ROAD.MEASURE.SHOULDER_EXIST, :) = ON * ones(1,length(sim_time));
Road(ROAD.MEASURE.SHOULDER_WIDTH, :) = 3.5 * ones(1,length(sim_time));

eval(['Road(ROAD.MEASURE.CURVATURE, :) = data.Sensor_Road_' Road_Name '_Route_CurveXY.data;'])
eval(['Road(ROAD.MEASURE.ROAD_SLOPE, :) = data.Sensor_Road_' Road_Name '_Route_DevAng.data;'])

Road(ROAD.MEASURE.DISTANCE_TO_LEFTLANE, :) = data.LinePoly_d_L.data';
Road(ROAD.MEASURE.DISTANCE_TO_RIGHTLANE, :) = data.LinePoly_d_R.data';

%% Line

tmp_data_name = fieldnames(data);

for tmp_k = 1:length(tmp_data_name)
    tmp_one_data_name = tmp_data_name(tmp_k);
    if strcmp(tmp_one_data_name, 'LinePoly_d_L')
        Distance_to_Leftlane = data.LinePoly_d_L.data;
        Distance_to_Rightlane = data.LinePoly_d_R.data;
    else
        Distance_to_Leftlane = 1.75 * ones(length(sim_time),1);
        Distance_to_Rightlane = -1.75 * ones(length(sim_time),1);
    end
end

for tmp_k = 1:length(tmp_data_name)
    tmp_one_data_name = cell2mat(tmp_data_name(tmp_k));
    if ~isempty(char(regexp(tmp_one_data_name,'[^\n]*Sensor_Line_.*nLine_[^\n]*','match')))
        tmp_Line_Name = char(regexp(tmp_one_data_name,'_Line_.*_nLine_','match'));
        tmp_underbar_index = regexp(tmp_Line_Name,'_');
        Line_Name = tmp_Line_Name(tmp_underbar_index(2)+1:tmp_underbar_index(end-1)-1);
        break
    end
end

Line = zeros(LINE.MEASURE.STATE_NUMBER, 2, length(sim_time));

Line(LINE.MEASURE.CURVATURE_RATE, LINE.LEFT, :)         = data.LinePoly_a_L.data;
Line(LINE.MEASURE.CURVATURE, LINE.LEFT, :)              = data.LinePoly_b_L.data;
Line(LINE.MEASURE.ROAD_SLOPE, LINE.LEFT, :)             = data.LinePoly_c_L.data;
Line(LINE.MEASURE.DISTANCE_TO_LINE, LINE.LEFT, :)       = data.LinePoly_d_L.data';
eval(['Line(LINE.MEASURE.LINE_NUMBER, LINE.LEFT, :)     = data.Sensor_Line_' Line_Name '_nLine_Left.data;'])


Line(LINE.PREPROCESSING.CURVATURE_RATE, LINE.LEFT, :)    = 6*data.LinePoly_a_L.data;
Line(LINE.PREPROCESSING.CURVATURE, LINE.LEFT, :)         = 2*data.LinePoly_b_L.data;

Line(LINE.MEASURE.CURVATURE_RATE, LINE.RIGHT, :)    = data.LinePoly_a_R.data;
Line(LINE.MEASURE.CURVATURE, LINE.RIGHT, :)         = data.LinePoly_b_R.data;
Line(LINE.MEASURE.ROAD_SLOPE, LINE.RIGHT, :)        = data.LinePoly_c_R.data;
Line(LINE.MEASURE.DISTANCE_TO_LINE, LINE.RIGHT, :)  = data.LinePoly_d_R.data';
eval(['Line(LINE.MEASURE.LINE_NUMBER, LINE.RIGHT, :)     = data.Sensor_Line_' Line_Name '_nLine_Right.data;'])

Line(LINE.PREPROCESSING.CURVATURE_RATE, LINE.RIGHT, :)    = 6*data.LinePoly_a_R.data;
Line(LINE.PREPROCESSING.CURVATURE, LINE.RIGHT, :)         = 2*data.LinePoly_b_R.data;

leftLineNumber      = simParams.EGO_LANE .* ones(length(sim_time),1);
rightLineNumber     = (simParams.TOTAL_LANE - simParams.EGO_LANE + 1) .* ones(length(sim_time),1);

% data.Sensor_Line_Line_nLine_Left.data, data.Sensor_Line_Line_nLine_Right.data의 첫 값이 무조건 0으로 들어와서 index time 3부터 확인
% 안 그러면 index time 2와 1 차이가 무조건 발생하여 잘못된 차선 개수가 입력됨
for index_time = 1:length(sim_time)
    if index_time > 2
        changedLeftLineNumber = Line(LINE.MEASURE.LINE_NUMBER, LINE.LEFT, index_time) - Line(LINE.MEASURE.LINE_NUMBER, LINE.LEFT, index_time-1);
        changedRightLineNumber = Line(LINE.MEASURE.LINE_NUMBER, LINE.RIGHT, index_time) - Line(LINE.MEASURE.LINE_NUMBER, LINE.RIGHT, index_time-1);
        
        if changedLeftLineNumber ~= 0 || changedRightLineNumber ~= 0
            if strcmp(Cur_Scenario_Selection,'LK_CIR_MER')
                changedRightLineNumber = -1;
            end
            
            leftLineNumber(index_time:end) = leftLineNumber(index_time:end) + changedLeftLineNumber;
            rightLineNumber(index_time:end) = rightLineNumber(index_time:end) + changedRightLineNumber;
        end
    end
end


% figure
% subplot(211)
% hold on; grid on; box on;
% title('# leftLine')
% plot(sim_time, leftLineNumber)
%
%
% subplot(212)
% hold on; grid on; box on;
% title('# rightLine')
% plot(sim_time, rightLineNumber)

%% Generation of Training data

%   1) Add State to Training Data
for track_number = 1:Traffic_Number
    Training_data(:, TRAINING.TIME, track_number)                 = sim_time;
    Training_data(:, TRAINING.EGO_WIDTH, track_number)            = EGO_WIDTH;
    Training_data(:, TRAINING.EGO_LENGTH, track_number)           = EGO_LENGTH;
    
    Training_data(:, TRAINING.REL_POS_X, track_number)        = Class_B(CLASS_B.PREPROCESSING.REL_POS_X, track_number, :);
    Training_data(:, TRAINING.REL_POS_Y, track_number)        = Class_B(CLASS_B.PREPROCESSING.REL_POS_Y, track_number, :);
    Training_data(:, TRAINING.REL_VEL_X, track_number)        = Class_B(CLASS_B.PREPROCESSING.REL_VEL_X, track_number, :);
    Training_data(:, TRAINING.REL_VEL_Y, track_number)        = Class_B(CLASS_B.PREPROCESSING.REL_VEL_Y, track_number, :);
    Training_data(:, TRAINING.VELOCITY, track_number)         = sqrt(Class_B(CLASS_B.PREPROCESSING.REL_VEL_X, track_number, :).^2 + Class_B(CLASS_B.PREPROCESSING.REL_VEL_Y, track_number, :).^2);
    Training_data(:, TRAINING.HEADING_ANGLE, track_number)    = Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE, track_number, :) - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :);
    Training_data(:, TRAINING.TARGET_WIDTH, track_number)     = Class_B(CLASS_B.MEASURE.WIDTH, track_number, :);
    Training_data(:, TRAINING.TARGET_LENGTH, track_number)    = Class_B(CLASS_B.MEASURE.LENGTH, track_number, :);
end



%% Ego vehicle prediction
des_ES=0.9;
des_ELC=3.5;
DEC_AEB=-6;
ACC_PARAM=4;
DEC2_PARAM=-10;
sample=10;
period=CHANNEL.ts;

TJ_X=zeros(length(index_time),sample,7); % 자차 궤적 x (전략 판단 trajectory position: ACC(1) DEC(2) ESR(3) ESL(4) ELCR(5) ELCL(6) ESS(7))
TJ_Y=zeros(length(index_time),sample,7);
ry=zeros(length(sim_time),sample); % 종방향 데이터
rx=zeros(sample,1); % 횡방향 데이터

for index_time = 1:length(sim_time)
    ey_ini(index_time)=(Line(LINE.MEASURE.DISTANCE_TO_LINE, LINE.LEFT, index_time)+Line(LINE.MEASURE.DISTANCE_TO_LINE, LINE.RIGHT, index_time))/2;
    a3=Line(LINE.MEASURE.CURVATURE_RATE, LINE.LEFT, index_time);
    a2=Line(LINE.MEASURE.CURVATURE, LINE.LEFT, index_time);
    a1=Line(LINE.MEASURE.ROAD_SLOPE, LINE.LEFT, index_time);
    v_e=norm([Training_data(index_time, TRAINING.REL_VEL_Y, track_number) Training_data(index_time, TRAINING.REL_VEL_X, track_number)],2);
    t_comp_esl=real(sqrt((4*pi/3)*(des_ES+ey_ini(index_time)))); % refer : Design and evaluation of a model predictive vehicle control algorithm for automated driving using a vehicle traffic simulator
    t_comp_esr=real(sqrt((4*pi/3)*(des_ES-ey_ini(index_time))));
    t_comp_elcl=sqrt((4*pi/3)*(des_ELC+ey_ini(index_time)));
    t_comp_elcr=sqrt((4*pi/3)*(des_ELC-ey_ini(index_time)));

    if t_comp_esl<0.1
        t_comp_esl=0.1;
    end
    if t_comp_esr<0.1
        t_comp_esr=0.1;
    end

    AEB_act= data.LongCtrl_AEB_IsActive.data(index_time); % Fr1(body fixed)
    if AEB_act==0
        DEC_PARAM=0;
    else
        DEC_PARAM=DEC_AEB;
    end


    for i=1:sample
        t=i*period;

        % trajectory num: 1-ACC  2-DEC  3-ESR  4-ESL  5-ELCR  6-ELCL  7-ESS
        % ESR (y) 곡률 적용 전
        TJ_Y(index_time,i,3)=(3/2)*(t_comp_esr/(2*pi))^2*...
            sin((2*pi/t_comp_esr)*(t))...
            -(3*t_comp_esr/(4*pi))*(t);
        if abs(TJ_Y(index_time,i,3))>des_ES-ey_ini(index_time)&& i>1
            TJ_Y(index_time,i,3)=TJ_Y(index_time,i-1,3);
        end

        % ESL (y) 곡률 적용 전
        TJ_Y(index_time,i,4)=(3/2)*(t_comp_esl/(2*pi))^2*...
            sin((2*pi/t_comp_esl)*(t))...
            -(3*t_comp_esl/(4*pi))*(t);
        if abs(TJ_Y(index_time,i,4))>des_ES+ey_ini(index_time)&& i>1
            TJ_Y(index_time,i,4)=TJ_Y(index_time,i-1,4);
        end

        % ELCR (y) 곡률 적용 전
        TJ_Y(index_time,i,5)=(3/2)*(t_comp_elcr/(2*pi))^2*...
            sin((2*pi/t_comp_elcr)*(t))...
            -(3*t_comp_elcr/(4*pi))*(t);

        % ELCL (y) 곡률 적용 전
        TJ_Y(index_time,i,6)=(3/2)*(t_comp_elcr/(2*pi))^2*...
            sin((2*pi/t_comp_elcr)*(t))...
            -(3*t_comp_elcr/(4*pi))*(t);

        % ACC (x)
        TJ_X(index_time,i,1)=v_e*t+0.5*ACC_PARAM*t.^2;
        % DEC (x)
        TJ_X(index_time,i,2)=v_e*t+0.5*DEC2_PARAM*t.^2;
        % ESR (x) / ESL (x)
        TJ_X(index_time,i,3)=v_e*t+0.5*DEC_PARAM*t.^2;
        % ELCR (x) / ELCL (x)
        TJ_X(index_time,i,5)=v_e*t+0.5*DEC_PARAM*t.^2;

        % Ego trajectory
        rx(i)=TJ_X(index_time,i,3);
        ry(index_time,i)=a3*rx(i).^3+a2*rx(i).^2+a1*rx(i);

        % ESR (y) 곡률 적용 후
        TJ_Y(index_time,i,3)=TJ_Y(index_time,i,3)-ry(index_time,i);
        % ESL (y) 곡률 적용 후
        TJ_Y(index_time,i,4)=TJ_Y(index_time,i,4)-ry(index_time,i);
        % ELCR (y) 곡률 적용 후
        TJ_Y(index_time,i,5)=TJ_Y(index_time,i,5)-ry(index_time,i);
        % ELCL (y) 곡률 적용 후
        TJ_Y(index_time,i,6)=TJ_Y(index_time,i,6)-ry(index_time,i);
        % ESS (y) 곡률 적용 후
        TJ_Y(index_time,i,7)=TJ_Y(index_time,i,5);
    end
end

TJ_Y(:,:,1) = 0; % ACC (y)
TJ_Y(:,:,2) = 0; % DEC (y)

TJ_X(:,:,4) = TJ_X(:,:,3); % ESL (x) = ESR (x)
TJ_Y(:,:,4) = -TJ_Y(:,:,4); % ESL (y)

TJ_X(:,:,6) = TJ_X(:,:,5); % ELCL (x) = ELCR (x)
TJ_Y(:,:,6) = - TJ_Y(:,:,6); % ELCL (y)

TJ_X(:,:,7) = TJ_X(:,:,4); % ESS (x)
TJ_Y(:,:,7) = TJ_Y(:,:,7); % ESS (y)


%% surrounding vehicle prediction (IMM-UKF)
% param
Prob_ctrv_ini=0.5;
Prob_cv_ini=0.5;
SAMPLE_TIME=0.01;
Q_CTRV_IMM                                              = (diag([1 1 1*pi/180 2 15*pi/180])*SAMPLE_TIME).^2;
R_CTRV_IMM                                              = (diag([5 5 30*pi/180])).^2;
Q_CV_IMM                                                = (diag([1 1 1.5*pi/180 2 2])*SAMPLE_TIME).^2;
R_CV_IMM                                                = (diag([5 5 30*pi/180])).^2;
flag=0;
x_ctrv_tmp=zeros(5,length(sim_time),10);
P_ctrv_tmp=zeros(5,5,length(sim_time),10);
x_cv_tmp=zeros(5,length(sim_time),10);
P_cv_tmp=zeros(5,5,length(sim_time),10);
X_pred=zeros(5,length(sim_time),10,Traffic_Number); % [y x yaw v yawrate]
P_pred=zeros(5,5,length(sim_time),10,Traffic_Number);
% prob_matrix = [ 0.9 0.1; 0.1 0.9];
prob_matrix = [ 0.981 0.019; 0.019 0.981];

for track_number = 1:Traffic_Number
    for index_time = 1:length(sim_time)
        old_flag=flag;
        if (Training_data(index_time, TRAINING.REL_POS_X, track_number) >= X_MIN && Training_data(index_time, TRAINING.REL_POS_X, track_number) <= X_MAX ...
                && Training_data(index_time, TRAINING.REL_POS_Y, track_number) >= Y_MIN && Training_data(index_time, TRAINING.REL_POS_Y, track_number) <= Y_MAX)
            flag=1;
        else
            flag=0;
        end
        
        if old_flag==1
            old_flag=old_flag;
        end
        x=Training_data(index_time, TRAINING.REL_POS_X, track_number) ;
        y=Training_data(index_time, TRAINING.REL_POS_Y, track_number);
        v=-norm([Training_data(index_time, TRAINING.REL_VEL_Y, track_number) Training_data(index_time, TRAINING.REL_VEL_X, track_number)],2);

        theta=-Training_data(index_time, TRAINING.HEADING_ANGLE, track_number);
        if strcmp(Cur_Scenario_Selection, 'LK_OVE_ST')||strcmp(Cur_Scenario_Selection, 'LK_OVE_CU')
        v=-v;theta=-theta;
        end
        y_out = [y,x,theta]';
        x_ini = [y,x,theta,v,0]'; % [y,x,theta,v,w]
        if old_flag==0 &&flag==1 || index_time==1
            Prob_ctrv_old=Prob_ctrv_ini;
            Prob_cv_old=Prob_cv_ini;
            x_ctrv_old=zeros(5,1);
            x_cv_old=zeros(5,1);
            P_ctrv_old=zeros(5,5);
            P_cv_old=zeros(5,5);
        end
            x_ctrv_old=x_ini;%zeros(5,1);
            x_cv_old=x_ini;%zeros(5,1);

        [c_ctrv, x_ctrv_out, P_ctrv_out, c_cv, x_cv_out, P_cv_out]= Interacting(flag,Prob_ctrv_old,Prob_cv_old,x_ctrv_old,x_cv_old,P_ctrv_old,P_cv_old,prob_matrix);
        
        if  (Training_data(index_time, TRAINING.REL_POS_X, track_number) >= X_MIN && Training_data(index_time, TRAINING.REL_POS_X, track_number) <= X_MAX ...
                && Training_data(index_time, TRAINING.REL_POS_Y, track_number) >= Y_MIN && Training_data(index_time, TRAINING.REL_POS_Y, track_number) <= Y_MAX)

        for pred_length=1:10
            ts=pred_length*period;
            [x_ctrv,P_ctrv, mu_ctrv] = CTRV_MODEL(x_ctrv_out,P_ctrv_out,y_out,x_ini,flag,old_flag,Q_CTRV_IMM, R_CTRV_IMM,ts);
            if pred_length==1
                x_ctrv_old=x_ctrv;
                P_ctrv_old=P_ctrv;
            end
            
            [x_cv,P_cv, mu_cv] = CV_MODEL(x_cv_out,P_cv_out,y_out,x_ini,flag,old_flag,Q_CV_IMM, R_CV_IMM,ts);
            if pred_length==1
                x_cv_old=x_cv;
                P_cv_old=P_cv;
            end
            
            [Prob_ctrv,Prob_cv,X_c,P_c] = Mixing(c_ctrv,x_ctrv,P_ctrv,mu_ctrv,c_cv,x_cv,P_cv,mu_cv);
            
            if  pred_length==1
                Prob_ctrv_old=Prob_ctrv;
                Prob_cv_old=Prob_cv;
            end
            X_pred(:,index_time,pred_length,track_number)=X_c; % [state time length_of_pred track_num], state :[y,x,theta,v,w]
            P_pred(:,:,index_time,pred_length,track_number)= P_c;
            Pr_ctrv(:,index_time,pred_length,track_number)=Prob_ctrv;
            Pr_cv(:,index_time,pred_length,track_number)=Prob_cv;
        end
        else
            for pred_length=1:10
                X_pred(:,index_time,pred_length,track_number)=[1000 1000 0 0 0];
                P_pred(:,:,index_time,pred_length,track_number)= zeros(5,5);
            end
        end
    end
end
%% Threat Metric
for index_time = 1:length(sim_time)
    if index_time == 1001
        a = 1;
    end
    
    for track_number = 1:Traffic_Number
        
        % Add Threat Metric to Training Data
        target_vel_x = Training_data(index_time, TRAINING.REL_VEL_X, track_number) + In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time);
        target_vel_y = Training_data(index_time, TRAINING.REL_VEL_Y, track_number) + In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LAT_VEL,1,index_time);
        %% TTC
        [tmp_TTC] = TTC(Training_data(index_time, TRAINING.REL_POS_X, track_number), Training_data(index_time, TRAINING.REL_POS_Y, track_number), Training_data(index_time, TRAINING.REL_VEL_X, track_number), TTC_PARAM);
        TTC_out(index_time, 1) = tmp_TTC;
        TTC_inverse_out(index_time,1) = 1/tmp_TTC;
        if TTC_inverse_out(index_time)>TTC_INVERSE_PARAM.TTC_INVERSE_MAX
            TTC_inverse_out(index_time)=TTC_INVERSE_PARAM.TTC_INVERSE_MAX;
        end
        
        Training_data(index_time, TRAINING.TTC, track_number)                        = TTC_out(index_time, 1);
        Training_data(index_time, TRAINING.TTC_INVERSE, track_number)                = TTC_inverse_out(index_time, 1);
        %% TLC
        [tmp_TLC, tmp_DLC] = TLC(Training_data(index_time, TRAINING.REL_POS_Y, track_number), Training_data(index_time, TRAINING.REL_VEL_Y, track_number), ...
            Training_data(index_time, TRAINING.HEADING_ANGLE, track_number), Training_data(index_time, TRAINING.TARGET_WIDTH, track_number), Training_data(index_time, TRAINING.TARGET_LENGTH, track_number), ...
            Distance_to_Leftlane(index_time), Distance_to_Rightlane(index_time), TLC_PARAM);
        
        TLC_out(index_time, 1) = tmp_TLC;
        DLC_out(index_time) = tmp_DLC;
        TLC_inverse_out(index_time) = 1/tmp_TLC;
        if TLC_inverse_out(index_time)>TLC_INVERSE_PARAM.TLC_INVERSE_MAX
            TLC_inverse_out(index_time)=TLC_INVERSE_PARAM.TLC_INVERSE_MAX;
        end
        
        Training_data(index_time, TRAINING.TLC, track_number)                        = TLC_out(index_time, 1);
        Training_data(index_time, TRAINING.TLC_INVERSE, track_number)                = TLC_inverse_out(index_time, 1);
        %% Ilat (lateral collision index)
        % lateral: Ilat(combined and single),DLC and TLC
        % longitudinal : Ilong,dw,dbr,xp and TTC
        %         I_LAT_PARAM.TTC_INVERSE_THRESHOLD=4;
        %         I_LAT_PARAM.A_X_MAX=-10;
        
        %         I_LAT_PARAM.A_X_MAX=-1;
        [tmp_I_lat, tmp_I_long, tmp_TTC, tmp_x_p, tmp_d_br, tmp_d_w, tmp_DLC, tmp_TLC] = I_lat_nolimit(Training_data(index_time, TRAINING.REL_POS_X, track_number), Training_data(index_time, TRAINING.REL_POS_Y, track_number),...
            Training_data(index_time, TRAINING.REL_VEL_X, track_number), Training_data(index_time, TRAINING.REL_VEL_Y, track_number), ...
            In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time), target_vel_x, Training_data(index_time, TRAINING.HEADING_ANGLE, track_number),...
            Training_data(index_time, TRAINING.TARGET_WIDTH, track_number), Training_data(index_time, TRAINING.TARGET_LENGTH, track_number), Training_data(index_time, TRAINING.EGO_LENGTH, track_number), ...
            Distance_to_Leftlane(index_time), Distance_to_Rightlane(index_time), I_LAT_PARAM);
        
        I_lat_out(index_time) = tmp_I_lat;
        I_long_out(index_time) = tmp_I_long;
        TLC_out(index_time, 1) = tmp_TLC;

        x_p(index_time) = tmp_x_p;
        
        Training_data(index_time, TRAINING.I_LAT, track_number)                        = I_lat_out(index_time, 1);
        Training_data(index_time, TRAINING.I_LONG, track_number)                       = I_long_out(index_time, 1);
                Training_data(index_time, TRAINING.TLC, track_number)                  = TLC_out(index_time, 1);

        %         I_LAT_PARAM.A_X_MAX=-10;
        [tmp_I_lat, tmp_I_long, tmp_TTC, tmp_x_p, tmp_d_br, tmp_d_w, tmp_DLC, tmp_TLC] = I_lat(Training_data(index_time, TRAINING.REL_POS_X, track_number), Training_data(index_time, TRAINING.REL_POS_Y, track_number),...
            Training_data(index_time, TRAINING.REL_VEL_X, track_number), Training_data(index_time, TRAINING.REL_VEL_Y, track_number), ...
            In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time), target_vel_x, Training_data(index_time, TRAINING.HEADING_ANGLE, track_number),...
            Training_data(index_time, TRAINING.TARGET_WIDTH, track_number), Training_data(index_time, TRAINING.TARGET_LENGTH, track_number), Training_data(index_time, TRAINING.EGO_LENGTH, track_number), ...
            Distance_to_Leftlane(index_time), Distance_to_Rightlane(index_time), I_LAT_PARAM);
        I_lat_out2(index_time) = tmp_I_lat;
        I_long_out2(index_time) = tmp_I_long;
        x_p2(index_time) = tmp_x_p;

        %% RSS (minimum safe distance x and y)
        [tmp_rss_x,tmp_rss_y]= RSS_model(Training_data(index_time, TRAINING.REL_POS_X, track_number),Training_data(index_time, TRAINING.REL_POS_Y, track_number),In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time),...
            target_vel_x, In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LAT_VEL,1,index_time),target_vel_y,RSS_Param);
        
        RSS_x(index_time) = tmp_rss_x;
        RSS_y(index_time) = tmp_rss_y;
        
        Training_data(index_time, TRAINING.RSS_X, track_number)                        = RSS_x(index_time, 1);
        Training_data(index_time, TRAINING.RSS_Y, track_number)                        = RSS_y(index_time, 1);
        
        %% Honda warning and avoidance algorithm (dw,dbr)
        
        [tmp_HONDA_w, tmp_HONDA_br] = HONDA(Training_data(index_time, TRAINING.REL_POS_X, track_number), Training_data(index_time, TRAINING.REL_POS_Y, track_number),...
            Training_data(index_time, TRAINING.REL_VEL_X, track_number), In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time),target_vel_x, HONDA_PARAM);
        HONDA_w(index_time) = tmp_HONDA_w;
        HONDA_br(index_time) = tmp_HONDA_br;
        
        % THM(HONDA)
        THM(index_time)=(Training_data(index_time, TRAINING.REL_POS_X, track_number) - HONDA_br(index_time))/(HONDA_w(index_time)-HONDA_br(index_time));
        
        Training_data(index_time, TRAINING.HONDA_W, track_number)                        = HONDA_w(index_time, 1);
        Training_data(index_time, TRAINING.HONDA_BR, track_number)                       = HONDA_br(index_time, 1);
        Training_data(index_time, TRAINING.THM, track_number)                            = THM(index_time, 1);
        
        Training_data(:, TRAINING.AEB_ACT, track_number)  = data.LongCtrl_AEB_IsActive.data;

        %% MSS
        %         if abs(Training_data(index_time, TRAINING.REL_POS_Y, track_number))>2
        %            if   track_number==4 &&index_time==3000
        %                 track_number=track_number;
        %             end
        %             local_time= 0 : 0.1 : 2;
        %             REACT_TIME=0.1;
        %             if Training_data(index_time, TRAINING.REL_POS_X, track_number) > 0
        %                 if Training_data(index_time, TRAINING.REL_POS_Y, track_number)< 0 %FVR
        %                     FVRMSS_min  = max(-(Training_data(index_time, TRAINING.REL_VEL_X, track_number))*local_time + 0.5 * (In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_ACC,1,index_time)) *local_time.^2);
        %                     FVRMSS_safe = max(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time)*local_time + 0.5*In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_ACC,1,index_time)*local_time.^2);
        %                     if Training_data(index_time, TRAINING.REL_POS_X, track_number)  <= FVRMSS_min
        %                         Training_data(index_time, TRAINING.FVRMSS, track_number)  = 1;
        %                     elseif Training_data(index_time, TRAINING.REL_POS_X, track_number)  > FVRMSS_safe
        %                         Training_data(index_time, TRAINING.FVRMSS, track_number)  = 0;
        %                     else
        %                         Training_data(index_time, TRAINING.FVRMSS, track_number)  =...
        %                             1 - (Training_data(index_time, TRAINING.REL_POS_X, track_number) - FVRMSS_min)/(FVRMSS_safe-FVRMSS_min);
        %                     end
        %                 else
        %                     FVLMSS_min  = max(-(Training_data(index_time, TRAINING.REL_VEL_X, track_number))*local_time +...
        %                         0.5 * (In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_ACC,1,index_time)) *local_time.^2);
        %                     FVLMSS_safe = max(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time)*local_time +...
        %                         0.5*In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_ACC,1,index_time)*local_time.^2);
        %                     if Training_data(index_time, TRAINING.REL_POS_X, track_number)  <= FVLMSS_min
        %                         Training_data(index_time, TRAINING.FVLMSS, track_number)  = 1;
        %                     elseif Training_data(index_time, TRAINING.REL_POS_X, track_number)  > FVLMSS_safe
        %                         Training_data(index_time, TRAINING.FVLMSS, track_number)  = 0;
        %                     else
        %                         Training_data(index_time, TRAINING.FVLMSS, track_number)  = ...
        %                             1 - (Training_data(index_time, TRAINING.REL_POS_X, track_number) - FVLMSS_min)/(FVLMSS_safe-FVLMSS_min);
        %                     end
        %                 end
        %             elseif Training_data(index_time, TRAINING.REL_POS_X, track_number) < 0
        %                 if Training_data(index_time, TRAINING.REL_POS_Y, track_number)< 0 %RVR
        %                     RVRMSS_min  = max(-(Training_data(index_time, TRAINING.REL_VEL_X, track_number))*local_time - 0.5 * (0 - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_ACC,1,index_time)) *local_time.^2);
        %                     RVRMSS_safe = RVRMSS_min - REACT_TIME * (Training_data(index_time, TRAINING.REL_POS_X, track_number) + In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time)) + Training_data(index_time, TRAINING.REL_VEL_X, track_number)^2 * 4 / (2 * 9);
        %
        %                     if Training_data(index_time, TRAINING.REL_POS_X, track_number) <= RVRMSS_min
        %                         Training_data(index_time, TRAINING.RVRMSS, track_number) = 1;
        %                     elseif Training_data(index_time, TRAINING.REL_POS_X, track_number) > RVRMSS_safe
        %                         Training_data(index_time, TRAINING.RVRMSS, track_number) = 0;
        %                     else
        %                         Training_data(index_time, TRAINING.RVRMSS, track_number) = ...
        %                             1 - (Training_data(index_time, TRAINING.REL_POS_X, track_number) - RVRMSS_min)/(RVRMSS_safe-RVRMSS_min);
        %                     end
        %                 else
        %                     RVLMSS_min  = max(-(Training_data(index_time, TRAINING.REL_VEL_X, track_number))*local_time - 0.5 * (0 - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_ACC,1,index_time)) *local_time.^2);
        %                     RVLMSS_safe = RVLMSS_min - REACT_TIME * (Training_data(index_time, TRAINING.REL_VEL_X, track_number) + In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time)) + Training_data(index_time, TRAINING.REL_VEL_X, track_number)^2 * 4 / (2 * 9);
        %
        %                     if Training_data(index_time, TRAINING.REL_POS_X, track_number) <= RVLMSS_min
        %                         Training_data(index_time, TRAINING.RVLMSS, track_number) = 1;
        %                     elseif Training_data(index_time, TRAINING.REL_POS_X, track_number) > RVLMSS_safe
        %                         Training_data(index_time, TRAINING.RVLMSS, track_number) = 0;
        %                     else
        %                         Training_data(index_time, TRAINING.RVLMSS, track_number) =...
        %                             1 - (Training_data(index_time, TRAINING.REL_POS_X, track_number) - RVLMSS_min)/(RVLMSS_safe-RVLMSS_min);
        %                     end
        %                 end
        %             end
        %         end
        
    end
end
