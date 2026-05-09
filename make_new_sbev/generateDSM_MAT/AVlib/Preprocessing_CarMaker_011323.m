
% 좌표계 수식 정리 후 코드 반영
%% Initialization
sim_time = data.Time.data;

In_Vehicle_Sensor = zeros(IN_VEHICLE_SENSOR.STATE_NUMBER, 1, length(sim_time));
Class_B = zeros(CLASS_B.STATE_NUMBER, CLASS_B.TRACK_NUMBER, length(sim_time));
Fusion_Track = zeros(FUSION_TRACK.STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));

%% Vehicle Parameters
Search_OuterSkin = char(regexp(Vehicle_File,'[^\n]*Vehicle.OuterSkin =[^\n]*','match'));
Ego_OuterSkin = (Search_OuterSkin(strfind(Search_OuterSkin,'=')+2:end-1));
Ego_OuterSkin_split = strsplit(Ego_OuterSkin, ' ');
RearLowerLeftPoint_positionX = str2double(Ego_OuterSkin_split{1}); % CarMaker GUI > $#> # 
RearLowerLeftPoint_positionY = str2double(Ego_OuterSkin_split{2});
FrontUpperRightPoint_positionX = str2double(Ego_OuterSkin_split{4});
FrontUpperRightPoint_positionY = str2double(Ego_OuterSkin_split{5});

EGO_WIDTH = abs(FrontUpperRightPoint_positionY - RearLowerLeftPoint_positionY);
EGO_LENGTH = abs(FrontUpperRightPoint_positionX - RearLowerLeftPoint_positionX);

% Search_Width = char(regexp(Vehicle_File,'[^\n]*CarGen.Vehicle.Width =[^\n]*','match'));
% EGO_WIDTH = str2double(Search_Width(strfind(Search_Width,'=')+2:end-1))*1/1000;
% 
% Search_Length = char(regexp(Vehicle_File,'[^\n]*CarGen.Vehicle.Length =[^\n]*','match'));
% EGO_LENGTH = str2double(Search_Length(strfind(Search_Length,'=')+2:end-1))*1/1000;

Search_Ego_CG2Rear_Bumper = strtrim(char(regexp(Vehicle_File,'[^\n]*Body.pos =[^\n]*','match')));
eval(['tmp_Ego_CG2Rear_Bumper = [' Search_Ego_CG2Rear_Bumper(strfind(Search_Ego_CG2Rear_Bumper,'=')+2:end) '];']);
EGO_CG2_REAR_BUMPER = tmp_Ego_CG2Rear_Bumper(1,1);
EGO_CG2_FRONT_BUMPER = EGO_LENGTH - EGO_CG2_REAR_BUMPER;

EGO_VEHICLE.EGO_WIDTH = EGO_WIDTH;
EGO_VEHICLE.EGO_LENGTH = EGO_LENGTH;

Search_Body_Mass = char(regexp(Vehicle_File,'[^\n]*Body.mass =[^\n]*','match'));
Body_Mass = str2double(Search_Body_Mass(strfind(Search_Body_Mass,'=')+2:end-1))*1;

Search_WheelCarrier_FL_Mass = char(regexp(Vehicle_File,'[^\n]*WheelCarrier.fl.mass =[^\n]*','match'));
WheelCarrier_FL_Mass = str2double(Search_WheelCarrier_FL_Mass(strfind(Search_WheelCarrier_FL_Mass,'=')+2:end-1))*1;

Search_WheelCarrier_RL_Mass = char(regexp(Vehicle_File,'[^\n]*WheelCarrier.rl.mass =[^\n]*','match'));
WheelCarrier_RL_Mass = str2double(Search_WheelCarrier_RL_Mass(strfind(Search_WheelCarrier_RL_Mass,'=')+2:end-1))*1;

Search_Wheel_FL_Mass = char(regexp(Vehicle_File,'[^\n]*Wheel.fl.mass =[^\n]*','match'));
Wheel_FL_Mass = str2double(Search_Wheel_FL_Mass(strfind(Search_Wheel_FL_Mass,'=')+2:end-1))*1;

Search_Wheel_RL_Mass = char(regexp(Vehicle_File,'[^\n]*Wheel.rl.mass =[^\n]*','match'));
Wheel_RL_Mass = str2double(Search_Wheel_RL_Mass(strfind(Search_Wheel_RL_Mass,'=')+2:end-1))*1;

EGO_MASS = Body_Mass + WheelCarrier_FL_Mass*2 + WheelCarrier_RL_Mass*2 + Wheel_FL_Mass*2 + Wheel_RL_Mass*2;

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
    SIG_Num = 1;
    if strcmp(tmp_Traffic_Name, 'P00')
        eval(['Class_B(CLASS_B.MEASURE.CLASSIFICATION,' num2str(SIG_Num) ', :) = CLASS_B.DESCRIPTION_CLASSIFICATION.PEDESTRIAN;']);
    elseif strcmp(tmp_Traffic_Name, 'C00')
        eval(['Class_B(CLASS_B.MEASURE.CLASSIFICATION,' num2str(SIG_Num) ', :) = CLASS_B.DESCRIPTION_CLASSIFICATION.CYCLIST;']);
    elseif strcmp(tmp_Traffic_Name, 'E00')
        eval(['Class_B(CLASS_B.MEASURE.CLASSIFICATION,' num2str(SIG_Num) ', :) = CLASS_B.DESCRIPTION_CLASSIFICATION.E_SCOOTER;']);
    else
        eval(['Class_B(CLASS_B.MEASURE.CLASSIFICATION,' num2str(SIG_Num) ', :) = CLASS_B.DESCRIPTION_CLASSIFICATION.CAR;']);
    end
    
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
        
        if strcmp(tmp_Traffic_Name, 'P00')
            eval(['Class_B(CLASS_B.MEASURE.CLASSIFICATION,' num2str(SIG_Num) ', :) = CLASS_B.DESCRIPTION_CLASSIFICATION.PEDESTRIAN;']);
        elseif strcmp(tmp_Traffic_Name, 'C00')
            eval(['Class_B(CLASS_B.MEASURE.CLASSIFICATION,' num2str(SIG_Num) ', :) = CLASS_B.DESCRIPTION_CLASSIFICATION.BICYCLE;']);
        elseif strcmp(tmp_Traffic_Name, 'E00')
            eval(['Class_B(CLASS_B.MEASURE.CLASSIFICATION,' num2str(SIG_Num) ', :) = CLASS_B.DESCRIPTION_CLASSIFICATION.E_SCOOTER;']);
        else
            eval(['Class_B(CLASS_B.MEASURE.CLASSIFICATION,' num2str(SIG_Num) ', :) = CLASS_B.DESCRIPTION_CLASSIFICATION.CAR;']);
        end
        
    end
end


%% Preprocessing - Coordinate Transform

Traffic_Number = Traffic_Num;

In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE,:) = data.Car_Yaw.data';
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_POS_X,:) = data.Car_tx.data';  % Fr0(global)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_POS_Y,:) = data.Car_ty.data';  % Fr0(global)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.VEHICLE_SPEED,:) = data.Car_v.data'; % wheel velocity
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_ACC,:) = data.Car_ax.data'; % Fr1(body fixed)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LAT_ACC,:) = data.Car_ay.data'; % Fr1(body fixed)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,:) = data.Car_vx.data'; % Fr1(body fixed)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LAT_VEL,:) = data.Car_vy.data'; % Fr1(body fixed)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.YAW_RATE,:) = data.Car_YawRate.data'; % Fr1(body fixed)

In_Vehicle_Sensor(IN_VEHICLE_SENSOR.PREPROCESSING.LONG_ACC,:) = data.Car_ax.data'; % Fr1(body fixed)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.PREPROCESSING.LAT_ACC,:) = data.Car_ay.data'; % Fr1(body fixed)
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.PREPROCESSING.VEHICLE_SPEED,:) = data.Car_v.data'; % wheel velocity
In_Vehicle_Sensor(IN_VEHICLE_SENSOR.PREPROCESSING.YAW_RATE,:) = data.Car_YawRate.data'; % Fr1(body fixed)


for SIG_Num = 1:Traffic_Number
    VAR_Num = SIG_Num;
    
    tmp_Traffic_Name = char(Traffic_Name_Cell(1,SIG_Num));
    
    eval(['Class_B(CLASS_B.MEASURE.GLO_POS_Y,' num2str(SIG_Num) ', :) = data.Traffic_' tmp_Traffic_Name '_ty.data;']); % Fr0 (global)
    eval(['Class_B(CLASS_B.MEASURE.GLO_POS_X,' num2str(SIG_Num) ', :) = data.Traffic_' tmp_Traffic_Name '_tx.data;']); % Fr0 (global)
    eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_Y,' num2str(SIG_Num) ', :) = data.Traffic_' tmp_Traffic_Name '_v_0_y.data;']); % Fr0 (global)
    eval(['Class_B(CLASS_B.MEASURE.GLO_VEL_X,' num2str(SIG_Num) ', :) = data.Traffic_' tmp_Traffic_Name '_v_0_x.data;']); % Fr0 (global)
    eval(['Class_B(CLASS_B.MEASURE.GLO_ACC_Y,' num2str(SIG_Num) ', :) = data.Traffic_' tmp_Traffic_Name '_a_0_y.data;']); % Fr0 (global)
    eval(['Class_B(CLASS_B.MEASURE.GLO_ACC_X,' num2str(SIG_Num) ', :) = data.Traffic_' tmp_Traffic_Name '_a_0_x.data;']); % Fr0 (global)
    eval(['Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE,' num2str(SIG_Num) ', :) = data.Traffic_' tmp_Traffic_Name '_rz.data;']); % Fr0 (global)
    eval(['Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE_RATE,' num2str(SIG_Num) ', :) = data.Traffic_' tmp_Traffic_Name '_rzv.data;']); % Fr0 (global)
end

for index_time = 1:length(sim_time)
    if index_time > 1
        for track_number = 1:Traffic_Number
            if abs( Class_B(CLASS_B.MEASURE.GLO_ACC_X, track_number, index_time) - Class_B(CLASS_B.MEASURE.GLO_ACC_X, track_number, index_time-1) ) > 50
                Class_B(CLASS_B.MEASURE.GLO_ACC_X, track_number, index_time) = Class_B(CLASS_B.MEASURE.GLO_ACC_X, track_number, index_time-1);
            end

            if abs( Class_B(CLASS_B.MEASURE.GLO_ACC_Y, track_number, index_time) - Class_B(CLASS_B.MEASURE.GLO_ACC_Y, track_number, index_time-1) ) > 50
                Class_B(CLASS_B.MEASURE.GLO_ACC_Y, track_number, index_time) = Class_B(CLASS_B.MEASURE.GLO_ACC_Y, track_number, index_time-1);
            end
        end
    end
end


for track_number = 1:Traffic_Number
    
    % relative heading angle
    Class_B(CLASS_B.PREPROCESSING.HEADING_ANGLE, track_number, :) = (Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE, track_number, :) - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :));
    
    % relative position
    X_FrontCenter_A = EGO_CG2_FRONT_BUMPER.*cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_POS_X, 1, :);
    Y_FrontCenter_A = EGO_CG2_FRONT_BUMPER.*sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_POS_Y, 1, :);
    
    X_AB = Class_B(CLASS_B.MEASURE.GLO_POS_X, track_number, :) - X_FrontCenter_A;
    Y_AB = Class_B(CLASS_B.MEASURE.GLO_POS_Y, track_number, :) - Y_FrontCenter_A;
    
    x_AB = X_AB .* cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + Y_AB .* sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :));
    y_AB = -X_AB .* sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + Y_AB .* cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :));
    
    Class_B(CLASS_B.PREPROCESSING.REL_POS_Y, track_number, :) = y_AB;
    Class_B(CLASS_B.PREPROCESSING.REL_POS_X, track_number, :) = x_AB;

    Class_B(CLASS_B.PREPROCESSING.CENTER_REL_POS_Y, track_number, :) = Class_B(CLASS_B.PREPROCESSING.REL_POS_Y, track_number, :) + (sign(Class_B(CLASS_B.PREPROCESSING.HEADING_ANGLE, track_number, :))) .* Class_B(CLASS_B.MEASURE.LENGTH, track_number, :)/2.*sin(Class_B(CLASS_B.PREPROCESSING.HEADING_ANGLE, track_number, :).*sign(Class_B(CLASS_B.PREPROCESSING.HEADING_ANGLE, track_number, :)));
    Class_B(CLASS_B.PREPROCESSING.CENTER_REL_POS_X, track_number, :) = Class_B(CLASS_B.PREPROCESSING.REL_POS_X, track_number, :) + Class_B(CLASS_B.MEASURE.LENGTH, track_number, :)/2.*cos(+Class_B(CLASS_B.PREPROCESSING.HEADING_ANGLE, track_number, :).*sign(Class_B(CLASS_B.PREPROCESSING.HEADING_ANGLE, track_number, :)));
    
    % relative velocity    
    Class_B(CLASS_B.PREPROCESSING.REL_VEL_X, track_number, :) = Class_B(CLASS_B.MEASURE.GLO_VEL_X, track_number, :) .* cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + ...
        Class_B(CLASS_B.MEASURE.GLO_VEL_Y, track_number, :).*sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL, 1, :);

    Class_B(CLASS_B.PREPROCESSING.REL_VEL_Y, track_number, :) =  -Class_B(CLASS_B.MEASURE.GLO_VEL_X, track_number, :) .* sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) + ...
        Class_B(CLASS_B.MEASURE.GLO_VEL_Y, track_number, :).*cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LAT_VEL, 1, :);
    
    % relative acceleration
    Class_B(CLASS_B.PREPROCESSING.REL_ACC_X, track_number, :) = ( Class_B(CLASS_B.MEASURE.GLO_ACC_X, track_number, :) .* cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) ...
        - Class_B(CLASS_B.MEASURE.GLO_VEL_X, track_number, :) .* sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) .* In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.YAW_RATE,1,:) ...
        + Class_B(CLASS_B.MEASURE.GLO_ACC_Y, track_number, :) .* sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) ...
        + Class_B(CLASS_B.MEASURE.GLO_VEL_Y, track_number, :) .* cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) .* In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.YAW_RATE,1,:) ...
        - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_ACC, 1, :) ...
        - Class_B(CLASS_B.PREPROCESSING.REL_VEL_Y, track_number, :) .* In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.YAW_RATE,1,:) );
    
    Class_B(CLASS_B.PREPROCESSING.REL_ACC_Y, track_number, :) = ( Class_B(CLASS_B.PREPROCESSING.REL_VEL_X, track_number, :) .* In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.YAW_RATE,1,:) ...
        - Class_B(CLASS_B.MEASURE.GLO_ACC_X, track_number, :) .* sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) ...
        - Class_B(CLASS_B.MEASURE.GLO_VEL_X, track_number, :) .* cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) .* In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.YAW_RATE,1,:) ...
        + Class_B(CLASS_B.MEASURE.GLO_ACC_Y, track_number, :) .* cos(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) ...
        - Class_B(CLASS_B.MEASURE.GLO_VEL_Y, track_number, :) .* sin(In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :)) .* In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.YAW_RATE,1,:) ...
        - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LAT_ACC, 1, :) );

    Class_B(CLASS_B.PREPROCESSING.HEADING_ANGLE_RATE, track_number, :) = (Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE_RATE, track_number, :) - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.YAW_RATE, 1, :));
end

%% Delete state of vehicle in occlusion
if strcmp(Cur_Scenario_Selection, 'LK_COL_STP_SH') || strcmp(Cur_Scenario_Selection, 'LK_COL_STP_ST') || strcmp(Cur_Scenario_Selection, 'LK_COR_STP_CU') || strcmp(Cur_Scenario_Selection, 'LK_COR_STP_ST')
    
    tmp_data_name = fieldnames(data);
    for tmp_k = 1:length(tmp_data_name)
        tmp_one_data_name = cell2mat(tmp_data_name(tmp_k));

        if ~isempty(char(regexp(tmp_one_data_name,'[^\n]*Sensor_Camera_.*nObj[^\n]*','match')))
            tmp_underbar_index = regexp(tmp_one_data_name,'_');
            Camera_Name = tmp_one_data_name(tmp_underbar_index(2)+1:tmp_underbar_index(end)-1);
            break
        end
    end

    eval(['Detected = zeros(length(data.Sensor_Camera_' Camera_Name '_nObj.data), Traffic_Num);'])
    eval(['Data_Camera = data.Sensor_Camera_' Camera_Name '_nObj.data;'])

    for SIG_Num = 1:length(Traffic_Name_Cell(1,:))
        tmp_Traffic_Name = char(Traffic_Name_Cell(1,SIG_Num));

        Occluded_Target_ID = 'T00';


        Search_Occluded_Target = char(regexp(Scenario_File,['[^\n]*Name = ' tmp_Traffic_Name '[^\n]*'],'match'));
        str_split_Pedstrain_ObjID = strsplit(Search_Occluded_Target,'.');
        Occluded_ObjID = 16000000+str2double(char(str_split_Pedstrain_ObjID(2)));



        for sample = 1 : length(Data_Camera)

            if sample >= 1436
                a = 1;
            end
            eval(['Detected_obj_num = data.Sensor_Camera_' Camera_Name '_nObj.data(sample);'])

            if Detected_obj_num ~= 0
                for i =  0 : 49
                    eval(['tmp_ObjID = data.Sensor_Camera_' Camera_Name '_Obj_' num2str(i) '_ObjID.data(sample);']);

                    if tmp_ObjID == Occluded_ObjID
                        Detected(sample, SIG_Num) = 1;
                    end
                end
            end
        end
    end

    for index_time = 1:length(sim_time)
        if index_time == 1095
            a = 1;
        end

        for track_number = 1:Traffic_Num
            if Class_B(CLASS_B.PREPROCESSING.REL_POS_X, track_number, index_time)^2 + Class_B(CLASS_B.PREPROCESSING.REL_POS_Y, track_number, index_time)^2 ~= 0
                if Class_B(CLASS_B.MEASURE.CLASSIFICATION, track_number, index_time) == CLASS_B.DESCRIPTION_CLASSIFICATION.CAR && Detected(index_time, track_number) == 0
                    Class_B(:, track_number, index_time) = zeros(CLASS_B.STATE_NUMBER,1);
                end
            end
        end
    end
end


%% Delete state of VRU in Occlusion

if strcmp(Cur_Scenario_Selection, 'LK_PCSR_STP_ST') || strcmp(Cur_Scenario_Selection, 'LK_PCSL_STP_ST') || strcmp(Cur_Scenario_Selection, 'LK_ECSL_STP_ST')
    Detected = zeros(length(data.Sensor_Camera_CA00_nObj.data), Traffic_Num);

    for SIG_Num = 1:length(Traffic_Name_Cell(1,:))
        tmp_Traffic_Name = char(Traffic_Name_Cell(1,SIG_Num));
        if strcmp(Cur_Scenario_Selection, 'LK_PCSR_STP_ST') || strcmp(Cur_Scenario_Selection, 'LK_PCSL_STP_ST')
            Occluded_Target_ID = 'P00';
        elseif strcmp(Cur_Scenario_Selection, 'LK_ECSL_STP_ST')
            Occluded_Target_ID = 'E00';
        end

        if strcmp(tmp_Traffic_Name, Occluded_Target_ID)
            Search_Occluded_Target = char(regexp(Scenario_File,['[^\n]*Name = ' tmp_Traffic_Name '[^\n]*'],'match'));
            str_split_Pedstrain_ObjID = strsplit(Search_Occluded_Target,'.');
            Occluded_ObjID = 16000000+str2double(char(str_split_Pedstrain_ObjID(2)));



            for sample = 1 : length(data.Sensor_Camera_CA00_nObj.data)
                Detected_obj_num = data.Sensor_Camera_CA00_nObj.data(sample);

                if Detected_obj_num ~= 0
                    for i =  0 : 49
                        eval(['tmp_ObjID = data.Sensor_Camera_CA00_Obj_' num2str(i) '_ObjID.data(sample);']);

                        if tmp_ObjID == Occluded_ObjID
                            Detected(sample, SIG_Num) = 1;
                        end
                    end
                end
            end
        end
    end

    for index_time = 1:length(sim_time)
        if index_time == 1095
            a = 1;
        end
        if strcmp(Cur_Scenario_Selection, 'LK_PCSR_STP_ST') || strcmp(Cur_Scenario_Selection, 'LK_PCSL_STP_ST')
            for track_number = 1:Traffic_Num
                if Class_B(CLASS_B.PREPROCESSING.REL_POS_X, track_number, index_time)^2 + Class_B(CLASS_B.PREPROCESSING.REL_POS_Y, track_number, index_time)^2 ~= 0
                    if Class_B(CLASS_B.MEASURE.CLASSIFICATION, track_number, index_time) == CLASS_B.DESCRIPTION_CLASSIFICATION.PEDESTRIAN && Detected(index_time, track_number) == 0
                        Class_B(:, track_number, index_time) = zeros(CLASS_B.STATE_NUMBER,1);
                    end
                end
            end
        elseif strcmp(Cur_Scenario_Selection, 'LK_ECSL_STP_ST')
            for track_number = 1:Traffic_Num
                if Class_B(CLASS_B.PREPROCESSING.REL_POS_X, track_number, index_time)^2 + Class_B(CLASS_B.PREPROCESSING.REL_POS_Y, track_number, index_time)^2 ~= 0
                    if Class_B(CLASS_B.MEASURE.CLASSIFICATION, track_number, index_time) == CLASS_B.DESCRIPTION_CLASSIFICATION.E_SCOOTER && Detected(index_time, track_number) == 0
                        Class_B(:, track_number, index_time) = zeros(CLASS_B.STATE_NUMBER,1);
                    end
                end
            end
        end

    end

end

%% Generation of Fusion Track
for track_number = 1:Traffic_Number
    Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, :)           = Class_B(CLASS_B.PREPROCESSING.REL_POS_Y, track_number, :);
    Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, :)           = Class_B(CLASS_B.PREPROCESSING.REL_POS_X, track_number, :);
    Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_Y, track_number, :)           = Class_B(CLASS_B.PREPROCESSING.REL_VEL_Y, track_number, :);
    Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_X, track_number, :)           = Class_B(CLASS_B.PREPROCESSING.REL_VEL_X, track_number, :);
    Fusion_Track(FUSION_TRACK.MEASURE.ABS_VEL, track_number, :)             = sqrt(Class_B(CLASS_B.PREPROCESSING.REL_VEL_X, track_number, :).^2 + Class_B(CLASS_B.PREPROCESSING.REL_VEL_Y, track_number, :).^2);
    Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_Y, track_number, :)           = Class_B(CLASS_B.PREPROCESSING.REL_ACC_Y, track_number, :);
    Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_X, track_number, :)           = Class_B(CLASS_B.PREPROCESSING.REL_ACC_X, track_number, :);
    Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, :)               = Class_B(CLASS_B.MEASURE.WIDTH, track_number, :);
    Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, :)              = Class_B(CLASS_B.MEASURE.LENGTH, track_number, :);
    Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, :)       = Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE, track_number, :) - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.GLO_HEADING_ANGLE, 1, :);
    Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE_RATE, track_number, :)  = Class_B(CLASS_B.MEASURE.GLO_HEADING_ANGLE_RATE, track_number, :) - In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.YAW_RATE, 1, :);
end


for index_time = 1:length(sim_time)
    for track_number = 1:Traffic_Number
        if Class_B(CLASS_B.PREPROCESSING.REL_POS_X, track_number, index_time)^2 + Class_B(CLASS_B.PREPROCESSING.REL_POS_Y, track_number, index_time)^2 ~= 0
            if Class_B(CLASS_B.MEASURE.CLASSIFICATION, track_number, index_time) == CLASS_B.DESCRIPTION_CLASSIFICATION.PEDESTRIAN
                Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time)   = SHAPE.PEDESTRIAN_CONFIRMED;
                
            elseif Class_B(CLASS_B.MEASURE.CLASSIFICATION, track_number, index_time) == CLASS_B.DESCRIPTION_CLASSIFICATION.CAR
                Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time)   = SHAPE.VEHICLE_CONFIRMED;
                
            elseif Class_B(CLASS_B.MEASURE.CLASSIFICATION, track_number, index_time) == CLASS_B.DESCRIPTION_CLASSIFICATION.CYCLIST
                Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time)   = SHAPE.BICYCLE_CONFIRMED;
                
            elseif Class_B(CLASS_B.MEASURE.CLASSIFICATION, track_number, index_time) == CLASS_B.DESCRIPTION_CLASSIFICATION.E_SCOOTER
                Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time)   = SHAPE.E_SCOOTER_CONFIRMED;
            end
        end
    end
end
%%  Road
Road = zeros(ROAD.MEASURE.STATE_NUMBER, 1, length(sim_time));

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


% Road(ROAD.MEASURE.TOTAL_LANE_NUMBER, :) = TOTAL_LINE_NUM;
Road(ROAD.MEASURE.WIDTH, :) = 3.5 * ones(1,length(sim_time));

eval(['Road(ROAD.MEASURE.CURVATURE, :) = data.Sensor_Road_' Road_Name '_Route_CurveXY.data;'])
eval(['Road(ROAD.MEASURE.ROAD_SLOPE, :) = data.Sensor_Road_' Road_Name '_Route_DevAng.data;'])


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
for Linei=1:length(Line)
    if Line(LINE.MEASURE.DISTANCE_TO_LINE, LINE.LEFT, Linei)==0
        Line(LINE.MEASURE.DISTANCE_TO_LINE, LINE.LEFT, Linei) = data.LinePoly_d_R.data(Linei)+3.5;
        Line(LINE.MEASURE.CURVATURE_RATE, LINE.LEFT, :)       = data.LinePoly_a_R.data(Linei);
        Line(LINE.MEASURE.CURVATURE, LINE.LEFT, :)            = data.LinePoly_b_R.data(Linei);
        Line(LINE.MEASURE.ROAD_SLOPE, LINE.LEFT, :)           = data.LinePoly_c_R.data(Linei);
    end
end
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


% if FOV_Switch
%     tmp_data_name = fieldnames(data);
%     
%     Front_Vision_Detect_index = find(data.Sensor_Object_Camera_Obj_RV_dtct.data ~= 0);
%     Front_Radar_Detect_index = [];
%     
%     for tmp_k = 1:length(tmp_data_name)
%        if strcmp(tmp_data_name(tmp_k), 'Sensor_Object_FrontRadar_relvTgt_dtct')
%            Front_Radar_Detect_index = find(data.Sensor_Object_FrontRadar_relvTgt_dtct.data ~= 0);
%            break
%        elseif strcmp(tmp_data_name(tmp_k), 'Sensor_Object_FrontRadar_Obj_RV_dtct')
%            Front_Radar_Detect_index = find(data.Sensor_Object_FrontRadar_Obj_RV_dtct.data ~= 0);
%            break
%        end
%         
%     end
%     
%     if ~isempty(Front_Radar_Detect_index)
%         Detect_index = union(Front_Vision_Detect_index, Front_Radar_Detect_index);
%     else
%         Detect_index = Front_Vision_Detect_index;
%     end
%     Detect_index = union(Front_Vision_Detect_index, Front_Radar_Detect_index);
%     
%     total_time_index = 1:length(sim_time);
%     
%     Not_Detect_index = setdiff(total_time_index, Detect_index);
%     
%     Training_data(Not_Detect_index,:) = 0;
% end



