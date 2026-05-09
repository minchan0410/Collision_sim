function [SBEV_out, LANE_MARK_FLAG_out, EGO_SHAPE_FLAG_out] = DSMMaxCollisionProbabilityCarMaker_v5(SBEV_in, State_trajectory, laneInfoL, laneInfoR, Target_X_pred, LANE_MARK_FLAG_in, EGO_SHAPE_FLAG_in, SBEV_PARAM, TRAJ_PARAM, EGO_VEHICLE, TRACKING, SAMPLE_TIME)
% simplifiedBEV3Prediction function generates SBEV
%
% BEV_Window_out = simplifiedBEV(Test_start_index, Test_end_index, Length_time_window, time_index, BEV_Window_in, Tmp_State, TRAINING, RANGE, CHANNEL)
% Test_start_index {double}   : start index of time for one concrete scenraio
% Test_end_index {double}     : end index of time for one concrete scenario
% Length_time_window {double} : length of the time window for BEV Window
% time_index {double}         : time index from for-loop
% BEV_Window_in {double}      : initialized BEV Window
% Tmp_State {double}          : Training data
% TRAINING {struct}           : variable description for Training data
% RANGE {struct}              : parameters for normalization and rasterization
% CHANNEL {struct}            : channel related information

% ROI 밖에서 안으로 앞범퍼부터 들어올 때 SBEV 생성되지 않는 부분 수정 120522
% ROI 내에 shape없이 과거 궤적만 존재하는 경우 empty SBEV로 생성
% bounding box의 꼭지점이 1개일 경우 box가 찌그러져 1개일때 1개 초과일 때 구분해서 shape 생성

% simplifiedBEV3Prediction_test9 에서 마지막 predicted state가 ROI 경계에 걸칠때 box가 찌그러지는 문제 수정 
% DSMMaxCollisionProbabilityCarMaker_v1 : simplifiedBEV3Prediction_test10에서 max collision probability를 B ch에 입력
% DSMMaxCollisionProbabilityCarMaker_v2 : DSMMaxCollisionProbabilityCarMaker_v1 에서 prediction option을 반영하도록 수정
% DSMMaxCollisionProbabilityCarMaker_v3 : DSMMaxCollisionProbabilityCarMaker_v2 에서 trajectory에 fading 적용 가능하도록 수정
% DSMMaxCollisionProbabilityCarMaker_v4 : DSMMaxCollisionProbabilityCarMaker_v3 에서 예측 위치를 궤적으로 나타낼 때 과거 trajectory option 적용되던 부분 안되게 수정
% DSMMaxCollisionProbabilityCarMaker_v5 : DSMMaxCollisionProbabilityCarMaker_v4 에서 궤적에 위험도와 fading 동시 적용 가능하도록 수정

Lane_Width = 3.5;

if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
    empty_SBEV = zeros(SBEV_PARAM.IMAGE_HEIGHT, SBEV_PARAM.IMAGE_WIDTH, SBEV_PARAM.IMAGE_CHANNEL);
elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
    empty_SBEV = 255*ones(SBEV_PARAM.IMAGE_HEIGHT, SBEV_PARAM.IMAGE_WIDTH, SBEV_PARAM.IMAGE_CHANNEL);
end

LANE_MARK_FLAG_out = 0;
EGO_SHAPE_FLAG_out = 0;
Target_Shape_Exist_Flag = 0;
Target_Trajectory_Exist_Flag = 0;
Target_Exist_in_Input_SBEV = 0;

SBEV_out = SBEV_in;
ch_field = fieldnames(SBEV_PARAM.CHANNEL_INFO);
CH_LENGTH = length(SBEV_PARAM.CHANNEL_INFO);

if ~isequal(empty_SBEV, SBEV_out)
    Target_Exist_in_Input_SBEV = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check target in ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tmp_target_y_vertex = [-State_trajectory(TRAJ_PARAM.WIDTH, end)/2, -State_trajectory(TRAJ_PARAM.WIDTH, end)/2,...
    State_trajectory(TRAJ_PARAM.WIDTH, end)/2, State_trajectory(TRAJ_PARAM.WIDTH, end)/2, -State_trajectory(TRAJ_PARAM.WIDTH, end)/2];
tmp_target_x_vertex = [0, State_trajectory(TRAJ_PARAM.LENGTH, end), State_trajectory(TRAJ_PARAM.LENGTH, end), 0, 0];

target_y_vertex_rot = tmp_target_x_vertex.*sin(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end)) + tmp_target_y_vertex.*cos(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end));
target_x_vertex_rot = tmp_target_x_vertex.*cos(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end)) - tmp_target_y_vertex.*sin(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end));

target_y = target_y_vertex_rot + State_trajectory(TRAJ_PARAM.REL_POS_Y, end);
target_x = target_x_vertex_rot + State_trajectory(TRAJ_PARAM.REL_POS_X, end);

if ( (min(target_y) >= SBEV_PARAM.RANGE.Y_MIN && min(target_y) <= SBEV_PARAM.RANGE.Y_MAX) || (max(target_y) >= SBEV_PARAM.RANGE.Y_MIN && max(target_y) <= SBEV_PARAM.RANGE.Y_MAX)) ...
        && ((min(target_x) >= SBEV_PARAM.RANGE.X_MIN && min(target_x) <= SBEV_PARAM.RANGE.X_MAX) || (max(target_x) >= SBEV_PARAM.RANGE.X_MIN && max(target_x) <= SBEV_PARAM.RANGE.X_MAX))

    Target_Shape_Exist_Flag = 1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lane Mark
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if SBEV_PARAM.LANE_MARK.ON
    if LANE_MARK_FLAG_in == 0 && Target_Shape_Exist_Flag == 1
        num_left_line = laneInfoL(5);
        num_right_line = laneInfoR(5);

        x_lane_left = SBEV_PARAM.RANGE.X_RANGE;

        i = 0;
        while i < num_left_line
            tmp_lane_y = 0;
            tmp_lane_y= tmp_lane_y + laneInfoL(1)*x_lane_left.^3+...
                laneInfoL(2)*x_lane_left.^2+...
                laneInfoL(3)*x_lane_left+...
                laneInfoL(4);

            tmp_lane_y = tmp_lane_y + i*Lane_Width;

            for i_line=1:length(tmp_lane_y)
                if x_lane_left(i_line) >= SBEV_PARAM.RANGE.X_MIN && x_lane_left(i_line) <= SBEV_PARAM.RANGE.X_MAX...
                        && tmp_lane_y(i_line) >= SBEV_PARAM.RANGE.Y_MIN && tmp_lane_y(i_line) <= SBEV_PARAM.RANGE.Y_MAX
                    [~,X_LINE_uint8] = min(abs(x_lane_left(i_line) - SBEV_PARAM.RANGE.X_RANGE));
                    [~,Y_LINE_uint8] = min(abs(tmp_lane_y(i_line) - SBEV_PARAM.RANGE.Y_RANGE));

                    if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0
                            SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1
                            for i_SBEV = 1:CH_LENGTH/3
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                            end
                        end
                    elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0
                            SBEV_out(X_LINE_uint8, Y_LINE_uint8, SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                            if rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 1 % if remainder = 1, R ch -> current channel number +1, +2 = 0
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = SBEV_PARAM.RGB_MIN;
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 2) = SBEV_PARAM.RGB_MIN;
                            elseif rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 2 % if remainder = 2, G ch -> current channel number -1, +1 = 0
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = SBEV_PARAM.RGB_MIN;
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = SBEV_PARAM.RGB_MIN;
                            elseif rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 3 % if remainder = 3, B ch -> current channel number -2, -1 = 0
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 2) = SBEV_PARAM.RGB_MIN;
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = SBEV_PARAM.RGB_MIN;
                            end

                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1
                            for i_SBEV = 1:CH_LENGTH/3
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                if rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 1 % if remainder = 1, R ch -> current channel number +1, +2 = 0
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = SBEV_PARAM.RGB_MIN;
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 2) = SBEV_PARAM.RGB_MIN;
                                elseif rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 2 % if remainder = 2, G ch -> current channel number -1, +1 = 0
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = SBEV_PARAM.RGB_MIN;
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = SBEV_PARAM.RGB_MIN;
                                elseif rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 3 % if remainder = 3, B ch -> current channel number -2, -1 = 0
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 2) = SBEV_PARAM.RGB_MIN;
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = SBEV_PARAM.RGB_MIN;
                                end
                            end
                        end
                    end
                end
            end

            i = i+1;
        end

        %         if time_index == 2272
        %             figure
        %             imshow(uint8(SBEV_out))
        %         end

        x_lane_right = SBEV_PARAM.RANGE.X_RANGE;

        i = 0;
        while i < num_right_line
            tmp_lane_y = 0;
            tmp_lane_y= tmp_lane_y + laneInfoR(1)*x_lane_right.^3+...
                laneInfoR(2)*x_lane_right.^2+...
                laneInfoR(3)*x_lane_right+...
                laneInfoR(4);

            tmp_lane_y = tmp_lane_y - i*Lane_Width;

            for i_line=1:length(tmp_lane_y)
                if x_lane_right(i_line) >= SBEV_PARAM.RANGE.X_MIN && x_lane_right(i_line) <= SBEV_PARAM.RANGE.X_MAX...
                        && tmp_lane_y(i_line) >= SBEV_PARAM.RANGE.Y_MIN && tmp_lane_y(i_line) <= SBEV_PARAM.RANGE.Y_MAX
                    [~,X_LINE_uint8] = min(abs(x_lane_right(i_line) - SBEV_PARAM.RANGE.X_RANGE));
                    [~,Y_LINE_uint8] = min(abs(tmp_lane_y(i_line) - SBEV_PARAM.RANGE.Y_RANGE));

                    if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0
                            SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1
                            for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                            end
                        end
                    elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0
                            SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                            if rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 1 % if remainder = 1, R ch -> current channel number +1, +2 = 0
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = SBEV_PARAM.RGB_MIN;
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 2) = SBEV_PARAM.RGB_MIN;
                            elseif rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 2 % if remainder = 2, G ch -> current channel number -1, +1 = 0
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = SBEV_PARAM.RGB_MIN;
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = SBEV_PARAM.RGB_MIN;
                            elseif rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 3 % if remainder = 3, B ch -> current channel number -2, -1 = 0
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 2) = SBEV_PARAM.RGB_MIN;
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = SBEV_PARAM.RGB_MIN;
                            end

                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1
                            for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                if rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 1 % if remainder = 1, R ch -> current channel number +1, +2 = 0
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = SBEV_PARAM.RGB_MIN;
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 2) = SBEV_PARAM.RGB_MIN;
                                elseif rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 2 % if remainder = 2, G ch -> current channel number -1, +1 = 0
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = SBEV_PARAM.RGB_MIN;
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = SBEV_PARAM.RGB_MIN;
                                elseif rem(SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 3 % if remainder = 3, B ch -> current channel number -2, -1 = 0
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 2) = SBEV_PARAM.RGB_MIN;
                                    SBEV_out(X_LINE_uint8,Y_LINE_uint8,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = SBEV_PARAM.RGB_MIN;
                                end
                            end
                        end
                    end
                end
            end

            i = i+1;
        end

        LANE_MARK_FLAG_in = 1;
        LANE_MARK_FLAG_out = LANE_MARK_FLAG_in;

        %         if time_index == 954
        %             figure
        %             imshow(uint8(SBEV_out))
        %         end

    else
        LANE_MARK_FLAG_out = LANE_MARK_FLAG_in;
    end
else
    LANE_MARK_FLAG_out = LANE_MARK_FLAG_in;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ego Shape
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if SBEV_PARAM.SHAPE.EGO == 1
    if Target_Shape_Exist_Flag == 1 && EGO_SHAPE_FLAG_in == 0
        
        tmp_ego_y_vertex = [-EGO_VEHICLE.EGO_WIDTH/2, -EGO_VEHICLE.EGO_WIDTH/2,...
            EGO_VEHICLE.EGO_WIDTH/2, EGO_VEHICLE.EGO_WIDTH/2, -EGO_VEHICLE.EGO_WIDTH/2];
        tmp_ego_x_vertex = [-EGO_VEHICLE.EGO_LENGTH, 0, 0, -EGO_VEHICLE.EGO_LENGTH, -EGO_VEHICLE.EGO_LENGTH];
        
        ego_x_contour_total = zeros(200,1);
        ego_y_contour_total = zeros(200,1);
        
        i_row = 1;
        f_row = 0;
        
        for tmp_i = 1:length(tmp_ego_y_vertex) - 1
            tmp_y_vertex0 = tmp_ego_y_vertex(tmp_i);
            tmp_x_vertex0 = tmp_ego_x_vertex(tmp_i);
            
            tmp_y_vertex1 = tmp_ego_y_vertex(tmp_i+1);
            tmp_x_vertex1 = tmp_ego_x_vertex(tmp_i+1);
            
            [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
            [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));
            
            [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
            [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));
            
            [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);
            
            f_row = i_row + length(tmp_x_contour) - 1;
            ego_x_contour_total(i_row:f_row) = tmp_x_contour;
            ego_y_contour_total(i_row:f_row) = tmp_y_contour;
            
            i_row = f_row + 1;
        end
        ego_x_contour_total(f_row+1:end) = [];
        ego_y_contour_total(f_row+1:end) = [];
        
        pixel_info = zeros(f_row,3);
        [sorted_x_contour_total, sorted_index] = sort(ego_x_contour_total);
        sorted_y_contour_total = ego_y_contour_total(sorted_index);
        y_i = 1000;
        y_f = 0;
        i_row = 0;
        
        for tmp_i = 1:length(ego_x_contour_total) - 1
            
            if sorted_x_contour_total(tmp_i) == sorted_x_contour_total(tmp_i + 1)
                
                tmp_y = sorted_y_contour_total(tmp_i);
                
                if tmp_y > y_f
                    y_f = tmp_y;
                end
                
                if tmp_y < y_i
                    y_i = tmp_y;
                end
                
                if tmp_i == length(ego_x_contour_total) - 1
                    i_row = i_row + 1;
                    pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);
                    
                    if y_i > sorted_y_contour_total(tmp_i + 1)
                        y_i = sorted_y_contour_total(tmp_i + 1);
                    end
                    
                    if y_f < sorted_y_contour_total(tmp_i + 1)
                        y_f = sorted_y_contour_total(tmp_i + 1);
                    end
                    
                    pixel_info(i_row,2) = y_i;
                    pixel_info(i_row,3) = y_f;
                end
                
            else
                i_row = i_row + 1;
                pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);
                
                if tmp_i == 1
                    y_i = sorted_y_contour_total(tmp_i);
                    y_f = y_i;
                elseif tmp_i == length(ego_x_contour_total) - 1
                    pixel_info(i_row + 1,2) = sorted_y_contour_total(tmp_i + 1);
                    pixel_info(i_row + 1,3) = sorted_y_contour_total(tmp_i + 1);
                else
                    if y_i == y_f
                        if sorted_y_contour_total(tmp_i - 1) > sorted_y_contour_total(tmp_i)
                            y_i = sorted_y_contour_total(tmp_i);
                            y_f = sorted_y_contour_total(tmp_i - 1);
                        elseif sorted_y_contour_total(tmp_i - 1) < sorted_y_contour_total(tmp_i)
                            y_i = sorted_y_contour_total(tmp_i - 1);
                            y_f = sorted_y_contour_total(tmp_i);
                        else
                            y_i = sorted_y_contour_total(tmp_i - 1);
                            y_f = y_i;
                        end
                    else
                        if y_i > sorted_y_contour_total(tmp_i)
                            y_i = sorted_y_contour_total(tmp_i);
                        end
                        
                        if y_f < sorted_y_contour_total(tmp_i)
                            y_f = sorted_y_contour_total(tmp_i);
                        end
                    end
                    
                end
                pixel_info(i_row,2) = y_i;
                pixel_info(i_row,3) = y_f;
                
                y_i = 1000;
                y_f = 0;
            end
        end
        
        if SBEV_PARAM.RGB_IMAGE == 1
            if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                for tmp_j = 1:length(pixel_info(:,1))
                    if pixel_info(tmp_j,1) ~= 0
                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),:) = SBEV_PARAM.RGB_MAX;
                    else
                        break
                    end
                end
                
            elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                for tmp_j = 1:length(pixel_info(:,1))
                    if pixel_info(tmp_j,1) ~= 0
                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),:) = SBEV_PARAM.RGB_MIN;
                    else
                        break
                    end
                end
            end
            
        elseif SBEV_PARAM.GRAY_IMAGE == 1
            if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                for tmp_j = 1:length(pixel_info(:,1))
                    if pixel_info(tmp_j,1) ~= 0
                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),:) = SBEV_PARAM.RGB_MAX;
                    else
                        break
                    end
                end
                
            elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                for tmp_j = 1:length(pixel_info(:,1))
                    if pixel_info(tmp_j,1) ~= 0
                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),:) = SBEV_PARAM.RGB_MIN;
                    else
                        break
                    end
                end
            end
        end
        EGO_SHAPE_FLAG_in = 1;
    end
end
EGO_SHAPE_FLAG_out = EGO_SHAPE_FLAG_in;



% figure
% imshow(uint8(SBEV_out))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target Prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if SBEV_PARAM.PREDICTION.TARGET == 1 && Target_Shape_Exist_Flag == 1
    
    ROI_margin2ego = 0;
    
    if ~isequal(empty_SBEV, SBEV_out)
        
        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

        if SBEV_PARAM.COLLISION_PROBABILITY.ON
            [~, Collision_Probability_uint8] = min(abs(State_trajectory(TRAJ_PARAM.COLLISION_PROBABILITY, end) - SBEV_PARAM.RANGE.COLLISION_PROBABILITY_RANGE));
        end

%         if SBEV_PARAM.PREDICTION.FADING.ON
%             R_ALPHA_STEP = ( (SBEV_PARAM.PREDICTION.FADING.R_ALPHA_MAX - I_LAT_uint8) - SBEV_PARAM.PREDICTION.FADING.R_ALPHA_MIN) / (SBEV_PARAM.PREDICTION.TARGET_PRED_WINDOW / SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE);
% 
%             if SBEV_PARAM.COLLISION_PROBABILITY.ON
%                 B_ALPHA_STEP = ( (SBEV_PARAM.PREDICTION.FADING.B_ALPHA_MAX - Collision_Probability_uint8) - SBEV_PARAM.PREDICTION.FADING.B_ALPHA_MIN) / (SBEV_PARAM.PREDICTION.TARGET_PRED_WINDOW / SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE);
%             end
%         end
        
        
        if sum(Target_X_pred) ~= 0
            predicted_x = squeeze(Target_X_pred(TRACKING.REL_POS_X,1,:)); % x
            predicted_y = squeeze(Target_X_pred(TRACKING.REL_POS_Y,1,:)); % y


            % overlap 허용
            % 이전처럼 순서대로 그리면 됨


            % overlap 미허용
            % 순서대로 넘기면서 겹치는 시점 확인
            % 겹치는 시점에서 거꾸로 그려야함

            if SBEV_PARAM.PREDICTION.OVERLAP_FLAG == 1 % overlap 허용

                for index_pred = 1:SBEV_PARAM.PREDICTION.TARGET_PRED_WINDOW/SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE

                    index_pred_detail = round(index_pred*SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);

                    if SBEV_PARAM.PREDICTION.ALL_SHAPE_FLAG % hollow shape for all prediction time
                        tmp_target_y_vertex = [-State_trajectory(TRAJ_PARAM.WIDTH, end)/2, -State_trajectory(TRAJ_PARAM.WIDTH, end)/2,...
                            State_trajectory(TRAJ_PARAM.WIDTH, end)/2, State_trajectory(TRAJ_PARAM.WIDTH, end)/2, -State_trajectory(TRAJ_PARAM.WIDTH, end)/2];
                        tmp_target_x_vertex = [0, State_trajectory(TRAJ_PARAM.LENGTH, end), State_trajectory(TRAJ_PARAM.LENGTH, end), 0, 0];

                        target_y_vertex_rot = tmp_target_x_vertex.*sin(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end)) + tmp_target_y_vertex.*cos(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end));
                        target_x_vertex_rot = tmp_target_x_vertex.*cos(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end)) - tmp_target_y_vertex.*sin(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end));

                        target_y = target_y_vertex_rot + predicted_y(index_pred_detail);
                        target_x = target_x_vertex_rot + predicted_x(index_pred_detail);

                        ONLY_ONE_VERTEX_ROI_OUT_FLAG = 0;
                        TWO_VERTEX_ROI_OUT_FLAG = 0;
                        THREE_VERTEX_ROI_OUT_FLAG = 0;

                        if ~( all(target_y >= SBEV_PARAM.RANGE.Y_MIN) && all(target_y <= SBEV_PARAM.RANGE.Y_MAX) && all(target_x >= SBEV_PARAM.RANGE.X_MIN) && all(target_x <= SBEV_PARAM.RANGE.X_MAX) )

                            vertex_total = zeros(4, 4);

                            vertex_total(1, :) = target_y(1:4) >= SBEV_PARAM.RANGE.Y_MIN;
                            vertex_total(2, :) = target_y(1:4) <= SBEV_PARAM.RANGE.Y_MAX;
                            vertex_total(3, :) = target_x(1:4) >= SBEV_PARAM.RANGE.X_MIN;
                            vertex_total(4, :) = target_x(1:4) <= SBEV_PARAM.RANGE.X_MAX;

                            vertex_out_flag = all(vertex_total);

                            if nnz(vertex_out_flag) == 3 % only one vertex out of ROI
                                ONLY_ONE_VERTEX_ROI_OUT_FLAG = 1;
                            elseif nnz(vertex_out_flag) == 2 % two vertex out of ROI
                                TWO_VERTEX_ROI_OUT_FLAG = 1;
                            elseif nnz(vertex_out_flag) == 1 % three vertex out of ROI
                                THREE_VERTEX_ROI_OUT_FLAG = 1;
                            end
                        end

                        if ( (min(target_y) >= SBEV_PARAM.RANGE.Y_MIN && min(target_y) <= SBEV_PARAM.RANGE.Y_MAX) || (max(target_y) >= SBEV_PARAM.RANGE.Y_MIN && max(target_y) <= SBEV_PARAM.RANGE.Y_MAX)) ...
                                && ((min(target_x) >= SBEV_PARAM.RANGE.X_MIN && min(target_x) <= SBEV_PARAM.RANGE.X_MAX) || (max(target_x) >= SBEV_PARAM.RANGE.X_MIN && max(target_x) <= SBEV_PARAM.RANGE.X_MAX))

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Find pixel of contour corresponding to predicted position
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            x_contour_total = zeros(200,1);
                            y_contour_total = zeros(200,1);

                            i_row = 1;
                            f_row = 0;

                            if ONLY_ONE_VERTEX_ROI_OUT_FLAG
                                target_y_correction = target_y;
                                target_x_correction = target_x;

                                y_cross = 0;
                                x_cross = 0;

                                for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                    tmp_y_vertex0 = target_y(tmp_i);
                                    tmp_x_vertex0 = target_x(tmp_i);

                                    tmp_y_vertex1 = target_y(tmp_i+1);
                                    tmp_x_vertex1 = target_x(tmp_i+1);

                                    if tmp_i == 1
                                        tmp_y_vertex_1 = target_y(4);
                                        tmp_x_vertex_1 = target_x(4);
                                    else
                                        tmp_y_vertex_1 = target_y(tmp_i - 1);
                                        tmp_x_vertex_1 = target_x(tmp_i - 1);
                                    end

                                    if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                            tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                        if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next and before vertex in ROI
                                                tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX) && ...
                                                (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_i == 1
                                                % tmp_i == 1
                                                % next_start_index_origin = 2;
                                                % next_end_index_origin = 4;
                                                % next_start_index_correction = 3;
                                                % next_end_index_correction = 5;
                                                % tmp_i에 1-1, tmp_i+1에 1-2 new vertex

                                                target_y_correction(3:5) = target_y_correction(2:4);
                                                target_x_correction(3:5) = target_x_correction(2:4);

                                            elseif tmp_i == 2
                                                % tmp_i == 2
                                                % next_start_index_origin = 3;
                                                % next_end_index_origin = 5; -> 4
                                                % next_start_index_correction = 4;
                                                % next_end_index_correction = 6; -> 5
                                                % tmp_i에 2-1, tmp_i+1에 2-2 new vertex

                                                target_y_correction(4:5) = target_y_correction(3:4);
                                                target_x_correction(4:5) = target_x_correction(3:4);

                                            elseif tmp_i == 3
                                                % tmp_i == 3
                                                % next_start_index_origin = 4;
                                                % next_end_index_origin = 6; -> 4
                                                % next_start_index_correction = 5;
                                                % next_end_index_correction = 7; -> 5
                                                % tmp_i에 3-1, tmp_i+1에 3-2 new vertex

                                                target_y_correction(5) = target_y_correction(4);
                                                target_x_correction(5) = target_x_correction(4);

                                            elseif tmp_i == 4
                                                % tmp_i == 4
                                                % next_start_index_origin = 5; -> []
                                                % next_end_index_origin = 7; -> []
                                                % next_start_index_correction = 6; -> []
                                                % next_end_index_correction = 8; -> []
                                                % tmp_i에 4-1, tmp_i+1에 4-2 new vertex

                                            end

                                            % current ~ before vertex
                                            m_1 = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                            tmp_base_1 = tmp_y_vertex_1 - m_1*tmp_x_vertex_1;

                                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                y_cross_1 = SBEV_PARAM.RANGE.Y_MIN;
                                                x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                y_cross_1 = SBEV_PARAM.RANGE.Y_MAX;
                                                x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                                            elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                x_cross_1 = SBEV_PARAM.RANGE.X_MIN;
                                                y_cross_1 = m_1*x_cross_1 + tmp_base_1;

                                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                x_cross_1 = SBEV_PARAM.RANGE.X_MAX;
                                                y_cross_1 = m_1*x_cross_1 + tmp_base_1;
                                            end


                                            % current ~ next vertex
                                            m_2 = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                            tmp_base_2 = tmp_y_vertex1 - m_2*tmp_x_vertex1;

                                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                y_cross_2 = SBEV_PARAM.RANGE.Y_MIN;
                                                x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                y_cross_2 = SBEV_PARAM.RANGE.Y_MAX;
                                                x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                                            elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                x_cross_2 = SBEV_PARAM.RANGE.X_MIN;
                                                y_cross_2 = m_2*x_cross_2 + tmp_base_2;

                                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                x_cross_2 = SBEV_PARAM.RANGE.X_MAX;
                                                y_cross_2 = m_2*x_cross_2 + tmp_base_2;
                                            end

                                            target_y_correction(tmp_i) = y_cross_1;
                                            target_x_correction(tmp_i) = x_cross_1;

                                            target_y_correction(tmp_i+1) = y_cross_2;
                                            target_x_correction(tmp_i+1) = x_cross_2;


                                        elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_x_vertex0 == tmp_x_vertex_1
                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = tmp_x_vertex0;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = tmp_x_vertex0;
                                                end

                                            elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = tmp_y_vertex0;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = tmp_y_vertex0;
                                                end
                                            else
                                                m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = m*x_cross + tmp_base;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = m*x_cross + tmp_base;
                                                end
                                            end


                                            if tmp_i == 1
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;

                                                target_y_correction(5) = y_cross;
                                                target_x_correction(5) = x_cross;
                                            else
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;
                                            end

                                        end
                                    end
                                end

                                for tmp_i = 1:length(target_y_correction)
                                    tmp_y_vertex0 = target_y_correction(tmp_i);
                                    tmp_x_vertex0 = target_x_correction(tmp_i);

                                    if tmp_i < 5
                                        tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                        tmp_x_vertex1 = target_x_correction(tmp_i+1);
                                    else
                                        tmp_y_vertex1 = target_y_correction(1);
                                        tmp_x_vertex1 = target_x_correction(1);
                                    end

                                    [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                    [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                    [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                    f_row = i_row + length(tmp_x_contour) - 1;
                                    x_contour_total(i_row:f_row) = tmp_x_contour;
                                    y_contour_total(i_row:f_row) = tmp_y_contour;

                                    i_row = f_row + 1;

                                end

                            elseif TWO_VERTEX_ROI_OUT_FLAG
                                target_y_correction = target_y;
                                target_x_correction = target_x;

                                y_cross = 0;
                                x_cross = 0;

                                for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                    tmp_y_vertex0 = target_y(tmp_i);
                                    tmp_x_vertex0 = target_x(tmp_i);

                                    tmp_y_vertex1 = target_y(tmp_i+1);
                                    tmp_x_vertex1 = target_x(tmp_i+1);

                                    if tmp_i == 1
                                        tmp_y_vertex_1 = target_y(4);
                                        tmp_x_vertex_1 = target_x(4);
                                    else
                                        tmp_y_vertex_1 = target_y(tmp_i - 1);
                                        tmp_x_vertex_1 = target_x(tmp_i - 1);
                                    end

                                    if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                            tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                        if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                                                tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_x_vertex0 == tmp_x_vertex1
                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = tmp_x_vertex0;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = tmp_x_vertex0;
                                                end

                                            elseif tmp_y_vertex0 == tmp_y_vertex1
                                                if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    y_cross = tmp_y_vertex0;
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    y_cross = tmp_y_vertex0;
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                end
                                            else
                                                m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                                tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = m*x_cross + tmp_base;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = m*x_cross + tmp_base;
                                                end
                                            end

                                            if tmp_i == 1
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;

                                                target_y_correction(5) = y_cross;
                                                target_x_correction(5) = x_cross;
                                            else
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;
                                            end


                                        elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_x_vertex0 == tmp_x_vertex_1
                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = tmp_x_vertex0;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = tmp_x_vertex0;
                                                end

                                            elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = tmp_y_vertex0;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = tmp_y_vertex0;
                                                end
                                            else
                                                m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = m*x_cross + tmp_base;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = m*x_cross + tmp_base;
                                                end
                                            end

                                            if tmp_i == 1
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;

                                                target_y_correction(5) = y_cross;
                                                target_x_correction(5) = x_cross;
                                            else
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;
                                            end
                                        end
                                    end
                                end

                                for tmp_i = 1:length(target_y_correction) - 1
                                    tmp_y_vertex0 = target_y_correction(tmp_i);
                                    tmp_x_vertex0 = target_x_correction(tmp_i);

                                    tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                    tmp_x_vertex1 = target_x_correction(tmp_i+1);

                                    [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                    [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                    [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                    f_row = i_row + length(tmp_x_contour) - 1;
                                    x_contour_total(i_row:f_row) = tmp_x_contour;
                                    y_contour_total(i_row:f_row) = tmp_y_contour;

                                    i_row = f_row + 1;

                                end

                            elseif THREE_VERTEX_ROI_OUT_FLAG
                                target_y_correction = target_y;
                                target_x_correction = target_x;

                                y_cross = 0;
                                x_cross = 0;

                                vertex_index_beforeCurrentNext_all_out = 0;

                                for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                    tmp_y_vertex0 = target_y(tmp_i);
                                    tmp_x_vertex0 = target_x(tmp_i);

                                    tmp_y_vertex1 = target_y(tmp_i+1);
                                    tmp_x_vertex1 = target_x(tmp_i+1);

                                    if tmp_i == 1
                                        tmp_y_vertex_1 = target_y(4);
                                        tmp_x_vertex_1 = target_x(4);
                                    else
                                        tmp_y_vertex_1 = target_y(tmp_i - 1);
                                        tmp_x_vertex_1 = target_x(tmp_i - 1);
                                    end

                                    if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                            tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                        if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                                                tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_x_vertex0 == tmp_x_vertex1
                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = tmp_x_vertex0;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = tmp_x_vertex0;
                                                end

                                            elseif tmp_y_vertex0 == tmp_y_vertex1
                                                if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    y_cross = tmp_y_vertex0;
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    y_cross = tmp_y_vertex0;
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                end
                                            else
                                                m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                                tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = m*x_cross + tmp_base;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = m*x_cross + tmp_base;
                                                end
                                            end

                                            if tmp_i == 1
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;

                                                target_y_correction(5) = y_cross;
                                                target_x_correction(5) = x_cross;
                                            else
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;
                                            end


                                        elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_x_vertex0 == tmp_x_vertex_1
                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = tmp_x_vertex0;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = tmp_x_vertex0;
                                                end

                                            elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = tmp_y_vertex0;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = tmp_y_vertex0;
                                                end
                                            else
                                                m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = m*x_cross + tmp_base;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = m*x_cross + tmp_base;
                                                end
                                            end

                                            if tmp_i == 1
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;

                                                target_y_correction(5) = y_cross;
                                                target_x_correction(5) = x_cross;
                                            else
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;
                                            end

                                        else % current, next, before vertex all out of ROI
                                            vertex_index_beforeCurrentNext_all_out = tmp_i;
                                        end
                                    end
                                end

                                if vertex_index_beforeCurrentNext_all_out ~= 0

                                    % vertex x,y 중 하나라도 ROI에 포함되는 경우
                                    if ( target_y_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.Y_MAX ) || ...
                                            ( target_x_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.X_MAX )

                                        if vertex_index_beforeCurrentNext_all_out == 1
                                            target_y_correction(1) = target_y_correction(4);
                                            target_x_correction(1) = target_x_correction(4);
                                        else
                                            target_y_correction(vertex_index_beforeCurrentNext_all_out) = target_y_correction(vertex_index_beforeCurrentNext_all_out-1);
                                            target_x_correction(vertex_index_beforeCurrentNext_all_out) = target_x_correction(vertex_index_beforeCurrentNext_all_out-1);
                                        end

                                        % 모두 벗어나는 경우
                                    elseif ~(target_y_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.Y_MAX &&...
                                            target_x_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.X_MAX)

                                        if target_y_correction(vertex_index_beforeCurrentNext_all_out) < SBEV_PARAM.RANGE.Y_MIN
                                            target_y_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.Y_MIN;
                                        elseif target_y_correction(vertex_index_beforeCurrentNext_all_out) > SBEV_PARAM.RANGE.Y_MAX
                                            target_y_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.Y_MAX;
                                        end

                                        if target_x_correction(vertex_index_beforeCurrentNext_all_out) < SBEV_PARAM.RANGE.X_MIN
                                            target_x_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.X_MIN;
                                        elseif target_x_correction(vertex_index_beforeCurrentNext_all_out) > SBEV_PARAM.RANGE.X_MAX
                                            target_x_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.X_MAX;
                                        end
                                    end
                                end

                                for tmp_i = 1:length(target_y_correction) - 1
                                    tmp_y_vertex0 = target_y_correction(tmp_i);
                                    tmp_x_vertex0 = target_x_correction(tmp_i);

                                    tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                    tmp_x_vertex1 = target_x_correction(tmp_i+1);

                                    [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                    [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                    [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                    f_row = i_row + length(tmp_x_contour) - 1;
                                    x_contour_total(i_row:f_row) = tmp_x_contour;
                                    y_contour_total(i_row:f_row) = tmp_y_contour;

                                    i_row = f_row + 1;

                                end

                            else
                                for tmp_i = 1:length(tmp_target_y_vertex) - 1
                                    tmp_y_vertex0 = target_y(tmp_i);
                                    tmp_x_vertex0 = target_x(tmp_i);

                                    tmp_y_vertex1 = target_y(tmp_i+1);
                                    tmp_x_vertex1 = target_x(tmp_i+1);

                                    [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                    [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                    [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                    f_row = i_row + length(tmp_x_contour) - 1;
                                    x_contour_total(i_row:f_row) = tmp_x_contour;
                                    y_contour_total(i_row:f_row) = tmp_y_contour;

                                    i_row = f_row + 1;
                                end
                            end

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Find pixel to fill bounding box corresponding to predicted position
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            x_contour_total(f_row+1:end) = [];
                            y_contour_total(f_row+1:end) = [];

                            pixel_info = zeros(f_row,3);
                            [sorted_x_contour_total, sorted_index] = sort(x_contour_total);
                            sorted_y_contour_total = y_contour_total(sorted_index);
                            y_i = 1000;
                            y_f = 0;
                            i_row = 0;

                            for tmp_i = 1:length(x_contour_total) - 1

                                if sorted_x_contour_total(tmp_i) == sorted_x_contour_total(tmp_i + 1)

                                    tmp_y = sorted_y_contour_total(tmp_i);

                                    if tmp_y > y_f
                                        y_f = tmp_y;
                                    end

                                    if tmp_y < y_i
                                        y_i = tmp_y;
                                    end

                                    if tmp_i == length(x_contour_total) - 1
                                        i_row = i_row + 1;
                                        pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);

                                        if y_i > sorted_y_contour_total(tmp_i + 1)
                                            y_i = sorted_y_contour_total(tmp_i + 1);
                                        end

                                        if y_f < sorted_y_contour_total(tmp_i + 1)
                                            y_f = sorted_y_contour_total(tmp_i + 1);
                                        end

                                        pixel_info(i_row,2) = y_i;
                                        pixel_info(i_row,3) = y_f;
                                    end

                                else
                                    i_row = i_row + 1;
                                    pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);

                                    if tmp_i == 1
                                        y_i = sorted_y_contour_total(tmp_i);
                                        y_f = y_i;
                                    elseif tmp_i == length(x_contour_total) - 1
                                        pixel_info(i_row + 1,2) = sorted_y_contour_total(tmp_i + 1);
                                        pixel_info(i_row + 1,3) = sorted_y_contour_total(tmp_i + 1);
                                    else
                                        if y_i == y_f
                                            if sorted_y_contour_total(tmp_i - 1) > sorted_y_contour_total(tmp_i)
                                                y_i = sorted_y_contour_total(tmp_i);
                                                y_f = sorted_y_contour_total(tmp_i - 1);
                                            elseif sorted_y_contour_total(tmp_i - 1) < sorted_y_contour_total(tmp_i)
                                                y_i = sorted_y_contour_total(tmp_i - 1);
                                                y_f = sorted_y_contour_total(tmp_i);
                                            else
                                                y_i = sorted_y_contour_total(tmp_i - 1);
                                                y_f = y_i;
                                            end
                                        else
                                            if y_i > sorted_y_contour_total(tmp_i)
                                                y_i = sorted_y_contour_total(tmp_i);
                                            end

                                            if y_f < sorted_y_contour_total(tmp_i)
                                                y_f = sorted_y_contour_total(tmp_i);
                                            end
                                        end

                                    end
                                    pixel_info(i_row,2) = y_i;
                                    pixel_info(i_row,3) = y_f;

                                    y_i = 1000;
                                    y_f = 0;
                                end
                            end

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Apply pixel information to DSM
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            if SBEV_PARAM.RGB_IMAGE == 1
                                if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                                    if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                        for i_ch = 1:CH_LENGTH
                                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                        else
                                                            break
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                        else
                                                            break
                                                        end
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                for tmp_j = 1:length(pixel_info(:,1))
                                                    if pixel_info(tmp_j,1) ~= 0
                                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                    else
                                                        break
                                                    end
                                                end
                                            end
                                        end

                                    elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                                        for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                            for i_info = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end


                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0
                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end

                                                end
                                            end
                                        end
                                    end

                                elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1

                                    % 수정본
                                    if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                        for i_ch = 1:CH_LENGTH
                                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                if ( SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 ) || SBEV_PARAM.PREDICTION.ON % threat metric in R channel

                                                    if SBEV_PARAM.PREDICTION.FADING.ON
                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - ( I_LAT_uint8 - 1 );
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = ( I_LAT_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                        end
                                                    else
                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                                    
                                                    if SBEV_PARAM.PREDICTION.FADING.ON
                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;
                                                        end
                                                    else
                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

%                                                     for tmp_j = 1:length(pixel_info(:,1))
%                                                         if pixel_info(tmp_j,1) ~= 0
%                                                             SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
%                                                         else
%                                                             break
%                                                         end
%                                                     end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                for tmp_j = 1:length(pixel_info(:,1))
                                                    if pixel_info(tmp_j,1) ~= 0
                                                        SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                    else
                                                        break
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY

                                                if SBEV_PARAM.PREDICTION.FADING.ON
                                                    tmp_white_vector = SBEV_PARAM.RGB_MAX - ( Collision_Probability_uint8 - 1 );
                                                    fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                    for tmp_j = 1:length(x_contour_total)
                                                        SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = ( Collision_Probability_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                    end
                                                else
                                                    for tmp_j = 1:length(x_contour_total)
                                                        SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA

                                                if SBEV_PARAM.PREDICTION.FADING.ON
                                                    tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                    fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                    for tmp_j = 1:length(x_contour_total)
                                                        SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;
                                                    end
                                                else
                                                    for tmp_j = 1:length(x_contour_total)
                                                        SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                    end
                                                end

                                            end
                                        end

                                    elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                                        for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                            for i_info = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0
                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end

                            elseif SBEV_PARAM.GRAY_IMAGE == 1
                                if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1 || SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                                    if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                        for i_ch = 1:CH_LENGTH
                                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                        else
                                                            break
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                        else
                                                            break
                                                        end
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                for tmp_j = 1:length(pixel_info(:,1))
                                                    if pixel_info(tmp_j,1) ~= 0
                                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                    else
                                                        break
                                                    end
                                                end

                                            end
                                        end
                                    end
                                end
                            end
                        end

                        %                     figure
                        %                     imshow(uint8(SBEV_out))

                        a = 1;

                    else % trajectory + hollow shape(last prediction time)

                        if index_pred == SBEV_PARAM.PREDICTION.TARGET_PRED_WINDOW/SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE
                            
                            tmp_target_y_vertex = [-State_trajectory(TRAJ_PARAM.WIDTH, end)/2, -State_trajectory(TRAJ_PARAM.WIDTH, end)/2,...
                                State_trajectory(TRAJ_PARAM.WIDTH, end)/2, State_trajectory(TRAJ_PARAM.WIDTH, end)/2, -State_trajectory(TRAJ_PARAM.WIDTH, end)/2];
                            tmp_target_x_vertex = [0, State_trajectory(TRAJ_PARAM.LENGTH, end), State_trajectory(TRAJ_PARAM.LENGTH, end), 0, 0];

                            target_y_vertex_rot = tmp_target_x_vertex.*sin(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end)) + tmp_target_y_vertex.*cos(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end));
                            target_x_vertex_rot = tmp_target_x_vertex.*cos(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end)) - tmp_target_y_vertex.*sin(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end));

                            target_y = target_y_vertex_rot + predicted_y(index_pred_detail);
                            target_x = target_x_vertex_rot + predicted_x(index_pred_detail);

                            ONLY_ONE_VERTEX_ROI_OUT_FLAG = 0;
                            TWO_VERTEX_ROI_OUT_FLAG = 0;
                            THREE_VERTEX_ROI_OUT_FLAG = 0;

                            if ~( all(target_y >= SBEV_PARAM.RANGE.Y_MIN) && all(target_y <= SBEV_PARAM.RANGE.Y_MAX) && all(target_x >= SBEV_PARAM.RANGE.X_MIN) && all(target_x <= SBEV_PARAM.RANGE.X_MAX) )

                                vertex_total = zeros(4, 4);

                                vertex_total(1, :) = target_y(1:4) >= SBEV_PARAM.RANGE.Y_MIN;
                                vertex_total(2, :) = target_y(1:4) <= SBEV_PARAM.RANGE.Y_MAX;
                                vertex_total(3, :) = target_x(1:4) >= SBEV_PARAM.RANGE.X_MIN;
                                vertex_total(4, :) = target_x(1:4) <= SBEV_PARAM.RANGE.X_MAX;

                                vertex_out_flag = all(vertex_total);

                                if nnz(vertex_out_flag) == 3 % only one vertex out of ROI
                                    ONLY_ONE_VERTEX_ROI_OUT_FLAG = 1;
                                elseif nnz(vertex_out_flag) == 2 % two vertex out of ROI
                                    TWO_VERTEX_ROI_OUT_FLAG = 1;
                                elseif nnz(vertex_out_flag) == 1 % three vertex out of ROI
                                    THREE_VERTEX_ROI_OUT_FLAG = 1;
                                end
                            end

                            if ( (min(target_y) >= SBEV_PARAM.RANGE.Y_MIN && min(target_y) <= SBEV_PARAM.RANGE.Y_MAX) || (max(target_y) >= SBEV_PARAM.RANGE.Y_MIN && max(target_y) <= SBEV_PARAM.RANGE.Y_MAX)) ...
                                    && ((min(target_x) >= SBEV_PARAM.RANGE.X_MIN && min(target_x) <= SBEV_PARAM.RANGE.X_MAX) || (max(target_x) >= SBEV_PARAM.RANGE.X_MIN && max(target_x) <= SBEV_PARAM.RANGE.X_MAX))

                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                % Find pixel of contour corresponding to predicted position
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                x_contour_total = zeros(200,1);
                                y_contour_total = zeros(200,1);

                                i_row = 1;
                                f_row = 0;

                                if ONLY_ONE_VERTEX_ROI_OUT_FLAG
                                    target_y_correction = target_y;
                                    target_x_correction = target_x;

                                    y_cross = 0;
                                    x_cross = 0;

                                    for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                        tmp_y_vertex0 = target_y(tmp_i);
                                        tmp_x_vertex0 = target_x(tmp_i);

                                        tmp_y_vertex1 = target_y(tmp_i+1);
                                        tmp_x_vertex1 = target_x(tmp_i+1);

                                        if tmp_i == 1
                                            tmp_y_vertex_1 = target_y(4);
                                            tmp_x_vertex_1 = target_x(4);
                                        else
                                            tmp_y_vertex_1 = target_y(tmp_i - 1);
                                            tmp_x_vertex_1 = target_x(tmp_i - 1);
                                        end

                                        if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                            if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next and before vertex in ROI
                                                    tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX) && ...
                                                    (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                    tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_i == 1
                                                    % tmp_i == 1
                                                    % next_start_index_origin = 2;
                                                    % next_end_index_origin = 4;
                                                    % next_start_index_correction = 3;
                                                    % next_end_index_correction = 5;
                                                    % tmp_i에 1-1, tmp_i+1에 1-2 new vertex

                                                    target_y_correction(3:5) = target_y_correction(2:4);
                                                    target_x_correction(3:5) = target_x_correction(2:4);

                                                elseif tmp_i == 2
                                                    % tmp_i == 2
                                                    % next_start_index_origin = 3;
                                                    % next_end_index_origin = 5; -> 4
                                                    % next_start_index_correction = 4;
                                                    % next_end_index_correction = 6; -> 5
                                                    % tmp_i에 2-1, tmp_i+1에 2-2 new vertex

                                                    target_y_correction(4:5) = target_y_correction(3:4);
                                                    target_x_correction(4:5) = target_x_correction(3:4);

                                                elseif tmp_i == 3
                                                    % tmp_i == 3
                                                    % next_start_index_origin = 4;
                                                    % next_end_index_origin = 6; -> 4
                                                    % next_start_index_correction = 5;
                                                    % next_end_index_correction = 7; -> 5
                                                    % tmp_i에 3-1, tmp_i+1에 3-2 new vertex

                                                    target_y_correction(5) = target_y_correction(4);
                                                    target_x_correction(5) = target_x_correction(4);

                                                elseif tmp_i == 4
                                                    % tmp_i == 4
                                                    % next_start_index_origin = 5; -> []
                                                    % next_end_index_origin = 7; -> []
                                                    % next_start_index_correction = 6; -> []
                                                    % next_end_index_correction = 8; -> []
                                                    % tmp_i에 4-1, tmp_i+1에 4-2 new vertex

                                                end

                                                % current ~ before vertex
                                                m_1 = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                tmp_base_1 = tmp_y_vertex_1 - m_1*tmp_x_vertex_1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross_1 = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross_1 = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross_1 = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross_1 = m_1*x_cross_1 + tmp_base_1;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross_1 = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross_1 = m_1*x_cross_1 + tmp_base_1;
                                                end


                                                % current ~ next vertex
                                                m_2 = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                                tmp_base_2 = tmp_y_vertex1 - m_2*tmp_x_vertex1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross_2 = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross_2 = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross_2 = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross_2 = m_2*x_cross_2 + tmp_base_2;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross_2 = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross_2 = m_2*x_cross_2 + tmp_base_2;
                                                end

                                                target_y_correction(tmp_i) = y_cross_1;
                                                target_x_correction(tmp_i) = x_cross_1;

                                                target_y_correction(tmp_i+1) = y_cross_2;
                                                target_x_correction(tmp_i+1) = x_cross_2;


                                            elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                    tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_x_vertex0 == tmp_x_vertex_1
                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = tmp_x_vertex0;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = tmp_x_vertex0;
                                                    end

                                                elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                    if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = tmp_y_vertex0;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = tmp_y_vertex0;
                                                    end
                                                else
                                                    m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                    tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = m*x_cross + tmp_base;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = m*x_cross + tmp_base;
                                                    end
                                                end


                                                if tmp_i == 1
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;

                                                    target_y_correction(5) = y_cross;
                                                    target_x_correction(5) = x_cross;
                                                else
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;
                                                end

                                            end
                                        end
                                    end

                                    for tmp_i = 1:length(target_y_correction)
                                        tmp_y_vertex0 = target_y_correction(tmp_i);
                                        tmp_x_vertex0 = target_x_correction(tmp_i);

                                        if tmp_i < 5
                                            tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                            tmp_x_vertex1 = target_x_correction(tmp_i+1);
                                        else
                                            tmp_y_vertex1 = target_y_correction(1);
                                            tmp_x_vertex1 = target_x_correction(1);
                                        end

                                        [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                        [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                        [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                        f_row = i_row + length(tmp_x_contour) - 1;
                                        x_contour_total(i_row:f_row) = tmp_x_contour;
                                        y_contour_total(i_row:f_row) = tmp_y_contour;

                                        i_row = f_row + 1;

                                    end

                                elseif TWO_VERTEX_ROI_OUT_FLAG
                                    target_y_correction = target_y;
                                    target_x_correction = target_x;

                                    y_cross = 0;
                                    x_cross = 0;

                                    for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                        tmp_y_vertex0 = target_y(tmp_i);
                                        tmp_x_vertex0 = target_x(tmp_i);

                                        tmp_y_vertex1 = target_y(tmp_i+1);
                                        tmp_x_vertex1 = target_x(tmp_i+1);

                                        if tmp_i == 1
                                            tmp_y_vertex_1 = target_y(4);
                                            tmp_x_vertex_1 = target_x(4);
                                        else
                                            tmp_y_vertex_1 = target_y(tmp_i - 1);
                                            tmp_x_vertex_1 = target_x(tmp_i - 1);
                                        end

                                        if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                            if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                                                    tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_x_vertex0 == tmp_x_vertex1
                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = tmp_x_vertex0;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = tmp_x_vertex0;
                                                    end

                                                elseif tmp_y_vertex0 == tmp_y_vertex1
                                                    if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        y_cross = tmp_y_vertex0;
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        y_cross = tmp_y_vertex0;
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    end
                                                else
                                                    m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                                    tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = m*x_cross + tmp_base;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = m*x_cross + tmp_base;
                                                    end
                                                end

                                                if tmp_i == 1
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;

                                                    target_y_correction(5) = y_cross;
                                                    target_x_correction(5) = x_cross;
                                                else
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;
                                                end


                                            elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                    tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_x_vertex0 == tmp_x_vertex_1
                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = tmp_x_vertex0;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = tmp_x_vertex0;
                                                    end

                                                elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                    if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = tmp_y_vertex0;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = tmp_y_vertex0;
                                                    end
                                                else
                                                    m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                    tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = m*x_cross + tmp_base;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = m*x_cross + tmp_base;
                                                    end
                                                end

                                                if tmp_i == 1
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;

                                                    target_y_correction(5) = y_cross;
                                                    target_x_correction(5) = x_cross;
                                                else
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;
                                                end
                                            end
                                        end
                                    end

                                    for tmp_i = 1:length(target_y_correction) - 1
                                        tmp_y_vertex0 = target_y_correction(tmp_i);
                                        tmp_x_vertex0 = target_x_correction(tmp_i);

                                        tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                        tmp_x_vertex1 = target_x_correction(tmp_i+1);

                                        [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                        [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                        [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                        f_row = i_row + length(tmp_x_contour) - 1;
                                        x_contour_total(i_row:f_row) = tmp_x_contour;
                                        y_contour_total(i_row:f_row) = tmp_y_contour;

                                        i_row = f_row + 1;

                                    end

                                elseif THREE_VERTEX_ROI_OUT_FLAG
                                    target_y_correction = target_y;
                                    target_x_correction = target_x;

                                    y_cross = 0;
                                    x_cross = 0;

                                    vertex_index_beforeCurrentNext_all_out = 0;

                                    for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                        tmp_y_vertex0 = target_y(tmp_i);
                                        tmp_x_vertex0 = target_x(tmp_i);

                                        tmp_y_vertex1 = target_y(tmp_i+1);
                                        tmp_x_vertex1 = target_x(tmp_i+1);

                                        if tmp_i == 1
                                            tmp_y_vertex_1 = target_y(4);
                                            tmp_x_vertex_1 = target_x(4);
                                        else
                                            tmp_y_vertex_1 = target_y(tmp_i - 1);
                                            tmp_x_vertex_1 = target_x(tmp_i - 1);
                                        end

                                        if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                            if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                                                    tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_x_vertex0 == tmp_x_vertex1
                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = tmp_x_vertex0;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = tmp_x_vertex0;
                                                    end

                                                elseif tmp_y_vertex0 == tmp_y_vertex1
                                                    if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        y_cross = tmp_y_vertex0;
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        y_cross = tmp_y_vertex0;
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    end
                                                else
                                                    m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                                    tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = m*x_cross + tmp_base;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = m*x_cross + tmp_base;
                                                    end
                                                end

                                                if tmp_i == 1
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;

                                                    target_y_correction(5) = y_cross;
                                                    target_x_correction(5) = x_cross;
                                                else
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;
                                                end


                                            elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                    tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_x_vertex0 == tmp_x_vertex_1
                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = tmp_x_vertex0;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = tmp_x_vertex0;
                                                    end

                                                elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                    if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = tmp_y_vertex0;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = tmp_y_vertex0;
                                                    end
                                                else
                                                    m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                    tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = m*x_cross + tmp_base;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = m*x_cross + tmp_base;
                                                    end
                                                end

                                                if tmp_i == 1
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;

                                                    target_y_correction(5) = y_cross;
                                                    target_x_correction(5) = x_cross;
                                                else
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;
                                                end

                                            else % current, next, before vertex all out of ROI
                                                vertex_index_beforeCurrentNext_all_out = tmp_i;
                                            end
                                        end
                                    end

                                    if vertex_index_beforeCurrentNext_all_out ~= 0

                                        % vertex x,y 중 하나라도 ROI에 포함되는 경우
                                        if ( target_y_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.Y_MAX ) || ...
                                                ( target_x_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.X_MAX )

                                            if vertex_index_beforeCurrentNext_all_out == 1
                                                target_y_correction(1) = target_y_correction(4);
                                                target_x_correction(1) = target_x_correction(4);
                                            else
                                                target_y_correction(vertex_index_beforeCurrentNext_all_out) = target_y_correction(vertex_index_beforeCurrentNext_all_out-1);
                                                target_x_correction(vertex_index_beforeCurrentNext_all_out) = target_x_correction(vertex_index_beforeCurrentNext_all_out-1);
                                            end

                                            % 모두 벗어나는 경우
                                        elseif ~(target_y_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                target_x_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.X_MAX)

                                            if target_y_correction(vertex_index_beforeCurrentNext_all_out) < SBEV_PARAM.RANGE.Y_MIN
                                                target_y_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.Y_MIN;
                                            elseif target_y_correction(vertex_index_beforeCurrentNext_all_out) > SBEV_PARAM.RANGE.Y_MAX
                                                target_y_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.Y_MAX;
                                            end

                                            if target_x_correction(vertex_index_beforeCurrentNext_all_out) < SBEV_PARAM.RANGE.X_MIN
                                                target_x_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.X_MIN;
                                            elseif target_x_correction(vertex_index_beforeCurrentNext_all_out) > SBEV_PARAM.RANGE.X_MAX
                                                target_x_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.X_MAX;
                                            end
                                        end
                                    end

                                    for tmp_i = 1:length(target_y_correction) - 1
                                        tmp_y_vertex0 = target_y_correction(tmp_i);
                                        tmp_x_vertex0 = target_x_correction(tmp_i);

                                        tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                        tmp_x_vertex1 = target_x_correction(tmp_i+1);

                                        [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                        [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                        [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                        f_row = i_row + length(tmp_x_contour) - 1;
                                        x_contour_total(i_row:f_row) = tmp_x_contour;
                                        y_contour_total(i_row:f_row) = tmp_y_contour;

                                        i_row = f_row + 1;

                                    end

                                else
                                    for tmp_i = 1:length(tmp_target_y_vertex) - 1
                                        tmp_y_vertex0 = target_y(tmp_i);
                                        tmp_x_vertex0 = target_x(tmp_i);

                                        tmp_y_vertex1 = target_y(tmp_i+1);
                                        tmp_x_vertex1 = target_x(tmp_i+1);

                                        [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                        [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                        [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                        f_row = i_row + length(tmp_x_contour) - 1;
                                        x_contour_total(i_row:f_row) = tmp_x_contour;
                                        y_contour_total(i_row:f_row) = tmp_y_contour;

                                        i_row = f_row + 1;
                                    end
                                end

                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                % Find pixel to fill bounding box corresponding to predicted position
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                x_contour_total(f_row+1:end) = [];
                                y_contour_total(f_row+1:end) = [];

                                pixel_info = zeros(f_row,3);
                                [sorted_x_contour_total, sorted_index] = sort(x_contour_total);
                                sorted_y_contour_total = y_contour_total(sorted_index);
                                y_i = 1000;
                                y_f = 0;
                                i_row = 0;

                                for tmp_i = 1:length(x_contour_total) - 1

                                    if sorted_x_contour_total(tmp_i) == sorted_x_contour_total(tmp_i + 1)

                                        tmp_y = sorted_y_contour_total(tmp_i);

                                        if tmp_y > y_f
                                            y_f = tmp_y;
                                        end

                                        if tmp_y < y_i
                                            y_i = tmp_y;
                                        end

                                        if tmp_i == length(x_contour_total) - 1
                                            i_row = i_row + 1;
                                            pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);

                                            if y_i > sorted_y_contour_total(tmp_i + 1)
                                                y_i = sorted_y_contour_total(tmp_i + 1);
                                            end

                                            if y_f < sorted_y_contour_total(tmp_i + 1)
                                                y_f = sorted_y_contour_total(tmp_i + 1);
                                            end

                                            pixel_info(i_row,2) = y_i;
                                            pixel_info(i_row,3) = y_f;
                                        end

                                    else
                                        i_row = i_row + 1;
                                        pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);

                                        if tmp_i == 1
                                            y_i = sorted_y_contour_total(tmp_i);
                                            y_f = y_i;
                                        elseif tmp_i == length(x_contour_total) - 1
                                            pixel_info(i_row + 1,2) = sorted_y_contour_total(tmp_i + 1);
                                            pixel_info(i_row + 1,3) = sorted_y_contour_total(tmp_i + 1);
                                        else
                                            if y_i == y_f
                                                if sorted_y_contour_total(tmp_i - 1) > sorted_y_contour_total(tmp_i)
                                                    y_i = sorted_y_contour_total(tmp_i);
                                                    y_f = sorted_y_contour_total(tmp_i - 1);
                                                elseif sorted_y_contour_total(tmp_i - 1) < sorted_y_contour_total(tmp_i)
                                                    y_i = sorted_y_contour_total(tmp_i - 1);
                                                    y_f = sorted_y_contour_total(tmp_i);
                                                else
                                                    y_i = sorted_y_contour_total(tmp_i - 1);
                                                    y_f = y_i;
                                                end
                                            else
                                                if y_i > sorted_y_contour_total(tmp_i)
                                                    y_i = sorted_y_contour_total(tmp_i);
                                                end

                                                if y_f < sorted_y_contour_total(tmp_i)
                                                    y_f = sorted_y_contour_total(tmp_i);
                                                end
                                            end

                                        end
                                        pixel_info(i_row,2) = y_i;
                                        pixel_info(i_row,3) = y_f;

                                        y_i = 1000;
                                        y_f = 0;
                                    end
                                end

                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                % Apply pixel information to DSM
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                if SBEV_PARAM.RGB_IMAGE == 1
                                    if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                            for i_ch = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end
                                                end
                                            end

                                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                                            for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                                for i_info = 1:CH_LENGTH
                                                    if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                        if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1
                                                            [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                            for tmp_j = 1:length(pixel_info(:,1))
                                                                if pixel_info(tmp_j,1) ~= 0
                                                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                                else
                                                                    break
                                                                end
                                                            end


                                                        elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0
                                                            for tmp_j = 1:length(pixel_info(:,1))
                                                                if pixel_info(tmp_j,1) ~= 0
                                                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                                else
                                                                    break
                                                                end
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    end
                                                end
                                            end
                                        end

                                    elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1

                                        % 수정본
                                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                            for i_ch = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                                        if SBEV_PARAM.PREDICTION.FADING.ON
                                                            tmp_white_vector = SBEV_PARAM.RGB_MAX - ( I_LAT_uint8 - 1 );
                                                            fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                            for tmp_j = 1:length(x_contour_total)
                                                                SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = ( I_LAT_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                            end
                                                        else
                                                            for tmp_j = 1:length(x_contour_total)
                                                                SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel

                                                        if SBEV_PARAM.PREDICTION.FADING.ON
                                                            tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                            fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                            for tmp_j = 1:length(x_contour_total)
                                                                SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;
                                                            end
                                                        else
                                                            for tmp_j = 1:length(x_contour_total)
                                                                SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

%                                                         for tmp_j = 1:length(pixel_info(:,1))
%                                                             if pixel_info(tmp_j,1) ~= 0
%                                                                 SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
%                                                             else
%                                                                 break
%                                                             end
%                                                         end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    %                                                     [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY

                                                    if SBEV_PARAM.PREDICTION.FADING.ON
                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - ( Collision_Probability_uint8 - 1 );
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = ( Collision_Probability_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                        end
                                                    else
                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA

                                                    if SBEV_PARAM.PREDICTION.FADING.ON

                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;
                                                        end
                                                    else
                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                        end
                                                    end

                                                end
                                            end

                                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                                            for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                                for i_info = 1:CH_LENGTH
                                                    if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                        if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1
                                                            [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                            for tmp_j = 1:length(pixel_info(:,1))
                                                                if pixel_info(tmp_j,1) ~= 0
                                                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                                else
                                                                    break
                                                                end
                                                            end

                                                        elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0
                                                            for tmp_j = 1:length(pixel_info(:,1))
                                                                if pixel_info(tmp_j,1) ~= 0
                                                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                                else
                                                                    break
                                                                end
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end

                                elseif SBEV_PARAM.GRAY_IMAGE == 1
                                    if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1 || SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                            for i_ch = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end

                                                end
                                            end
                                        end
                                    end
                                end
                            end

                            %                     figure
                            %                     imshow(uint8(SBEV_out))

                            a = 1;

                        else
                            target_y = predicted_y(index_pred_detail);
                            target_x = predicted_x(index_pred_detail);

                            if target_y >= SBEV_PARAM.RANGE.Y_MIN && target_y <= SBEV_PARAM.RANGE.Y_MAX ...
                                    && target_x >= SBEV_PARAM.RANGE.X_MIN && target_x <= SBEV_PARAM.RANGE.X_MAX

                                [~,Image_Position_X] = min(abs(target_x - SBEV_PARAM.RANGE.X_RANGE));
                                [~,Image_Position_Y] = min(abs(target_y - SBEV_PARAM.RANGE.Y_RANGE));

                                if SBEV_PARAM.RGB_IMAGE == 1
                                    if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0
                                            for i_ch = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                                    if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                                    elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                    elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                                        if index_traj ~= length(State_trajectory(1,:))
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                        elseif index_traj == length(State_trajectory(1,:))
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                end
                                            end

                                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                                            for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                                for i_info = 1:CH_LENGTH
                                                    if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                                        if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                                        elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                        elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                                            if index_traj ~= length(State_trajectory(1,:))
                                                                SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            elseif index_traj == length(State_trajectory(1,:))
                                                                SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                        SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                    end
                                                end
                                            end
                                        end

                                    elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0
                                            for i_ch = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.PREDICTION.TRAJECTORY_THREAT
                                                        if SBEV_PARAM.PREDICTION.FADING.ON
                                                            tmp_white_vector = SBEV_PARAM.RGB_MAX - ( I_LAT_uint8 - 1 );
                                                            fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = ( I_LAT_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                        else
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8 - 1;
                                                        end
                                                    else
                                                        if SBEV_PARAM.PREDICTION.FADING.ON
                                                            tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                            fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;
                                                        else
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY

                                                    if SBEV_PARAM.PREDICTION.FADING.ON
                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - ( Collision_Probability_uint8 - 1 );
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = ( Collision_Probability_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                    else
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA

                                                    if SBEV_PARAM.PREDICTION.FADING.ON
                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;                                                        
                                                    else
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                    end

                                                end
                                            end

                                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1
                                            for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                                for i_info = 1:CH_LENGTH
                                                    if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                                        if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                                        elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                        elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                                            if index_traj ~= length(State_trajectory(1,:))
                                                                SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            elseif index_traj == length(State_trajectory(1,:))
                                                                SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                        SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                    end
                                                end
                                            end
                                        end
                                    end

                                elseif SBEV_PARAM.GRAY_IMAGE == 1
                                    if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1

                                        for i_ch = 1:CH_LENGTH
                                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                                if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                                elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                                    if index_traj ~= length(State_trajectory(1,:))
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                    elseif index_traj == length(State_trajectory(1,:))
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                            end
                                        end

                                    elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                                        for i_ch = 1:CH_LENGTH
                                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                                if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;

                                                elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX - (I_LAT_uint8-1);

                                                elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                                    if index_traj ~= length(State_trajectory(1,:))
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                    elseif index_traj == length(State_trajectory(1,:))
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX - (I_LAT_uint8-1);
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX - (I_LAT_uint8-1);
                                            end
                                        end
                                    end
                                end
                            end

                            %                     figure
                            %                     imshow(uint8(SBEV_out))
                        end

                    end

                end


            else % overlap 미허용

                % forward 방향으로 자차와 겹치는 예측 시점 존재하는지 확인

                % if 존재
                %   안 겹치기 시작하는 역순으로 DSM에 반영

                % else % 미존재
                %   전체 예측 시점을 역순으로 DSM에 반영

                % end


                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % prediction의 forward 방향으로 자차와 겹치는 예측 시점 확인
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                overlap_btw_ego_pred_flag = 0;
                prediction_time_step_at_overlap = 0;

                ego_vertex = [-EGO_VEHICLE.EGO_WIDTH/2 - ROI_margin2ego, -EGO_VEHICLE.EGO_WIDTH/2 - ROI_margin2ego, EGO_VEHICLE.EGO_WIDTH/2 + ROI_margin2ego, EGO_VEHICLE.EGO_WIDTH/2 + ROI_margin2ego, -EGO_VEHICLE.EGO_WIDTH/2 - ROI_margin2ego;
                    -EGO_VEHICLE.EGO_LENGTH, 0, 0, -EGO_VEHICLE.EGO_LENGTH, -EGO_VEHICLE.EGO_LENGTH];

                tmp_target_y_vertex = [-State_trajectory(TRAJ_PARAM.WIDTH, end)/2, -State_trajectory(TRAJ_PARAM.WIDTH, end)/2,...
                    State_trajectory(TRAJ_PARAM.WIDTH, end)/2, State_trajectory(TRAJ_PARAM.WIDTH, end)/2, -State_trajectory(TRAJ_PARAM.WIDTH, end)/2];
                tmp_target_x_vertex = [0, State_trajectory(TRAJ_PARAM.LENGTH, end), State_trajectory(TRAJ_PARAM.LENGTH, end), 0, 0];


                for index_pred = 1:SBEV_PARAM.PREDICTION.TARGET_PRED_WINDOW/SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE
                    index_pred_detail = round(index_pred*SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);

                    target_y_vertex_rot = tmp_target_x_vertex.*sin( Target_X_pred(TRACKING.HEADING_ANGLE, 1, index_pred_detail) ) + tmp_target_y_vertex.*cos( Target_X_pred(TRACKING.HEADING_ANGLE, 1, index_pred_detail) );
                    target_x_vertex_rot = tmp_target_x_vertex.*cos( Target_X_pred(TRACKING.HEADING_ANGLE, 1, index_pred_detail) ) - tmp_target_y_vertex.*sin( Target_X_pred(TRACKING.HEADING_ANGLE, 1, index_pred_detail) );

                    target_y = target_y_vertex_rot + Target_X_pred(TRACKING.REL_POS_Y, 1, index_pred_detail);
                    target_x = target_x_vertex_rot + Target_X_pred(TRACKING.REL_POS_X, 1, index_pred_detail);

                    target_vertex = [target_y; target_x];

                    in_egoVehicle = inpolygon(ego_vertex(2, :), ego_vertex(1, :), target_vertex(2, :), target_vertex(1, :));
                    in_targetVehicle = inpolygon(target_vertex(2, :), target_vertex(1, :), ego_vertex(2, :), ego_vertex(1, :));

                    if sum(in_egoVehicle, 'all') ~= 0 || sum(in_targetVehicle, 'all') ~= 0
                        overlap_btw_ego_pred_flag = 1;
                        prediction_time_step_at_overlap = index_pred; % 1, 2, 3, 4, 5
                        break
                    end
                end


                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % 겹치는 예측 시점 존재 유무에 따라 DSM 에 반영
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if overlap_btw_ego_pred_flag == 1 && prediction_time_step_at_overlap ~= 0 % 자차와 겹치는 예측 시점 존재
                    last_nonOverlap_prediction_time_step = prediction_time_step_at_overlap;

                else % 자차와 겹치는 예측 시점 미존재
                    last_nonOverlap_prediction_time_step = SBEV_PARAM.PREDICTION.TARGET_PRED_WINDOW / SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE;

                end


                % 세밀하게 자차와 겹치는 시점 탐색
                if last_nonOverlap_prediction_time_step ~= 0

                    last_pred_exist_between_time_step_flag = 0;
                    overlap_btw_ego_pred_before_detail_flag = 0;
                    prediction_time_step_detail_at_overlap = 0;

                    for index_pred_detail = round(last_nonOverlap_prediction_time_step*SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME) : -1 : round((last_nonOverlap_prediction_time_step-1)*SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME)+1

                        overlap_btw_ego_pred_detail_flag = 0;

                        target_y_vertex_rot = tmp_target_x_vertex.*sin( Target_X_pred(TRACKING.HEADING_ANGLE, 1, index_pred_detail) ) + tmp_target_y_vertex.*cos( Target_X_pred(TRACKING.HEADING_ANGLE, 1, index_pred_detail) );
                        target_x_vertex_rot = tmp_target_x_vertex.*cos( Target_X_pred(TRACKING.HEADING_ANGLE, 1, index_pred_detail) ) - tmp_target_y_vertex.*sin( Target_X_pred(TRACKING.HEADING_ANGLE, 1, index_pred_detail) );

                        target_y = target_y_vertex_rot + Target_X_pred(TRACKING.REL_POS_Y, 1, index_pred_detail);
                        target_x = target_x_vertex_rot + Target_X_pred(TRACKING.REL_POS_X, 1, index_pred_detail);

                        target_vertex = [target_y; target_x];

                        in_egoVehicle = inpolygon(ego_vertex(2, :), ego_vertex(1, :), target_vertex(2, :), target_vertex(1, :));
                        in_targetVehicle = inpolygon(target_vertex(2, :), target_vertex(1, :), ego_vertex(2, :), ego_vertex(1, :));

                        if sum(in_egoVehicle, 'all') ~= 0 || sum(in_targetVehicle, 'all') ~= 0
                            overlap_btw_ego_pred_detail_flag = 1;
                            overlap_btw_ego_pred_before_detail_flag = overlap_btw_ego_pred_detail_flag;
                            prediction_time_step_detail_at_overlap = index_pred_detail;

                        else
                            if overlap_btw_ego_pred_detail_flag == 0 && overlap_btw_ego_pred_before_detail_flag ~= 0

                                last_pred_exist_between_time_step_flag = 1;
                                break
                            end                            
                        end
                    end

                    % 첫 충돌 발생한 index_pred ~ index_pred-1 사이를 잘게 쪼개 그 구간에서 충돌 처음 나는 시점을 찾았는데 모두 충돌일 때
                    if last_pred_exist_between_time_step_flag == 0 && overlap_btw_ego_pred_detail_flag == 1 && overlap_btw_ego_pred_before_detail_flag == 1
                        last_pred_exist_between_time_step_flag = 1;
                    end
                end

                
                

                % 처음 겹치는 예측 시점 plot

                % 나머지 예측 시점 plot

                for index_pred = last_nonOverlap_prediction_time_step : -1 : 1

                    if SBEV_PARAM.PREDICTION.ALL_SHAPE_FLAG % hollow shape for 자차와 겹치지 않는 prediction time까지 그림

                        if index_pred == last_nonOverlap_prediction_time_step

                            if last_pred_exist_between_time_step_flag == 0
                                if index_pred == 1
                                    continue
                                else
                                    index_pred_detail = round(index_pred*SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);

                                    target_y = target_y_vertex_rot + Target_X_pred(TRACKING.REL_POS_Y, 1, index_pred_detail);
                                    target_x = target_x_vertex_rot + Target_X_pred(TRACKING.REL_POS_X, 1, index_pred_detail);
                                end

                            else
                                target_y = target_y_vertex_rot + Target_X_pred(TRACKING.REL_POS_Y, 1, prediction_time_step_detail_at_overlap);
                                target_x = target_x_vertex_rot + Target_X_pred(TRACKING.REL_POS_X, 1, prediction_time_step_detail_at_overlap);
                            end

                        else
                            index_pred_detail = round(index_pred*SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);

                            target_y = target_y_vertex_rot + Target_X_pred(TRACKING.REL_POS_Y, 1, index_pred_detail);
                            target_x = target_x_vertex_rot + Target_X_pred(TRACKING.REL_POS_X, 1, index_pred_detail);
                        end
                        

                        ONLY_ONE_VERTEX_ROI_OUT_FLAG = 0;
                        TWO_VERTEX_ROI_OUT_FLAG = 0;
                        THREE_VERTEX_ROI_OUT_FLAG = 0;

                        if ~( all(target_y >= SBEV_PARAM.RANGE.Y_MIN) && all(target_y <= SBEV_PARAM.RANGE.Y_MAX) && all(target_x >= SBEV_PARAM.RANGE.X_MIN) && all(target_x <= SBEV_PARAM.RANGE.X_MAX) )

                            vertex_total = zeros(4, 4);

                            vertex_total(1, :) = target_y(1:4) >= SBEV_PARAM.RANGE.Y_MIN;
                            vertex_total(2, :) = target_y(1:4) <= SBEV_PARAM.RANGE.Y_MAX;
                            vertex_total(3, :) = target_x(1:4) >= SBEV_PARAM.RANGE.X_MIN;
                            vertex_total(4, :) = target_x(1:4) <= SBEV_PARAM.RANGE.X_MAX;

                            vertex_out_flag = all(vertex_total);

                            if nnz(vertex_out_flag) == 3 % only one vertex out of ROI
                                ONLY_ONE_VERTEX_ROI_OUT_FLAG = 1;
                            elseif nnz(vertex_out_flag) == 2 % two vertex out of ROI
                                TWO_VERTEX_ROI_OUT_FLAG = 1;
                            elseif nnz(vertex_out_flag) == 1 % three vertex out of ROI
                                THREE_VERTEX_ROI_OUT_FLAG = 1;
                            end
                        end

                        if ( (min(target_y) >= SBEV_PARAM.RANGE.Y_MIN && min(target_y) <= SBEV_PARAM.RANGE.Y_MAX) || (max(target_y) >= SBEV_PARAM.RANGE.Y_MIN && max(target_y) <= SBEV_PARAM.RANGE.Y_MAX)) ...
                                && ((min(target_x) >= SBEV_PARAM.RANGE.X_MIN && min(target_x) <= SBEV_PARAM.RANGE.X_MAX) || (max(target_x) >= SBEV_PARAM.RANGE.X_MIN && max(target_x) <= SBEV_PARAM.RANGE.X_MAX))

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Find pixel of contour corresponding to predicted position
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            x_contour_total = zeros(200,1);
                            y_contour_total = zeros(200,1);

                            i_row = 1;
                            f_row = 0;

                            if ONLY_ONE_VERTEX_ROI_OUT_FLAG
                                target_y_correction = target_y;
                                target_x_correction = target_x;

                                y_cross = 0;
                                x_cross = 0;

                                for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                    tmp_y_vertex0 = target_y(tmp_i);
                                    tmp_x_vertex0 = target_x(tmp_i);

                                    tmp_y_vertex1 = target_y(tmp_i+1);
                                    tmp_x_vertex1 = target_x(tmp_i+1);

                                    if tmp_i == 1
                                        tmp_y_vertex_1 = target_y(4);
                                        tmp_x_vertex_1 = target_x(4);
                                    else
                                        tmp_y_vertex_1 = target_y(tmp_i - 1);
                                        tmp_x_vertex_1 = target_x(tmp_i - 1);
                                    end

                                    if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                            tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                        if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next and before vertex in ROI
                                                tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX) && ...
                                                (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_i == 1
                                                % tmp_i == 1
                                                % next_start_index_origin = 2;
                                                % next_end_index_origin = 4;
                                                % next_start_index_correction = 3;
                                                % next_end_index_correction = 5;
                                                % tmp_i에 1-1, tmp_i+1에 1-2 new vertex

                                                target_y_correction(3:5) = target_y_correction(2:4);
                                                target_x_correction(3:5) = target_x_correction(2:4);

                                            elseif tmp_i == 2
                                                % tmp_i == 2
                                                % next_start_index_origin = 3;
                                                % next_end_index_origin = 5; -> 4
                                                % next_start_index_correction = 4;
                                                % next_end_index_correction = 6; -> 5
                                                % tmp_i에 2-1, tmp_i+1에 2-2 new vertex

                                                target_y_correction(4:5) = target_y_correction(3:4);
                                                target_x_correction(4:5) = target_x_correction(3:4);

                                            elseif tmp_i == 3
                                                % tmp_i == 3
                                                % next_start_index_origin = 4;
                                                % next_end_index_origin = 6; -> 4
                                                % next_start_index_correction = 5;
                                                % next_end_index_correction = 7; -> 5
                                                % tmp_i에 3-1, tmp_i+1에 3-2 new vertex

                                                target_y_correction(5) = target_y_correction(4);
                                                target_x_correction(5) = target_x_correction(4);

                                            elseif tmp_i == 4
                                                % tmp_i == 4
                                                % next_start_index_origin = 5; -> []
                                                % next_end_index_origin = 7; -> []
                                                % next_start_index_correction = 6; -> []
                                                % next_end_index_correction = 8; -> []
                                                % tmp_i에 4-1, tmp_i+1에 4-2 new vertex

                                            end

                                            % current ~ before vertex
                                            m_1 = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                            tmp_base_1 = tmp_y_vertex_1 - m_1*tmp_x_vertex_1;

                                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                y_cross_1 = SBEV_PARAM.RANGE.Y_MIN;
                                                x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                y_cross_1 = SBEV_PARAM.RANGE.Y_MAX;
                                                x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                                            elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                x_cross_1 = SBEV_PARAM.RANGE.X_MIN;
                                                y_cross_1 = m_1*x_cross_1 + tmp_base_1;

                                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                x_cross_1 = SBEV_PARAM.RANGE.X_MAX;
                                                y_cross_1 = m_1*x_cross_1 + tmp_base_1;
                                            end


                                            % current ~ next vertex
                                            m_2 = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                            tmp_base_2 = tmp_y_vertex1 - m_2*tmp_x_vertex1;

                                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                y_cross_2 = SBEV_PARAM.RANGE.Y_MIN;
                                                x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                y_cross_2 = SBEV_PARAM.RANGE.Y_MAX;
                                                x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                                            elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                x_cross_2 = SBEV_PARAM.RANGE.X_MIN;
                                                y_cross_2 = m_2*x_cross_2 + tmp_base_2;

                                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                x_cross_2 = SBEV_PARAM.RANGE.X_MAX;
                                                y_cross_2 = m_2*x_cross_2 + tmp_base_2;
                                            end

                                            target_y_correction(tmp_i) = y_cross_1;
                                            target_x_correction(tmp_i) = x_cross_1;

                                            target_y_correction(tmp_i+1) = y_cross_2;
                                            target_x_correction(tmp_i+1) = x_cross_2;


                                        elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_x_vertex0 == tmp_x_vertex_1
                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = tmp_x_vertex0;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = tmp_x_vertex0;
                                                end

                                            elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = tmp_y_vertex0;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = tmp_y_vertex0;
                                                end
                                            else
                                                m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = m*x_cross + tmp_base;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = m*x_cross + tmp_base;
                                                end
                                            end


                                            if tmp_i == 1
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;

                                                target_y_correction(5) = y_cross;
                                                target_x_correction(5) = x_cross;
                                            else
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;
                                            end

                                        end
                                    end
                                end

                                for tmp_i = 1:length(target_y_correction)
                                    tmp_y_vertex0 = target_y_correction(tmp_i);
                                    tmp_x_vertex0 = target_x_correction(tmp_i);

                                    if tmp_i < 5
                                        tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                        tmp_x_vertex1 = target_x_correction(tmp_i+1);
                                    else
                                        tmp_y_vertex1 = target_y_correction(1);
                                        tmp_x_vertex1 = target_x_correction(1);
                                    end

                                    [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                    [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                    [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                    f_row = i_row + length(tmp_x_contour) - 1;
                                    x_contour_total(i_row:f_row) = tmp_x_contour;
                                    y_contour_total(i_row:f_row) = tmp_y_contour;

                                    i_row = f_row + 1;

                                end

                            elseif TWO_VERTEX_ROI_OUT_FLAG
                                target_y_correction = target_y;
                                target_x_correction = target_x;

                                y_cross = 0;
                                x_cross = 0;

                                for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                    tmp_y_vertex0 = target_y(tmp_i);
                                    tmp_x_vertex0 = target_x(tmp_i);

                                    tmp_y_vertex1 = target_y(tmp_i+1);
                                    tmp_x_vertex1 = target_x(tmp_i+1);

                                    if tmp_i == 1
                                        tmp_y_vertex_1 = target_y(4);
                                        tmp_x_vertex_1 = target_x(4);
                                    else
                                        tmp_y_vertex_1 = target_y(tmp_i - 1);
                                        tmp_x_vertex_1 = target_x(tmp_i - 1);
                                    end

                                    if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                            tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                        if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                                                tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_x_vertex0 == tmp_x_vertex1
                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = tmp_x_vertex0;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = tmp_x_vertex0;
                                                end

                                            elseif tmp_y_vertex0 == tmp_y_vertex1
                                                if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    y_cross = tmp_y_vertex0;
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    y_cross = tmp_y_vertex0;
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                end
                                            else
                                                m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                                tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = m*x_cross + tmp_base;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = m*x_cross + tmp_base;
                                                end
                                            end

                                            if tmp_i == 1
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;

                                                target_y_correction(5) = y_cross;
                                                target_x_correction(5) = x_cross;
                                            else
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;
                                            end


                                        elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_x_vertex0 == tmp_x_vertex_1
                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = tmp_x_vertex0;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = tmp_x_vertex0;
                                                end

                                            elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = tmp_y_vertex0;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = tmp_y_vertex0;
                                                end
                                            else
                                                m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = m*x_cross + tmp_base;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = m*x_cross + tmp_base;
                                                end
                                            end

                                            if tmp_i == 1
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;

                                                target_y_correction(5) = y_cross;
                                                target_x_correction(5) = x_cross;
                                            else
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;
                                            end
                                        end
                                    end
                                end

                                for tmp_i = 1:length(target_y_correction) - 1
                                    tmp_y_vertex0 = target_y_correction(tmp_i);
                                    tmp_x_vertex0 = target_x_correction(tmp_i);

                                    tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                    tmp_x_vertex1 = target_x_correction(tmp_i+1);

                                    [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                    [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                    [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                    f_row = i_row + length(tmp_x_contour) - 1;
                                    x_contour_total(i_row:f_row) = tmp_x_contour;
                                    y_contour_total(i_row:f_row) = tmp_y_contour;

                                    i_row = f_row + 1;

                                end

                            elseif THREE_VERTEX_ROI_OUT_FLAG
                                target_y_correction = target_y;
                                target_x_correction = target_x;

                                y_cross = 0;
                                x_cross = 0;

                                vertex_index_beforeCurrentNext_all_out = 0;

                                for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                    tmp_y_vertex0 = target_y(tmp_i);
                                    tmp_x_vertex0 = target_x(tmp_i);

                                    tmp_y_vertex1 = target_y(tmp_i+1);
                                    tmp_x_vertex1 = target_x(tmp_i+1);

                                    if tmp_i == 1
                                        tmp_y_vertex_1 = target_y(4);
                                        tmp_x_vertex_1 = target_x(4);
                                    else
                                        tmp_y_vertex_1 = target_y(tmp_i - 1);
                                        tmp_x_vertex_1 = target_x(tmp_i - 1);
                                    end

                                    if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                            tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                        if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                                                tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_x_vertex0 == tmp_x_vertex1
                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = tmp_x_vertex0;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = tmp_x_vertex0;
                                                end

                                            elseif tmp_y_vertex0 == tmp_y_vertex1
                                                if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    y_cross = tmp_y_vertex0;
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    y_cross = tmp_y_vertex0;
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                end
                                            else
                                                m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                                tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = m*x_cross + tmp_base;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = m*x_cross + tmp_base;
                                                end
                                            end

                                            if tmp_i == 1
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;

                                                target_y_correction(5) = y_cross;
                                                target_x_correction(5) = x_cross;
                                            else
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;
                                            end


                                        elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                            if tmp_x_vertex0 == tmp_x_vertex_1
                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = tmp_x_vertex0;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = tmp_x_vertex0;
                                                end

                                            elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = tmp_y_vertex0;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = tmp_y_vertex0;
                                                end
                                            else
                                                m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross = (y_cross - tmp_base)/m;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross = m*x_cross + tmp_base;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross = m*x_cross + tmp_base;
                                                end
                                            end

                                            if tmp_i == 1
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;

                                                target_y_correction(5) = y_cross;
                                                target_x_correction(5) = x_cross;
                                            else
                                                target_y_correction(tmp_i) = y_cross;
                                                target_x_correction(tmp_i) = x_cross;
                                            end

                                        else % current, next, before vertex all out of ROI
                                            vertex_index_beforeCurrentNext_all_out = tmp_i;
                                        end
                                    end
                                end

                                if vertex_index_beforeCurrentNext_all_out ~= 0

                                    % vertex x,y 중 하나라도 ROI에 포함되는 경우
                                    if ( target_y_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.Y_MAX ) || ...
                                            ( target_x_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.X_MAX )

                                        if vertex_index_beforeCurrentNext_all_out == 1
                                            target_y_correction(1) = target_y_correction(4);
                                            target_x_correction(1) = target_x_correction(4);
                                        else
                                            target_y_correction(vertex_index_beforeCurrentNext_all_out) = target_y_correction(vertex_index_beforeCurrentNext_all_out-1);
                                            target_x_correction(vertex_index_beforeCurrentNext_all_out) = target_x_correction(vertex_index_beforeCurrentNext_all_out-1);
                                        end

                                        % 모두 벗어나는 경우
                                    elseif ~(target_y_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.Y_MAX &&...
                                            target_x_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.X_MAX)

                                        if target_y_correction(vertex_index_beforeCurrentNext_all_out) < SBEV_PARAM.RANGE.Y_MIN
                                            target_y_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.Y_MIN;
                                        elseif target_y_correction(vertex_index_beforeCurrentNext_all_out) > SBEV_PARAM.RANGE.Y_MAX
                                            target_y_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.Y_MAX;
                                        end

                                        if target_x_correction(vertex_index_beforeCurrentNext_all_out) < SBEV_PARAM.RANGE.X_MIN
                                            target_x_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.X_MIN;
                                        elseif target_x_correction(vertex_index_beforeCurrentNext_all_out) > SBEV_PARAM.RANGE.X_MAX
                                            target_x_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.X_MAX;
                                        end
                                    end
                                end

                                for tmp_i = 1:length(target_y_correction) - 1
                                    tmp_y_vertex0 = target_y_correction(tmp_i);
                                    tmp_x_vertex0 = target_x_correction(tmp_i);

                                    tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                    tmp_x_vertex1 = target_x_correction(tmp_i+1);

                                    [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                    [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                    [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                    f_row = i_row + length(tmp_x_contour) - 1;
                                    x_contour_total(i_row:f_row) = tmp_x_contour;
                                    y_contour_total(i_row:f_row) = tmp_y_contour;

                                    i_row = f_row + 1;

                                end

                            else
                                for tmp_i = 1:length(tmp_target_y_vertex) - 1
                                    tmp_y_vertex0 = target_y(tmp_i);
                                    tmp_x_vertex0 = target_x(tmp_i);

                                    tmp_y_vertex1 = target_y(tmp_i+1);
                                    tmp_x_vertex1 = target_x(tmp_i+1);

                                    [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                    [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                    [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                    [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                    f_row = i_row + length(tmp_x_contour) - 1;
                                    x_contour_total(i_row:f_row) = tmp_x_contour;
                                    y_contour_total(i_row:f_row) = tmp_y_contour;

                                    i_row = f_row + 1;
                                end
                            end

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Find pixel to fill bounding box corresponding to predicted position
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            x_contour_total(f_row+1:end) = [];
                            y_contour_total(f_row+1:end) = [];

                            pixel_info = zeros(f_row,3);
                            [sorted_x_contour_total, sorted_index] = sort(x_contour_total);
                            sorted_y_contour_total = y_contour_total(sorted_index);
                            y_i = 1000;
                            y_f = 0;
                            i_row = 0;

                            for tmp_i = 1:length(x_contour_total) - 1

                                if sorted_x_contour_total(tmp_i) == sorted_x_contour_total(tmp_i + 1)

                                    tmp_y = sorted_y_contour_total(tmp_i);

                                    if tmp_y > y_f
                                        y_f = tmp_y;
                                    end

                                    if tmp_y < y_i
                                        y_i = tmp_y;
                                    end

                                    if tmp_i == length(x_contour_total) - 1
                                        i_row = i_row + 1;
                                        pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);

                                        if y_i > sorted_y_contour_total(tmp_i + 1)
                                            y_i = sorted_y_contour_total(tmp_i + 1);
                                        end

                                        if y_f < sorted_y_contour_total(tmp_i + 1)
                                            y_f = sorted_y_contour_total(tmp_i + 1);
                                        end

                                        pixel_info(i_row,2) = y_i;
                                        pixel_info(i_row,3) = y_f;
                                    end

                                else
                                    i_row = i_row + 1;
                                    pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);

                                    if tmp_i == 1
                                        y_i = sorted_y_contour_total(tmp_i);
                                        y_f = y_i;
                                    elseif tmp_i == length(x_contour_total) - 1
                                        pixel_info(i_row + 1,2) = sorted_y_contour_total(tmp_i + 1);
                                        pixel_info(i_row + 1,3) = sorted_y_contour_total(tmp_i + 1);
                                    else
                                        if y_i == y_f
                                            if sorted_y_contour_total(tmp_i - 1) > sorted_y_contour_total(tmp_i)
                                                y_i = sorted_y_contour_total(tmp_i);
                                                y_f = sorted_y_contour_total(tmp_i - 1);
                                            elseif sorted_y_contour_total(tmp_i - 1) < sorted_y_contour_total(tmp_i)
                                                y_i = sorted_y_contour_total(tmp_i - 1);
                                                y_f = sorted_y_contour_total(tmp_i);
                                            else
                                                y_i = sorted_y_contour_total(tmp_i - 1);
                                                y_f = y_i;
                                            end
                                        else
                                            if y_i > sorted_y_contour_total(tmp_i)
                                                y_i = sorted_y_contour_total(tmp_i);
                                            end

                                            if y_f < sorted_y_contour_total(tmp_i)
                                                y_f = sorted_y_contour_total(tmp_i);
                                            end
                                        end

                                    end
                                    pixel_info(i_row,2) = y_i;
                                    pixel_info(i_row,3) = y_f;

                                    y_i = 1000;
                                    y_f = 0;
                                end
                            end

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Apply pixel information to DSM
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            if SBEV_PARAM.RGB_IMAGE == 1
                                if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                                    if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                        for i_ch = 1:CH_LENGTH
                                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                        else
                                                            break
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                        else
                                                            break
                                                        end
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                for tmp_j = 1:length(pixel_info(:,1))
                                                    if pixel_info(tmp_j,1) ~= 0
                                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                    else
                                                        break
                                                    end
                                                end
                                            end
                                        end

                                    elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                                        for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                            for i_info = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end


                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0
                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end

                                                end
                                            end
                                        end
                                    end

                                elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1

                                    % 수정본
                                    if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                        for i_ch = 1:CH_LENGTH
                                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                                    if SBEV_PARAM.PREDICTION.FADING.ON
                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - ( I_LAT_uint8 - 1 );
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = ( I_LAT_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                        end
                                                    else
                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel

                                                    if SBEV_PARAM.PREDICTION.FADING.ON
                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;
                                                        end
                                                    else
                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

%                                                     for tmp_j = 1:length(pixel_info(:,1))
%                                                         if pixel_info(tmp_j,1) ~= 0
%                                                             SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
%                                                         else
%                                                             break
%                                                         end
%                                                     end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                for tmp_j = 1:length(pixel_info(:,1))
                                                    if pixel_info(tmp_j,1) ~= 0
                                                        SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                    else
                                                        break
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY

                                                if SBEV_PARAM.PREDICTION.FADING.ON
                                                    tmp_white_vector = SBEV_PARAM.RGB_MAX - ( Collision_Probability_uint8 - 1 );
                                                    fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                    for tmp_j = 1:length(x_contour_total)
                                                        SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = ( Collision_Probability_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                    end
                                                else
                                                    for tmp_j = 1:length(x_contour_total)
                                                        SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA

                                                if SBEV_PARAM.PREDICTION.FADING.ON
                                                    tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                    fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                    for tmp_j = 1:length(x_contour_total)
                                                        SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;
                                                    end
                                                else
                                                    for tmp_j = 1:length(x_contour_total)
                                                        SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                    end
                                                end

                                            end
                                        end

                                    elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                                        for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                            for i_info = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0
                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end

                            elseif SBEV_PARAM.GRAY_IMAGE == 1
                                if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1 || SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                                    if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                        for i_ch = 1:CH_LENGTH
                                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                        else
                                                            break
                                                        end
                                                    end

                                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                        else
                                                            break
                                                        end
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                for tmp_j = 1:length(pixel_info(:,1))
                                                    if pixel_info(tmp_j,1) ~= 0
                                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                    else
                                                        break
                                                    end
                                                end

                                            end
                                        end
                                    end
                                end
                            end
                        end

                        %                     figure
                        %                     imshow(uint8(SBEV_out))

                        a = 1;


                    else % trajectory + hollow shape(자차와 겹치지 않는 prediction time)

                        if index_pred == last_nonOverlap_prediction_time_step % shape

                            if last_pred_exist_between_time_step_flag == 0
                                if index_pred == 1
                                    continue
                                else
                                    index_pred_detail = round(index_pred*SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);

                                    target_y = target_y_vertex_rot + Target_X_pred(TRACKING.REL_POS_Y, 1, index_pred_detail);
                                    target_x = target_x_vertex_rot + Target_X_pred(TRACKING.REL_POS_X, 1, index_pred_detail);
                                end

                            else
                                target_y = target_y_vertex_rot + Target_X_pred(TRACKING.REL_POS_Y, 1, prediction_time_step_detail_at_overlap);
                                target_x = target_x_vertex_rot + Target_X_pred(TRACKING.REL_POS_X, 1, prediction_time_step_detail_at_overlap);
                            end

                            
                            ONLY_ONE_VERTEX_ROI_OUT_FLAG = 0;
                            TWO_VERTEX_ROI_OUT_FLAG = 0;
                            THREE_VERTEX_ROI_OUT_FLAG = 0;

                            if ~( all(target_y >= SBEV_PARAM.RANGE.Y_MIN) && all(target_y <= SBEV_PARAM.RANGE.Y_MAX) && all(target_x >= SBEV_PARAM.RANGE.X_MIN) && all(target_x <= SBEV_PARAM.RANGE.X_MAX) )

                                vertex_total = zeros(4, 4);

                                vertex_total(1, :) = target_y(1:4) >= SBEV_PARAM.RANGE.Y_MIN;
                                vertex_total(2, :) = target_y(1:4) <= SBEV_PARAM.RANGE.Y_MAX;
                                vertex_total(3, :) = target_x(1:4) >= SBEV_PARAM.RANGE.X_MIN;
                                vertex_total(4, :) = target_x(1:4) <= SBEV_PARAM.RANGE.X_MAX;

                                vertex_out_flag = all(vertex_total);

                                if nnz(vertex_out_flag) == 3 % only one vertex out of ROI
                                    ONLY_ONE_VERTEX_ROI_OUT_FLAG = 1;
                                elseif nnz(vertex_out_flag) == 2 % two vertex out of ROI
                                    TWO_VERTEX_ROI_OUT_FLAG = 1;
                                elseif nnz(vertex_out_flag) == 1 % three vertex out of ROI
                                    THREE_VERTEX_ROI_OUT_FLAG = 1;
                                end
                            end

                            if ( (min(target_y) >= SBEV_PARAM.RANGE.Y_MIN && min(target_y) <= SBEV_PARAM.RANGE.Y_MAX) || (max(target_y) >= SBEV_PARAM.RANGE.Y_MIN && max(target_y) <= SBEV_PARAM.RANGE.Y_MAX)) ...
                                    && ((min(target_x) >= SBEV_PARAM.RANGE.X_MIN && min(target_x) <= SBEV_PARAM.RANGE.X_MAX) || (max(target_x) >= SBEV_PARAM.RANGE.X_MIN && max(target_x) <= SBEV_PARAM.RANGE.X_MAX))

                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                % Find pixel of contour corresponding to predicted position
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                x_contour_total = zeros(200,1);
                                y_contour_total = zeros(200,1);

                                i_row = 1;
                                f_row = 0;

                                if ONLY_ONE_VERTEX_ROI_OUT_FLAG
                                    target_y_correction = target_y;
                                    target_x_correction = target_x;

                                    y_cross = 0;
                                    x_cross = 0;

                                    for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                        tmp_y_vertex0 = target_y(tmp_i);
                                        tmp_x_vertex0 = target_x(tmp_i);

                                        tmp_y_vertex1 = target_y(tmp_i+1);
                                        tmp_x_vertex1 = target_x(tmp_i+1);

                                        if tmp_i == 1
                                            tmp_y_vertex_1 = target_y(4);
                                            tmp_x_vertex_1 = target_x(4);
                                        else
                                            tmp_y_vertex_1 = target_y(tmp_i - 1);
                                            tmp_x_vertex_1 = target_x(tmp_i - 1);
                                        end

                                        if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                            if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next and before vertex in ROI
                                                    tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX) && ...
                                                    (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                    tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_i == 1
                                                    % tmp_i == 1
                                                    % next_start_index_origin = 2;
                                                    % next_end_index_origin = 4;
                                                    % next_start_index_correction = 3;
                                                    % next_end_index_correction = 5;
                                                    % tmp_i에 1-1, tmp_i+1에 1-2 new vertex

                                                    target_y_correction(3:5) = target_y_correction(2:4);
                                                    target_x_correction(3:5) = target_x_correction(2:4);

                                                elseif tmp_i == 2
                                                    % tmp_i == 2
                                                    % next_start_index_origin = 3;
                                                    % next_end_index_origin = 5; -> 4
                                                    % next_start_index_correction = 4;
                                                    % next_end_index_correction = 6; -> 5
                                                    % tmp_i에 2-1, tmp_i+1에 2-2 new vertex

                                                    target_y_correction(4:5) = target_y_correction(3:4);
                                                    target_x_correction(4:5) = target_x_correction(3:4);

                                                elseif tmp_i == 3
                                                    % tmp_i == 3
                                                    % next_start_index_origin = 4;
                                                    % next_end_index_origin = 6; -> 4
                                                    % next_start_index_correction = 5;
                                                    % next_end_index_correction = 7; -> 5
                                                    % tmp_i에 3-1, tmp_i+1에 3-2 new vertex

                                                    target_y_correction(5) = target_y_correction(4);
                                                    target_x_correction(5) = target_x_correction(4);

                                                elseif tmp_i == 4
                                                    % tmp_i == 4
                                                    % next_start_index_origin = 5; -> []
                                                    % next_end_index_origin = 7; -> []
                                                    % next_start_index_correction = 6; -> []
                                                    % next_end_index_correction = 8; -> []
                                                    % tmp_i에 4-1, tmp_i+1에 4-2 new vertex

                                                end

                                                % current ~ before vertex
                                                m_1 = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                tmp_base_1 = tmp_y_vertex_1 - m_1*tmp_x_vertex_1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross_1 = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross_1 = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross_1 = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross_1 = m_1*x_cross_1 + tmp_base_1;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross_1 = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross_1 = m_1*x_cross_1 + tmp_base_1;
                                                end


                                                % current ~ next vertex
                                                m_2 = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                                tmp_base_2 = tmp_y_vertex1 - m_2*tmp_x_vertex1;

                                                if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                    y_cross_2 = SBEV_PARAM.RANGE.Y_MIN;
                                                    x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                                                elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                    y_cross_2 = SBEV_PARAM.RANGE.Y_MAX;
                                                    x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                                                elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                    x_cross_2 = SBEV_PARAM.RANGE.X_MIN;
                                                    y_cross_2 = m_2*x_cross_2 + tmp_base_2;

                                                elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                    x_cross_2 = SBEV_PARAM.RANGE.X_MAX;
                                                    y_cross_2 = m_2*x_cross_2 + tmp_base_2;
                                                end

                                                target_y_correction(tmp_i) = y_cross_1;
                                                target_x_correction(tmp_i) = x_cross_1;

                                                target_y_correction(tmp_i+1) = y_cross_2;
                                                target_x_correction(tmp_i+1) = x_cross_2;


                                            elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                    tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_x_vertex0 == tmp_x_vertex_1
                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = tmp_x_vertex0;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = tmp_x_vertex0;
                                                    end

                                                elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                    if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = tmp_y_vertex0;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = tmp_y_vertex0;
                                                    end
                                                else
                                                    m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                    tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = m*x_cross + tmp_base;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = m*x_cross + tmp_base;
                                                    end
                                                end


                                                if tmp_i == 1
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;

                                                    target_y_correction(5) = y_cross;
                                                    target_x_correction(5) = x_cross;
                                                else
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;
                                                end

                                            end
                                        end
                                    end

                                    for tmp_i = 1:length(target_y_correction)
                                        tmp_y_vertex0 = target_y_correction(tmp_i);
                                        tmp_x_vertex0 = target_x_correction(tmp_i);

                                        if tmp_i < 5
                                            tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                            tmp_x_vertex1 = target_x_correction(tmp_i+1);
                                        else
                                            tmp_y_vertex1 = target_y_correction(1);
                                            tmp_x_vertex1 = target_x_correction(1);
                                        end

                                        [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                        [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                        [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                        f_row = i_row + length(tmp_x_contour) - 1;
                                        x_contour_total(i_row:f_row) = tmp_x_contour;
                                        y_contour_total(i_row:f_row) = tmp_y_contour;

                                        i_row = f_row + 1;

                                    end

                                elseif TWO_VERTEX_ROI_OUT_FLAG
                                    target_y_correction = target_y;
                                    target_x_correction = target_x;

                                    y_cross = 0;
                                    x_cross = 0;

                                    for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                        tmp_y_vertex0 = target_y(tmp_i);
                                        tmp_x_vertex0 = target_x(tmp_i);

                                        tmp_y_vertex1 = target_y(tmp_i+1);
                                        tmp_x_vertex1 = target_x(tmp_i+1);

                                        if tmp_i == 1
                                            tmp_y_vertex_1 = target_y(4);
                                            tmp_x_vertex_1 = target_x(4);
                                        else
                                            tmp_y_vertex_1 = target_y(tmp_i - 1);
                                            tmp_x_vertex_1 = target_x(tmp_i - 1);
                                        end

                                        if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                            if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                                                    tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_x_vertex0 == tmp_x_vertex1
                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = tmp_x_vertex0;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = tmp_x_vertex0;
                                                    end

                                                elseif tmp_y_vertex0 == tmp_y_vertex1
                                                    if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        y_cross = tmp_y_vertex0;
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        y_cross = tmp_y_vertex0;
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    end
                                                else
                                                    m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                                    tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = m*x_cross + tmp_base;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = m*x_cross + tmp_base;
                                                    end
                                                end

                                                if tmp_i == 1
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;

                                                    target_y_correction(5) = y_cross;
                                                    target_x_correction(5) = x_cross;
                                                else
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;
                                                end


                                            elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                    tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_x_vertex0 == tmp_x_vertex_1
                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = tmp_x_vertex0;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = tmp_x_vertex0;
                                                    end

                                                elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                    if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = tmp_y_vertex0;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = tmp_y_vertex0;
                                                    end
                                                else
                                                    m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                    tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = m*x_cross + tmp_base;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = m*x_cross + tmp_base;
                                                    end
                                                end

                                                if tmp_i == 1
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;

                                                    target_y_correction(5) = y_cross;
                                                    target_x_correction(5) = x_cross;
                                                else
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;
                                                end
                                            end
                                        end
                                    end

                                    for tmp_i = 1:length(target_y_correction) - 1
                                        tmp_y_vertex0 = target_y_correction(tmp_i);
                                        tmp_x_vertex0 = target_x_correction(tmp_i);

                                        tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                        tmp_x_vertex1 = target_x_correction(tmp_i+1);

                                        [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                        [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                        [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                        f_row = i_row + length(tmp_x_contour) - 1;
                                        x_contour_total(i_row:f_row) = tmp_x_contour;
                                        y_contour_total(i_row:f_row) = tmp_y_contour;

                                        i_row = f_row + 1;

                                    end

                                elseif THREE_VERTEX_ROI_OUT_FLAG
                                    target_y_correction = target_y;
                                    target_x_correction = target_x;

                                    y_cross = 0;
                                    x_cross = 0;

                                    vertex_index_beforeCurrentNext_all_out = 0;

                                    for tmp_i = 1:length(tmp_target_y_vertex) - 1

                                        tmp_y_vertex0 = target_y(tmp_i);
                                        tmp_x_vertex0 = target_x(tmp_i);

                                        tmp_y_vertex1 = target_y(tmp_i+1);
                                        tmp_x_vertex1 = target_x(tmp_i+1);

                                        if tmp_i == 1
                                            tmp_y_vertex_1 = target_y(4);
                                            tmp_x_vertex_1 = target_x(4);
                                        else
                                            tmp_y_vertex_1 = target_y(tmp_i - 1);
                                            tmp_x_vertex_1 = target_x(tmp_i - 1);
                                        end

                                        if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                                            if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                                                    tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_x_vertex0 == tmp_x_vertex1
                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = tmp_x_vertex0;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = tmp_x_vertex0;
                                                    end

                                                elseif tmp_y_vertex0 == tmp_y_vertex1
                                                    if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        y_cross = tmp_y_vertex0;
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        y_cross = tmp_y_vertex0;
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                    end
                                                else
                                                    m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                                    tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = m*x_cross + tmp_base;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = m*x_cross + tmp_base;
                                                    end
                                                end

                                                if tmp_i == 1
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;

                                                    target_y_correction(5) = y_cross;
                                                    target_x_correction(5) = x_cross;
                                                else
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;
                                                end


                                            elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                                    tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                                                if tmp_x_vertex0 == tmp_x_vertex_1
                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = tmp_x_vertex0;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = tmp_x_vertex0;
                                                    end

                                                elseif tmp_y_vertex0 == tmp_y_vertex_1
                                                    if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = tmp_y_vertex0;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = tmp_y_vertex0;
                                                    end
                                                else
                                                    m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                                    tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                                    if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                                        y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                                        y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                                        x_cross = (y_cross - tmp_base)/m;

                                                    elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                                        x_cross = SBEV_PARAM.RANGE.X_MIN;
                                                        y_cross = m*x_cross + tmp_base;

                                                    elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                                        x_cross = SBEV_PARAM.RANGE.X_MAX;
                                                        y_cross = m*x_cross + tmp_base;
                                                    end
                                                end

                                                if tmp_i == 1
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;

                                                    target_y_correction(5) = y_cross;
                                                    target_x_correction(5) = x_cross;
                                                else
                                                    target_y_correction(tmp_i) = y_cross;
                                                    target_x_correction(tmp_i) = x_cross;
                                                end

                                            else % current, next, before vertex all out of ROI
                                                vertex_index_beforeCurrentNext_all_out = tmp_i;
                                            end
                                        end
                                    end

                                    if vertex_index_beforeCurrentNext_all_out ~= 0

                                        % vertex x,y 중 하나라도 ROI에 포함되는 경우
                                        if ( target_y_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.Y_MAX ) || ...
                                                ( target_x_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.X_MAX )

                                            if vertex_index_beforeCurrentNext_all_out == 1
                                                target_y_correction(1) = target_y_correction(4);
                                                target_x_correction(1) = target_x_correction(4);
                                            else
                                                target_y_correction(vertex_index_beforeCurrentNext_all_out) = target_y_correction(vertex_index_beforeCurrentNext_all_out-1);
                                                target_x_correction(vertex_index_beforeCurrentNext_all_out) = target_x_correction(vertex_index_beforeCurrentNext_all_out-1);
                                            end

                                            % 모두 벗어나는 경우
                                        elseif ~(target_y_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.Y_MAX &&...
                                                target_x_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.X_MAX)

                                            if target_y_correction(vertex_index_beforeCurrentNext_all_out) < SBEV_PARAM.RANGE.Y_MIN
                                                target_y_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.Y_MIN;
                                            elseif target_y_correction(vertex_index_beforeCurrentNext_all_out) > SBEV_PARAM.RANGE.Y_MAX
                                                target_y_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.Y_MAX;
                                            end

                                            if target_x_correction(vertex_index_beforeCurrentNext_all_out) < SBEV_PARAM.RANGE.X_MIN
                                                target_x_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.X_MIN;
                                            elseif target_x_correction(vertex_index_beforeCurrentNext_all_out) > SBEV_PARAM.RANGE.X_MAX
                                                target_x_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.X_MAX;
                                            end
                                        end
                                    end

                                    for tmp_i = 1:length(target_y_correction) - 1
                                        tmp_y_vertex0 = target_y_correction(tmp_i);
                                        tmp_x_vertex0 = target_x_correction(tmp_i);

                                        tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                        tmp_x_vertex1 = target_x_correction(tmp_i+1);

                                        [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                        [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                        [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                        f_row = i_row + length(tmp_x_contour) - 1;
                                        x_contour_total(i_row:f_row) = tmp_x_contour;
                                        y_contour_total(i_row:f_row) = tmp_y_contour;

                                        i_row = f_row + 1;

                                    end

                                else
                                    for tmp_i = 1:length(tmp_target_y_vertex) - 1
                                        tmp_y_vertex0 = target_y(tmp_i);
                                        tmp_x_vertex0 = target_x(tmp_i);

                                        tmp_y_vertex1 = target_y(tmp_i+1);
                                        tmp_x_vertex1 = target_x(tmp_i+1);

                                        [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                                        [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                                        [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                                        [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                                        f_row = i_row + length(tmp_x_contour) - 1;
                                        x_contour_total(i_row:f_row) = tmp_x_contour;
                                        y_contour_total(i_row:f_row) = tmp_y_contour;

                                        i_row = f_row + 1;
                                    end
                                end

                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                % Find pixel to fill bounding box corresponding to predicted position
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                x_contour_total(f_row+1:end) = [];
                                y_contour_total(f_row+1:end) = [];

                                pixel_info = zeros(f_row,3);
                                [sorted_x_contour_total, sorted_index] = sort(x_contour_total);
                                sorted_y_contour_total = y_contour_total(sorted_index);
                                y_i = 1000;
                                y_f = 0;
                                i_row = 0;

                                for tmp_i = 1:length(x_contour_total) - 1

                                    if sorted_x_contour_total(tmp_i) == sorted_x_contour_total(tmp_i + 1)

                                        tmp_y = sorted_y_contour_total(tmp_i);

                                        if tmp_y > y_f
                                            y_f = tmp_y;
                                        end

                                        if tmp_y < y_i
                                            y_i = tmp_y;
                                        end

                                        if tmp_i == length(x_contour_total) - 1
                                            i_row = i_row + 1;
                                            pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);

                                            if y_i > sorted_y_contour_total(tmp_i + 1)
                                                y_i = sorted_y_contour_total(tmp_i + 1);
                                            end

                                            if y_f < sorted_y_contour_total(tmp_i + 1)
                                                y_f = sorted_y_contour_total(tmp_i + 1);
                                            end

                                            pixel_info(i_row,2) = y_i;
                                            pixel_info(i_row,3) = y_f;
                                        end

                                    else
                                        i_row = i_row + 1;
                                        pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);

                                        if tmp_i == 1
                                            y_i = sorted_y_contour_total(tmp_i);
                                            y_f = y_i;
                                        elseif tmp_i == length(x_contour_total) - 1
                                            pixel_info(i_row + 1,2) = sorted_y_contour_total(tmp_i + 1);
                                            pixel_info(i_row + 1,3) = sorted_y_contour_total(tmp_i + 1);
                                        else
                                            if y_i == y_f
                                                if sorted_y_contour_total(tmp_i - 1) > sorted_y_contour_total(tmp_i)
                                                    y_i = sorted_y_contour_total(tmp_i);
                                                    y_f = sorted_y_contour_total(tmp_i - 1);
                                                elseif sorted_y_contour_total(tmp_i - 1) < sorted_y_contour_total(tmp_i)
                                                    y_i = sorted_y_contour_total(tmp_i - 1);
                                                    y_f = sorted_y_contour_total(tmp_i);
                                                else
                                                    y_i = sorted_y_contour_total(tmp_i - 1);
                                                    y_f = y_i;
                                                end
                                            else
                                                if y_i > sorted_y_contour_total(tmp_i)
                                                    y_i = sorted_y_contour_total(tmp_i);
                                                end

                                                if y_f < sorted_y_contour_total(tmp_i)
                                                    y_f = sorted_y_contour_total(tmp_i);
                                                end
                                            end

                                        end
                                        pixel_info(i_row,2) = y_i;
                                        pixel_info(i_row,3) = y_f;

                                        y_i = 1000;
                                        y_f = 0;
                                    end
                                end

                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                % Apply pixel information to DSM
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                if SBEV_PARAM.RGB_IMAGE == 1
                                    if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                            for i_ch = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end
                                                end
                                            end

                                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                                            for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                                for i_info = 1:CH_LENGTH
                                                    if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                        if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1
                                                            [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                            for tmp_j = 1:length(pixel_info(:,1))
                                                                if pixel_info(tmp_j,1) ~= 0
                                                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                                else
                                                                    break
                                                                end
                                                            end


                                                        elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0
                                                            for tmp_j = 1:length(pixel_info(:,1))
                                                                if pixel_info(tmp_j,1) ~= 0
                                                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                                else
                                                                    break
                                                                end
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    end
                                                end
                                            end
                                        end

                                    elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1

                                        % 수정본
                                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                            for i_ch = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                                        if SBEV_PARAM.PREDICTION.FADING.ON
                                                            tmp_white_vector = SBEV_PARAM.RGB_MAX - ( I_LAT_uint8 - 1 );
                                                            fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                            for tmp_j = 1:length(x_contour_total)
                                                                SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = ( I_LAT_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                            end
                                                        else
                                                            for tmp_j = 1:length(x_contour_total)
                                                                SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                                        
                                                        if SBEV_PARAM.PREDICTION.FADING.ON
                                                            tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                            fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                            for tmp_j = 1:length(x_contour_total)
                                                                SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;
                                                            end
                                                        else
                                                            for tmp_j = 1:length(x_contour_total)
                                                                SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

%                                                         for tmp_j = 1:length(pixel_info(:,1))
%                                                             if pixel_info(tmp_j,1) ~= 0
%                                                                 SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
%                                                             else
%                                                                 break
%                                                             end
%                                                         end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1), pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY

                                                    if SBEV_PARAM.PREDICTION.FADING.ON
                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - ( Collision_Probability_uint8 - 1 );
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = ( Collision_Probability_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                        end
                                                    else
                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA

                                                    if SBEV_PARAM.PREDICTION.FADING.ON
                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;

                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;
                                                        end
                                                    else
                                                        for tmp_j = 1:length(x_contour_total)
                                                            SBEV_out(x_contour_total(tmp_j), y_contour_total(tmp_j), SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                        end
                                                    end

                                                end
                                            end

                                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                                            for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                                for i_info = 1:CH_LENGTH
                                                    if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                        if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1
                                                            [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                            for tmp_j = 1:length(pixel_info(:,1))
                                                                if pixel_info(tmp_j,1) ~= 0
                                                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                                else
                                                                    break
                                                                end
                                                            end

                                                        elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0
                                                            for tmp_j = 1:length(pixel_info(:,1))
                                                                if pixel_info(tmp_j,1) ~= 0
                                                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                                else
                                                                    break
                                                                end
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end

                                elseif SBEV_PARAM.GRAY_IMAGE == 1
                                    if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1 || SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                                            for i_ch = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

                                                        for tmp_j = 1:length(pixel_info(:,1))
                                                            if pixel_info(tmp_j,1) ~= 0
                                                                SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            else
                                                                break
                                                            end
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                                    for tmp_j = 1:length(pixel_info(:,1))
                                                        if pixel_info(tmp_j,1) ~= 0
                                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3), SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        else
                                                            break
                                                        end
                                                    end

                                                end
                                            end
                                        end
                                    end
                                end
                            end

                            %                     figure
                            %                     imshow(uint8(SBEV_out))


                        else % trajectory

                            index_pred_detail = round(index_pred*SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);

                            target_y = predicted_y(index_pred_detail);
                            target_x = predicted_x(index_pred_detail);

                            if target_y >= SBEV_PARAM.RANGE.Y_MIN && target_y <= SBEV_PARAM.RANGE.Y_MAX ...
                                    && target_x >= SBEV_PARAM.RANGE.X_MIN && target_x <= SBEV_PARAM.RANGE.X_MAX

                                [~,Image_Position_X] = min(abs(target_x - SBEV_PARAM.RANGE.X_RANGE));
                                [~,Image_Position_Y] = min(abs(target_y - SBEV_PARAM.RANGE.Y_RANGE));

                                if SBEV_PARAM.RGB_IMAGE == 1
                                    if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0
                                            for i_ch = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                                    if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                                    elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                    elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                                        if index_traj ~= length(State_trajectory(1,:))
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                        elseif index_traj == length(State_trajectory(1,:))
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                end
                                            end

                                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                                            for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                                for i_info = 1:CH_LENGTH
                                                    if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                                        if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                                        elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                        elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                                            if index_traj ~= length(State_trajectory(1,:))
                                                                SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            elseif index_traj == length(State_trajectory(1,:))
                                                                SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                        SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                    end
                                                end
                                            end
                                        end

                                    elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                                        if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0
                                            for i_ch = 1:CH_LENGTH
                                                if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                                    if SBEV_PARAM.PREDICTION.TRAJECTORY_THREAT
                                                        if SBEV_PARAM.PREDICTION.FADING.ON
                                                            tmp_white_vector = SBEV_PARAM.RGB_MAX - ( I_LAT_uint8 - 1 );
                                                            fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = ( I_LAT_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                        else
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8 - 1;
                                                        end
                                                    else
                                                        if SBEV_PARAM.PREDICTION.FADING.ON
                                                            tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                            fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;
                                                        else
                                                            SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                        end
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY

                                                    if SBEV_PARAM.PREDICTION.FADING.ON
                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - ( Collision_Probability_uint8 - 1 );
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = ( Collision_Probability_uint8 - 1 ) + tmp_white_vector * fading_factor_step;
                                                    else
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;
                                                    end

                                                elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA

                                                    if SBEV_PARAM.PREDICTION.FADING.ON
                                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                                        fading_factor_step = SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_VALUE + ( index_pred - 1 ) * SBEV_PARAM.PREDICTION.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * fading_factor_step;
                                                    else
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                    end

                                                end
                                            end

                                        elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1
                                            for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                                                for i_info = 1:CH_LENGTH
                                                    if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                                        if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                                        elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                        elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                                            if index_traj ~= length(State_trajectory(1,:))
                                                                SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                            elseif index_traj == length(State_trajectory(1,:))
                                                                SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                            end
                                                        end

                                                    elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                        SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                    end
                                                end
                                            end
                                        end
                                    end

                                elseif SBEV_PARAM.GRAY_IMAGE == 1
                                    if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1

                                        for i_ch = 1:CH_LENGTH
                                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                                if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                                elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                                elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                                    if index_traj ~= length(State_trajectory(1,:))
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                                    elseif index_traj == length(State_trajectory(1,:))
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                            end
                                        end

                                    elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                                        for i_ch = 1:CH_LENGTH
                                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                                if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;

                                                elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                                    SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX - (I_LAT_uint8-1);

                                                elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                                    if index_traj ~= length(State_trajectory(1,:))
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                                    elseif index_traj == length(State_trajectory(1,:))
                                                        SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX - (I_LAT_uint8-1);
                                                    end
                                                end

                                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                                SBEV_out(Image_Position_X,Image_Position_Y, SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX - (I_LAT_uint8-1);
                                            end
                                        end
                                    end
                                end

                            end


                            %                     figure
                            %                     imshow(uint8(SBEV_out))

                        end

                    end

                end
                

            end                      
            
        end
        
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target Shape
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Target_Shape_Exist_Flag == 1 && (SBEV_PARAM.SHAPE.TARGET.POSITION == 1 || SBEV_PARAM.SHAPE.TARGET.THREAT == 1)
    
    tmp_target_y_vertex = [-State_trajectory(TRAJ_PARAM.WIDTH, end)/2, -State_trajectory(TRAJ_PARAM.WIDTH, end)/2,...
        State_trajectory(TRAJ_PARAM.WIDTH, end)/2, State_trajectory(TRAJ_PARAM.WIDTH, end)/2, -State_trajectory(TRAJ_PARAM.WIDTH, end)/2];
    tmp_target_x_vertex = [0, State_trajectory(TRAJ_PARAM.LENGTH, end), State_trajectory(TRAJ_PARAM.LENGTH, end), 0, 0];
    
    target_y_vertex_rot = tmp_target_x_vertex.*sin(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end)) + tmp_target_y_vertex.*cos(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end));
    target_x_vertex_rot = tmp_target_x_vertex.*cos(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end)) - tmp_target_y_vertex.*sin(State_trajectory(TRAJ_PARAM.HEADING_ANGLE, end));
    
    target_y = target_y_vertex_rot + State_trajectory(TRAJ_PARAM.REL_POS_Y, end);
    target_x = target_x_vertex_rot + State_trajectory(TRAJ_PARAM.REL_POS_X, end);
    
    ONLY_ONE_VERTEX_ROI_OUT_FLAG = 0;
    TWO_VERTEX_ROI_OUT_FLAG = 0;
    THREE_VERTEX_ROI_OUT_FLAG = 0;

    if ~( all(target_y >= SBEV_PARAM.RANGE.Y_MIN) && all(target_y <= SBEV_PARAM.RANGE.Y_MAX) && all(target_x >= SBEV_PARAM.RANGE.X_MIN) && all(target_x <= SBEV_PARAM.RANGE.X_MAX) )
        
        vertex_total = zeros(4, 4);

        vertex_total(1, :) = target_y(1:4) >= SBEV_PARAM.RANGE.Y_MIN;
        vertex_total(2, :) = target_y(1:4) <= SBEV_PARAM.RANGE.Y_MAX;
        vertex_total(3, :) = target_x(1:4) >= SBEV_PARAM.RANGE.X_MIN;
        vertex_total(4, :) = target_x(1:4) <= SBEV_PARAM.RANGE.X_MAX;

        vertex_out_flag = all(vertex_total);

        if nnz(vertex_out_flag) == 3 % only one vertex out of ROI
            ONLY_ONE_VERTEX_ROI_OUT_FLAG = 1;
        elseif nnz(vertex_out_flag) == 2 % two vertex out of ROI
            TWO_VERTEX_ROI_OUT_FLAG = 1;
        elseif nnz(vertex_out_flag) == 1 % three vertex out of ROI
            THREE_VERTEX_ROI_OUT_FLAG = 1;
        end
    end

    if ( (min(target_y) >= SBEV_PARAM.RANGE.Y_MIN && min(target_y) <= SBEV_PARAM.RANGE.Y_MAX) || (max(target_y) >= SBEV_PARAM.RANGE.Y_MIN && max(target_y) <= SBEV_PARAM.RANGE.Y_MAX)) ...
            && ((min(target_x) >= SBEV_PARAM.RANGE.X_MIN && min(target_x) <= SBEV_PARAM.RANGE.X_MAX) || (max(target_x) >= SBEV_PARAM.RANGE.X_MIN && max(target_x) <= SBEV_PARAM.RANGE.X_MAX))

%         Target_Shape_Exist_Flag = 1;

        x_contour_total = zeros(200,1);
        y_contour_total = zeros(200,1);

        i_row = 1;
        f_row = 0;

        if ONLY_ONE_VERTEX_ROI_OUT_FLAG
            target_y_correction = target_y;
            target_x_correction = target_x;

            y_cross = 0;
            x_cross = 0;

            for tmp_i = 1:length(tmp_target_y_vertex) - 1

                tmp_y_vertex0 = target_y(tmp_i);
                tmp_x_vertex0 = target_x(tmp_i);

                tmp_y_vertex1 = target_y(tmp_i+1);
                tmp_x_vertex1 = target_x(tmp_i+1);

                if tmp_i == 1
                    tmp_y_vertex_1 = target_y(4);
                    tmp_x_vertex_1 = target_x(4);
                else
                    tmp_y_vertex_1 = target_y(tmp_i - 1);
                    tmp_x_vertex_1 = target_x(tmp_i - 1);
                end

                if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                        tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                    if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next and before vertex in ROI
                            tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX) && ...
                            (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&...
                            tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                        if tmp_i == 1
                            % tmp_i == 1
                            % next_start_index_origin = 2;
                            % next_end_index_origin = 4;
                            % next_start_index_correction = 3;
                            % next_end_index_correction = 5;
                            % tmp_i에 1-1, tmp_i+1에 1-2 new vertex

                            target_y_correction(3:5) = target_y_correction(2:4);
                            target_x_correction(3:5) = target_x_correction(2:4);

                        elseif tmp_i == 2
                            % tmp_i == 2
                            % next_start_index_origin = 3;
                            % next_end_index_origin = 5; -> 4
                            % next_start_index_correction = 4;
                            % next_end_index_correction = 6; -> 5
                            % tmp_i에 2-1, tmp_i+1에 2-2 new vertex

                            target_y_correction(4:5) = target_y_correction(3:4);
                            target_x_correction(4:5) = target_x_correction(3:4);

                        elseif tmp_i == 3
                            % tmp_i == 3
                            % next_start_index_origin = 4;
                            % next_end_index_origin = 6; -> 4
                            % next_start_index_correction = 5;
                            % next_end_index_correction = 7; -> 5
                            % tmp_i에 3-1, tmp_i+1에 3-2 new vertex

                            target_y_correction(5) = target_y_correction(4);
                            target_x_correction(5) = target_x_correction(4);

                        elseif tmp_i == 4
                            % tmp_i == 4
                            % next_start_index_origin = 5; -> []
                            % next_end_index_origin = 7; -> []
                            % next_start_index_correction = 6; -> []
                            % next_end_index_correction = 8; -> []
                            % tmp_i에 4-1, tmp_i+1에 4-2 new vertex

                        end

                        % current ~ before vertex
                        m_1 = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                        tmp_base_1 = tmp_y_vertex_1 - m_1*tmp_x_vertex_1;

                        if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                            y_cross_1 = SBEV_PARAM.RANGE.Y_MIN;
                            x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                        elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                            y_cross_1 = SBEV_PARAM.RANGE.Y_MAX;
                            x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                        elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                            x_cross_1 = SBEV_PARAM.RANGE.X_MIN;
                            y_cross_1 = m_1*x_cross_1 + tmp_base_1;

                        elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                            x_cross_1 = SBEV_PARAM.RANGE.X_MAX;
                            y_cross_1 = m_1*x_cross_1 + tmp_base_1;
                        end


                        % current ~ next vertex
                        m_2 = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                        tmp_base_2 = tmp_y_vertex1 - m_2*tmp_x_vertex1;

                        if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                            y_cross_2 = SBEV_PARAM.RANGE.Y_MIN;
                            x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                        elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                            y_cross_2 = SBEV_PARAM.RANGE.Y_MAX;
                            x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                        elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                            x_cross_2 = SBEV_PARAM.RANGE.X_MIN;
                            y_cross_2 = m_2*x_cross_2 + tmp_base_2;

                        elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                            x_cross_2 = SBEV_PARAM.RANGE.X_MAX;
                            y_cross_2 = m_2*x_cross_2 + tmp_base_2;
                        end

                        target_y_correction(tmp_i) = y_cross_1;
                        target_x_correction(tmp_i) = x_cross_1;

                        target_y_correction(tmp_i+1) = y_cross_2;
                        target_x_correction(tmp_i+1) = x_cross_2;


                    elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                            tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                        if tmp_x_vertex0 == tmp_x_vertex_1
                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                x_cross = tmp_x_vertex0;

                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                x_cross = tmp_x_vertex0;
                            end

                        elseif tmp_y_vertex0 == tmp_y_vertex_1
                            if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                x_cross = SBEV_PARAM.RANGE.X_MIN;
                                y_cross = tmp_y_vertex0;

                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                x_cross = SBEV_PARAM.RANGE.X_MAX;
                                y_cross = tmp_y_vertex0;
                            end
                        else
                            m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                            tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                x_cross = SBEV_PARAM.RANGE.X_MIN;
                                y_cross = m*x_cross + tmp_base;

                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                x_cross = SBEV_PARAM.RANGE.X_MAX;
                                y_cross = m*x_cross + tmp_base;
                            end
                        end


                        if tmp_i == 1
                            target_y_correction(tmp_i) = y_cross;
                            target_x_correction(tmp_i) = x_cross;

                            target_y_correction(5) = y_cross;
                            target_x_correction(5) = x_cross;
                        else
                            target_y_correction(tmp_i) = y_cross;
                            target_x_correction(tmp_i) = x_cross;
                        end

                    end
                end
            end

            for tmp_i = 1:length(target_y_correction)
                tmp_y_vertex0 = target_y_correction(tmp_i);
                tmp_x_vertex0 = target_x_correction(tmp_i);

                if tmp_i < 5
                    tmp_y_vertex1 = target_y_correction(tmp_i+1);
                    tmp_x_vertex1 = target_x_correction(tmp_i+1);
                else
                    tmp_y_vertex1 = target_y_correction(1);
                    tmp_x_vertex1 = target_x_correction(1);
                end

                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                f_row = i_row + length(tmp_x_contour) - 1;
                x_contour_total(i_row:f_row) = tmp_x_contour;
                y_contour_total(i_row:f_row) = tmp_y_contour;

                i_row = f_row + 1;

            end

        elseif TWO_VERTEX_ROI_OUT_FLAG
            target_y_correction = target_y;
            target_x_correction = target_x;

            y_cross = 0;
            x_cross = 0;

            for tmp_i = 1:length(tmp_target_y_vertex) - 1

                tmp_y_vertex0 = target_y(tmp_i);
                tmp_x_vertex0 = target_x(tmp_i);

                tmp_y_vertex1 = target_y(tmp_i+1);
                tmp_x_vertex1 = target_x(tmp_i+1);

                if tmp_i == 1
                    tmp_y_vertex_1 = target_y(4);
                    tmp_x_vertex_1 = target_x(4);
                else
                    tmp_y_vertex_1 = target_y(tmp_i - 1);
                    tmp_x_vertex_1 = target_x(tmp_i - 1);
                end

                if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                        tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                    if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                            tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX)

                        if tmp_x_vertex0 == tmp_x_vertex1
                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                x_cross = tmp_x_vertex0;

                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                x_cross = tmp_x_vertex0;
                            end

                        elseif tmp_y_vertex0 == tmp_y_vertex1
                            if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                y_cross = tmp_y_vertex0;
                                x_cross = SBEV_PARAM.RANGE.X_MIN;

                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                y_cross = tmp_y_vertex0;
                                x_cross = SBEV_PARAM.RANGE.X_MAX;
                            end
                        else
                            m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                            tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                x_cross = SBEV_PARAM.RANGE.X_MIN;
                                y_cross = m*x_cross + tmp_base;

                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                x_cross = SBEV_PARAM.RANGE.X_MAX;
                                y_cross = m*x_cross + tmp_base;
                            end
                        end

                        if tmp_i == 1
                            target_y_correction(tmp_i) = y_cross;
                            target_x_correction(tmp_i) = x_cross;

                            target_y_correction(5) = y_cross;
                            target_x_correction(5) = x_cross;
                        else
                            target_y_correction(tmp_i) = y_cross;
                            target_x_correction(tmp_i) = x_cross;
                        end


                    elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                            tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                        if tmp_x_vertex0 == tmp_x_vertex_1
                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                x_cross = tmp_x_vertex0;

                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                x_cross = tmp_x_vertex0;
                            end

                        elseif tmp_y_vertex0 == tmp_y_vertex_1
                            if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                x_cross = SBEV_PARAM.RANGE.X_MIN;
                                y_cross = tmp_y_vertex0;

                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                x_cross = SBEV_PARAM.RANGE.X_MAX;
                                y_cross = tmp_y_vertex0;
                            end
                        else
                            m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                            tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                x_cross = SBEV_PARAM.RANGE.X_MIN;
                                y_cross = m*x_cross + tmp_base;

                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                x_cross = SBEV_PARAM.RANGE.X_MAX;
                                y_cross = m*x_cross + tmp_base;
                            end
                        end

                        if tmp_i == 1
                            target_y_correction(tmp_i) = y_cross;
                            target_x_correction(tmp_i) = x_cross;

                            target_y_correction(5) = y_cross;
                            target_x_correction(5) = x_cross;
                        else
                            target_y_correction(tmp_i) = y_cross;
                            target_x_correction(tmp_i) = x_cross;
                        end
                    end
                end
            end

            for tmp_i = 1:length(target_y_correction) - 1
                tmp_y_vertex0 = target_y_correction(tmp_i);
                tmp_x_vertex0 = target_x_correction(tmp_i);

                tmp_y_vertex1 = target_y_correction(tmp_i+1);
                tmp_x_vertex1 = target_x_correction(tmp_i+1);

                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                f_row = i_row + length(tmp_x_contour) - 1;
                x_contour_total(i_row:f_row) = tmp_x_contour;
                y_contour_total(i_row:f_row) = tmp_y_contour;

                i_row = f_row + 1;

            end

        elseif THREE_VERTEX_ROI_OUT_FLAG
            target_y_correction = target_y;
            target_x_correction = target_x;

            y_cross = 0;
            x_cross = 0;

            vertex_index_beforeCurrentNext_all_out = 0;

            for tmp_i = 1:length(tmp_target_y_vertex) - 1

                tmp_y_vertex0 = target_y(tmp_i);
                tmp_x_vertex0 = target_x(tmp_i);

                tmp_y_vertex1 = target_y(tmp_i+1);
                tmp_x_vertex1 = target_x(tmp_i+1);

                if tmp_i == 1
                    tmp_y_vertex_1 = target_y(4);
                    tmp_x_vertex_1 = target_x(4);
                else
                    tmp_y_vertex_1 = target_y(tmp_i - 1);
                    tmp_x_vertex_1 = target_x(tmp_i - 1);
                end

                if ~(tmp_y_vertex0 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= SBEV_PARAM.RANGE.Y_MAX &&...
                        tmp_x_vertex0 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= SBEV_PARAM.RANGE.X_MAX)

                    if (tmp_y_vertex1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= SBEV_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                            tmp_x_vertex1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= SBEV_PARAM.RANGE.X_MAX)

                        if tmp_x_vertex0 == tmp_x_vertex1
                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                x_cross = tmp_x_vertex0;

                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                x_cross = tmp_x_vertex0;
                            end

                        elseif tmp_y_vertex0 == tmp_y_vertex1
                            if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                y_cross = tmp_y_vertex0;
                                x_cross = SBEV_PARAM.RANGE.X_MIN;

                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                y_cross = tmp_y_vertex0;
                                x_cross = SBEV_PARAM.RANGE.X_MAX;
                            end
                        else
                            m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                            tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                x_cross = SBEV_PARAM.RANGE.X_MIN;
                                y_cross = m*x_cross + tmp_base;

                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                x_cross = SBEV_PARAM.RANGE.X_MAX;
                                y_cross = m*x_cross + tmp_base;
                            end
                        end

                        if tmp_i == 1
                            target_y_correction(tmp_i) = y_cross;
                            target_x_correction(tmp_i) = x_cross;

                            target_y_correction(5) = y_cross;
                            target_x_correction(5) = x_cross;
                        else
                            target_y_correction(tmp_i) = y_cross;
                            target_x_correction(tmp_i) = x_cross;
                        end


                    elseif (tmp_y_vertex_1 >= SBEV_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= SBEV_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                            tmp_x_vertex_1 >= SBEV_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= SBEV_PARAM.RANGE.X_MAX)

                        if tmp_x_vertex0 == tmp_x_vertex_1
                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                x_cross = tmp_x_vertex0;

                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                x_cross = tmp_x_vertex0;
                            end

                        elseif tmp_y_vertex0 == tmp_y_vertex_1
                            if tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                x_cross = SBEV_PARAM.RANGE.X_MIN;
                                y_cross = tmp_y_vertex0;

                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                x_cross = SBEV_PARAM.RANGE.X_MAX;
                                y_cross = tmp_y_vertex0;
                            end
                        else
                            m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                            tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                            if tmp_y_vertex0 < SBEV_PARAM.RANGE.Y_MIN
                                y_cross = SBEV_PARAM.RANGE.Y_MIN;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_y_vertex0 > SBEV_PARAM.RANGE.Y_MAX
                                y_cross = SBEV_PARAM.RANGE.Y_MAX;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_x_vertex0 < SBEV_PARAM.RANGE.X_MIN
                                x_cross = SBEV_PARAM.RANGE.X_MIN;
                                y_cross = m*x_cross + tmp_base;

                            elseif tmp_x_vertex0 > SBEV_PARAM.RANGE.X_MAX
                                x_cross = SBEV_PARAM.RANGE.X_MAX;
                                y_cross = m*x_cross + tmp_base;
                            end
                        end

                        if tmp_i == 1
                            target_y_correction(tmp_i) = y_cross;
                            target_x_correction(tmp_i) = x_cross;

                            target_y_correction(5) = y_cross;
                            target_x_correction(5) = x_cross;
                        else
                            target_y_correction(tmp_i) = y_cross;
                            target_x_correction(tmp_i) = x_cross;
                        end

                    else % current, next, before vertex all out of ROI
                        vertex_index_beforeCurrentNext_all_out = tmp_i;
                    end
                end
            end

            if vertex_index_beforeCurrentNext_all_out ~= 0

                % vertex x,y 중 하나라도 ROI에 포함되는 경우
                if ( target_y_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.Y_MAX ) || ...
                        ( target_x_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.X_MAX )

                    if vertex_index_beforeCurrentNext_all_out == 1
                        target_y_correction(1) = target_y_correction(4);
                        target_x_correction(1) = target_x_correction(4);
                    else
                        target_y_correction(vertex_index_beforeCurrentNext_all_out) = target_y_correction(vertex_index_beforeCurrentNext_all_out-1);
                        target_x_correction(vertex_index_beforeCurrentNext_all_out) = target_x_correction(vertex_index_beforeCurrentNext_all_out-1);
                    end

                % 모두 벗어나는 경우
                elseif ~(target_y_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.Y_MAX &&...
                        target_x_correction(vertex_index_beforeCurrentNext_all_out) >= SBEV_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= SBEV_PARAM.RANGE.X_MAX)

                    if target_y_correction(vertex_index_beforeCurrentNext_all_out) < SBEV_PARAM.RANGE.Y_MIN
                        target_y_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.Y_MIN;
                    elseif target_y_correction(vertex_index_beforeCurrentNext_all_out) > SBEV_PARAM.RANGE.Y_MAX
                        target_y_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.Y_MAX;
                    end

                    if target_x_correction(vertex_index_beforeCurrentNext_all_out) < SBEV_PARAM.RANGE.X_MIN
                        target_x_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.X_MIN;
                    elseif target_x_correction(vertex_index_beforeCurrentNext_all_out) > SBEV_PARAM.RANGE.X_MAX
                        target_x_correction(vertex_index_beforeCurrentNext_all_out) = SBEV_PARAM.RANGE.X_MAX;
                    end
                end
            end

            for tmp_i = 1:length(target_y_correction) - 1
                tmp_y_vertex0 = target_y_correction(tmp_i);
                tmp_x_vertex0 = target_x_correction(tmp_i);

                tmp_y_vertex1 = target_y_correction(tmp_i+1);
                tmp_x_vertex1 = target_x_correction(tmp_i+1);

                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                f_row = i_row + length(tmp_x_contour) - 1;
                x_contour_total(i_row:f_row) = tmp_x_contour;
                y_contour_total(i_row:f_row) = tmp_y_contour;

                i_row = f_row + 1;

            end

        else
            for tmp_i = 1:length(tmp_target_y_vertex) - 1
                tmp_y_vertex0 = target_y(tmp_i);
                tmp_x_vertex0 = target_x(tmp_i);

                tmp_y_vertex1 = target_y(tmp_i+1);
                tmp_x_vertex1 = target_x(tmp_i+1);

                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - SBEV_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - SBEV_PARAM.RANGE.X_RANGE));

                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - SBEV_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - SBEV_PARAM.RANGE.X_RANGE));

                [tmp_x_contour, tmp_y_contour] = plotLine(Image_Position_X0, Image_Position_Y0, Image_Position_X1, Image_Position_Y1);

                f_row = i_row + length(tmp_x_contour) - 1;
                x_contour_total(i_row:f_row) = tmp_x_contour;
                y_contour_total(i_row:f_row) = tmp_y_contour;

                i_row = f_row + 1;

            end
        end

        x_contour_total(f_row+1:end) = [];
        y_contour_total(f_row+1:end) = [];

        pixel_info = zeros(f_row,3);
        [sorted_x_contour_total, sorted_index] = sort(x_contour_total);
        sorted_y_contour_total = y_contour_total(sorted_index);
        y_i = 1000;
        y_f = 0;
        i_row = 0;

        for tmp_i = 1:length(x_contour_total) - 1

            if sorted_x_contour_total(tmp_i) == sorted_x_contour_total(tmp_i + 1)

                tmp_y = sorted_y_contour_total(tmp_i);

                if tmp_y > y_f
                    y_f = tmp_y;
                end

                if tmp_y < y_i
                    y_i = tmp_y;
                end

                if tmp_i == length(x_contour_total) - 1
                    i_row = i_row + 1;
                    pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);

                    if y_i > sorted_y_contour_total(tmp_i + 1)
                        y_i = sorted_y_contour_total(tmp_i + 1);
                    end

                    if y_f < sorted_y_contour_total(tmp_i + 1)
                        y_f = sorted_y_contour_total(tmp_i + 1);
                    end

                    pixel_info(i_row,2) = y_i;
                    pixel_info(i_row,3) = y_f;
                end

            else
                i_row = i_row + 1;
                pixel_info(i_row,1) = sorted_x_contour_total(tmp_i);

                if tmp_i == 1
                    y_i = sorted_y_contour_total(tmp_i);
                    y_f = y_i;
                elseif tmp_i == length(x_contour_total) - 1
                    pixel_info(i_row + 1,2) = sorted_y_contour_total(tmp_i + 1);
                    pixel_info(i_row + 1,3) = sorted_y_contour_total(tmp_i + 1);
                else
                    if y_i == y_f
                        if sorted_y_contour_total(tmp_i - 1) > sorted_y_contour_total(tmp_i)
                            y_i = sorted_y_contour_total(tmp_i);
                            y_f = sorted_y_contour_total(tmp_i - 1);
                        elseif sorted_y_contour_total(tmp_i - 1) < sorted_y_contour_total(tmp_i)
                            y_i = sorted_y_contour_total(tmp_i - 1);
                            y_f = sorted_y_contour_total(tmp_i);
                        else
                            y_i = sorted_y_contour_total(tmp_i - 1);
                            y_f = y_i;
                        end
                    else
                        if y_i > sorted_y_contour_total(tmp_i)
                            y_i = sorted_y_contour_total(tmp_i);
                        end

                        if y_f < sorted_y_contour_total(tmp_i)
                            y_f = sorted_y_contour_total(tmp_i);
                        end
                    end

                end
                pixel_info(i_row,2) = y_i;
                pixel_info(i_row,3) = y_f;

                y_i = 1000;
                y_f = 0;
            end
        end

        if SBEV_PARAM.RGB_IMAGE == 1
            if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                    for i_ch = 1:CH_LENGTH
                        if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                            if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                    else
                                        break
                                    end
                                end

                            elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                    else
                                        break
                                    end
                                end

                            elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                    else
                                        break
                                    end
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                            [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                            for tmp_j = 1:length(pixel_info(:,1))
                                if pixel_info(tmp_j,1) ~= 0
                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                else
                                    break
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_X % VX
                            [~,VX_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_X, end) - SBEV_PARAM.RANGE.VX_RANGE));

                            for tmp_j = 1:length(pixel_info(:,1))
                                if pixel_info(tmp_j,1) ~= 0
                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VX_uint8-1;
                                else
                                    break
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_Y % VY
                            [~,VY_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_Y, end) - SBEV_PARAM.RANGE.VY_RANGE));

                            for tmp_j = 1:length(pixel_info(:,1))
                                if pixel_info(tmp_j,1) ~= 0
                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VY_uint8-1;
                                else
                                    break
                                end
                            end
                        end
                    end

                elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                    for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                        for i_info = 1:CH_LENGTH
                            if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1
                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                    for tmp_j = 1:length(pixel_info(:,1))
                                        if pixel_info(tmp_j,1) ~= 0
                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                        else
                                            break
                                        end
                                    end


                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0
                                    for tmp_j = 1:length(pixel_info(:,1))
                                        if pixel_info(tmp_j,1) ~= 0
                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                        else
                                            break
                                        end
                                    end
                                end

                            elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                    else
                                        break
                                    end
                                end

                            elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.REL_VEL_X % VX
                                [~,VX_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_X, end) - SBEV_PARAM.RANGE.VX_RANGE));
                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = VX_uint8-1;
                                    else
                                        break
                                    end
                                end

                            elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.REL_VEL_Y % VY
                                [~,VY_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_Y, end) - SBEV_PARAM.RANGE.VY_RANGE));
                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = VY_uint8-1;
                                    else
                                        break
                                    end
                                end
                            end
                        end
                    end
                end

            elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1

                % 수정본
                if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                    for i_ch = 1:CH_LENGTH
                        if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                            if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                    else
                                        break
                                    end
                                end

                            elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                    else
                                        break
                                    end
                                end

                            elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                    else
                                        break
                                    end
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                            [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                            for tmp_j = 1:length(pixel_info(:,1))
                                if pixel_info(tmp_j,1) ~= 0
                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                else
                                    break
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_X % VX
                            [~,VX_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_X, end) - SBEV_PARAM.RANGE.VX_RANGE));

                            for tmp_j = 1:length(pixel_info(:,1))
                                if pixel_info(tmp_j,1) ~= 0
                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VX_uint8-1;
                                else
                                    break
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_Y % VY
                            [~,VY_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_Y, end) - SBEV_PARAM.RANGE.VY_RANGE));

                            for tmp_j = 1:length(pixel_info(:,1))
                                if pixel_info(tmp_j,1) ~= 0
                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VY_uint8-1;
                                else
                                    break
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY
                            [~, Collision_Probability_uint8] = min(abs(State_trajectory(TRAJ_PARAM.COLLISION_PROBABILITY, end) - SBEV_PARAM.RANGE.COLLISION_PROBABILITY_RANGE));

                            for tmp_j = 1:length(pixel_info(:,1))
                                if pixel_info(tmp_j,1) ~= 0
                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;
                                else
                                    break
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA
                            for tmp_j = 1:length(pixel_info(:,1))
                                if pixel_info(tmp_j,1) ~= 0
                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                else
                                    break
                                end
                            end
                        end
                    end

                elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                    for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                        for i_info = 1:CH_LENGTH
                            if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1
                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                    for tmp_j = 1:length(pixel_info(:,1))
                                        if pixel_info(tmp_j,1) ~= 0
                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                        else
                                            break
                                        end
                                    end


                                elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0
                                    for tmp_j = 1:length(pixel_info(:,1))
                                        if pixel_info(tmp_j,1) ~= 0
                                            SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                        else
                                            break
                                        end
                                    end
                                end

                            elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                    else
                                        break
                                    end
                                end

                            elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.REL_VEL_X % VX
                                [~,VX_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_X, end) - SBEV_PARAM.RANGE.VX_RANGE));
                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = VX_uint8-1;
                                    else
                                        break
                                    end
                                end

                            elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.REL_VEL_Y % VY
                                [~,VY_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_Y, end) - SBEV_PARAM.RANGE.VY_RANGE));
                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = VY_uint8-1;
                                    else
                                        break
                                    end
                                end

                            elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.NA
                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                    else
                                        break
                                    end
                                end
                            end
                        end
                    end
                end
            end

        elseif SBEV_PARAM.GRAY_IMAGE == 1
            if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1 || SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0

                    for i_ch = 1:CH_LENGTH
                        if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                            if SBEV_PARAM.SHAPE.TARGET.POSITION == 0 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % threat metric in R channel

                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                    else
                                        break
                                    end
                                end

                            elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 0 % Occupancy in R channel
                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                    else
                                        break
                                    end
                                end

                            elseif SBEV_PARAM.SHAPE.TARGET.POSITION == 1 && SBEV_PARAM.SHAPE.TARGET.THREAT == 1 % Occupancy in position channel, threat metric in threat channel

                                for tmp_j = 1:length(pixel_info(:,1))
                                    if pixel_info(tmp_j,1) ~= 0
                                        SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                    else
                                        break
                                    end
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                            [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - SBEV_PARAM.RANGE.I_LAT_RANGE));

                            for tmp_j = 1:length(pixel_info(:,1))
                                if pixel_info(tmp_j,1) ~= 0
                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                else
                                    break
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_X % VX
                            [~,VX_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_X, end) - SBEV_PARAM.RANGE.VX_RANGE));

                            for tmp_j = 1:length(pixel_info(:,1))
                                if pixel_info(tmp_j,1) ~= 0
                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VX_uint8-1;
                                else
                                    break
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_Y % VY
                            [~,VY_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_Y, end) - SBEV_PARAM.RANGE.VY_RANGE));

                            for tmp_j = 1:length(pixel_info(:,1))
                                if pixel_info(tmp_j,1) ~= 0
                                    SBEV_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VY_uint8-1;
                                else
                                    break
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    %         if time_index == 4236
    %             figure
    %             imshow(uint8(SBEV_out))
    %         end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trajectory (수정, 빈칸 찾고 한번에 연산하게 수정)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if SBEV_PARAM.TRAJECTORY.ON == 1 && Target_Shape_Exist_Flag == 1

    for index_traj = 1:length(State_trajectory(1,:))
        if index_traj == length(State_trajectory(1,:))
            a = 1;
        end

        if norm([State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj), State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj)],2) ~= 0 ...
                && State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj) >= SBEV_PARAM.RANGE.X_MIN && State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj) <= SBEV_PARAM.RANGE.X_MAX ...
                && State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj) >= SBEV_PARAM.RANGE.Y_MIN && State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj) <= SBEV_PARAM.RANGE.Y_MAX

            [~,Image_Position_X] = min(abs(State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj) - SBEV_PARAM.RANGE.X_RANGE));
            [~,Image_Position_Y] = min(abs(State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj) - SBEV_PARAM.RANGE.Y_RANGE));

            Target_Trajectory_Exist_Flag = 1;

            if SBEV_PARAM.TRAJECTORY_THREAT
                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
            elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT
                if index_traj == length(State_trajectory(1,:))
                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                end
            end

            if SBEV_PARAM.COLLISION_PROBABILITY.ON
                [~, Collision_Probability_uint8] = min(abs(State_trajectory(TRAJ_PARAM.COLLISION_PROBABILITY, index_traj) - SBEV_PARAM.RANGE.COLLISION_PROBABILITY_RANGE));
            end

%             if SBEV_PARAM.TRAJECTORY.FADING.ON
%                 TRAJECTORY_FADING_FACTOR_STEP = round(SBEV_PARAM.RGB_MAX - SBEV_PARAM.TRAJECTORY.FADING.TRAJECTORY_INITIAL_FADING_FACTOR)/(length(State_trajectory(1,:)) - 1);
%             end

            if SBEV_PARAM.RGB_IMAGE == 1
                if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1
                    if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0
                        for i_ch = 1:CH_LENGTH
                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                    SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                    SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;

                                elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                    if index_traj ~= length(State_trajectory(1,:))
                                        SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                    elseif index_traj == length(State_trajectory(1,:))
                                        SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                    end
                                end

                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;

                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_X % VX
                                [~,VX_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_X, index_traj) - SBEV_PARAM.RANGE.VX_RANGE));
                                SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VX_uint8-1;

                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_Y % VY
                                [~,VY_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_Y, index_traj) - SBEV_PARAM.RANGE.VY_RANGE));
                                SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VY_uint8-1;
                            end
                        end

                    elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1

                        for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                            for i_info = 1:CH_LENGTH
                                if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                    if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                        SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                    elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                        SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = I_LAT_uint8-1;

                                    elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                        if index_traj ~= length(State_trajectory(1,:))
                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                        elseif index_traj == length(State_trajectory(1,:))
                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                        end
                                    end

                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                    SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = I_LAT_uint8-1;

                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.REL_VEL_X % VX
                                    [~,VX_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_X, index_traj) - SBEV_PARAM.RANGE.VX_RANGE));
                                    SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = VX_uint8-1;

                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.REL_VEL_Y % VY
                                    [~,VY_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_Y, index_traj) - SBEV_PARAM.RANGE.VY_RANGE));
                                    SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = VY_uint8-1;
                                end
                            end
                        end
                    end

                elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                    if SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 0
                        for i_ch = 1:CH_LENGTH
                            if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position

                                if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                    if SBEV_PARAM.TRAJECTORY.FADING.ON
                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                        fading_factor_step = SBEV_PARAM.TRAJECTORY.FADING.FADING_FACTOR.FADING_VALUE + ( index_traj - 1 ) * SBEV_PARAM.TRAJECTORY.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                        SBEV_out(Image_Position_X, Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * ( 1 - fading_factor_step );
                                    else
                                        SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                    end

                                elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                    if SBEV_PARAM.TRAJECTORY.FADING.ON
                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - ( I_LAT_uint8 - 1 );
                                        fading_factor_step = SBEV_PARAM.TRAJECTORY.FADING.FADING_FACTOR.FADING_VALUE + ( index_traj - 1 ) * SBEV_PARAM.TRAJECTORY.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                        SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = ( I_LAT_uint8 - 1 ) + tmp_white_vector * ( 1 - fading_factor_step );
                                    else
                                        SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8 - 1;
                                    end

                                elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                    if index_traj ~= length(State_trajectory(1,:))
                                        SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                    elseif index_traj == length(State_trajectory(1,:))
                                        SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                    end
                                end

                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;

                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LONG % I_LONG
                                [~,I_LONG_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LONG, index_traj) - SBEV_PARAM.RANGE.I_LONG_RANGE));
                                SBEV_out(Image_Position_X, Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LONG_uint8-1;

                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_X % VX
                                [~,VX_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_X, index_traj) - SBEV_PARAM.RANGE.VX_RANGE));
                                SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VX_uint8-1;

                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_Y % VY
                                [~,VY_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_Y, index_traj) - SBEV_PARAM.RANGE.VY_RANGE));
                                SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VY_uint8-1;

                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.RSS_X % RSS_X
                                [~,RSS_X_uint8] = min(abs(State_trajectory(TRAJ_PARAM.RSS_X, index_traj) - SBEV_PARAM.RANGE.RSS_X_RANGE));
                                SBEV_out(Image_Position_X, Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = RSS_X_uint8-1;

                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.RSS_Y % RSS_Y
                                [~,RSS_Y_uint8] = min(abs(State_trajectory(TRAJ_PARAM.RSS_Y, index_traj) - SBEV_PARAM.RANGE.RSS_Y_RANGE));
                                SBEV_out(Image_Position_X, Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = RSS_Y_uint8-1;

                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY

                                if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                    if SBEV_PARAM.TRAJECTORY.FADING.ON
                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                        fading_factor_step = SBEV_PARAM.TRAJECTORY.FADING.FADING_FACTOR.FADING_VALUE + ( index_traj - 1 ) * SBEV_PARAM.TRAJECTORY.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                        SBEV_out(Image_Position_X, Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * ( 1 - fading_factor_step );
                                    else
                                        SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                    end                                    
                                else
                                    if SBEV_PARAM.TRAJECTORY.FADING.ON
                                        tmp_white_vector = SBEV_PARAM.RGB_MAX - ( Collision_Probability_uint8 - 1 );
                                        fading_factor_step = SBEV_PARAM.TRAJECTORY.FADING.FADING_FACTOR.FADING_VALUE + ( index_traj - 1 ) * SBEV_PARAM.TRAJECTORY.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                        SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = ( Collision_Probability_uint8 - 1 ) + tmp_white_vector * ( 1 - fading_factor_step );
                                    else
                                        SBEV_out(Image_Position_X, Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;
                                    end                                    
                                end

                            elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA

                                if SBEV_PARAM.TRAJECTORY.FADING.ON
                                    tmp_white_vector = SBEV_PARAM.RGB_MAX - SBEV_PARAM.RGB_MIN;
                                    fading_factor_step = SBEV_PARAM.TRAJECTORY.FADING.FADING_FACTOR.FADING_VALUE + ( index_traj - 1 ) * SBEV_PARAM.TRAJECTORY.FADING.FADING_FACTOR.FADING_FACTOR_STEP_VALUE;
                                    SBEV_out(Image_Position_X, Image_Position_Y, SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN + tmp_white_vector * ( 1 - fading_factor_step );
                                else
                                    SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                end
                            end
                        end

                    elseif SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED == 1
                        for i_SBEV = 1:SBEV_PARAM.IMAGE_CHANNEL/3
                            for i_info = 1:CH_LENGTH
                                if SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                    if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                        SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                                    elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                        SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = I_LAT_uint8-1;

                                    elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                        if index_traj ~= length(State_trajectory(1,:))
                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                        elseif index_traj == length(State_trajectory(1,:))
                                            SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                        end
                                    end

                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                    [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                                    SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = I_LAT_uint8-1;

                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.I_LONG % I_LONG
                                    [~,I_LONG_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LONG, index_traj) - SBEV_PARAM.RANGE.I_LONG_RANGE));
                                    SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = I_LONG_uint8-1;

                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.REL_VEL_X % VX
                                    [~,VX_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_X, index_traj) - SBEV_PARAM.RANGE.VX_RANGE));
                                    SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = VX_uint8-1;

                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.REL_VEL_Y % VY
                                    [~,VY_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_Y, index_traj) - SBEV_PARAM.RANGE.VY_RANGE));
                                    SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = VY_uint8-1;

                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.RSS_X % RSS_X
                                    [~,RSS_X_uint8] = min(abs(State_trajectory(TRAJ_PARAM.RSS_X, index_traj) - SBEV_PARAM.RANGE.RSS_X_RANGE));
                                    SBEV_out(Image_Position_X, Image_Position_Y, 3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = RSS_X_uint8-1;

                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.RSS_Y % RSS_Y
                                    [~,RSS_Y_uint8] = min(abs(State_trajectory(TRAJ_PARAM.RSS_Y, index_traj) - SBEV_PARAM.RANGE.RSS_Y_RANGE));
                                    SBEV_out(Image_Position_X, Image_Position_Y, 3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = RSS_Y_uint8-1;

                                elseif SBEV_PARAM.CHANNEL_INFO(i_info).TRAJ_STATE == TRAJ_PARAM.NA
                                    SBEV_out(Image_Position_X,Image_Position_Y,3*(i_SBEV - 1) + SBEV_PARAM.CHANNEL_INFO(i_info).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                end
                            end
                        end
                    end
                end

            elseif SBEV_PARAM.GRAY_IMAGE == 1
                if SBEV_PARAM.BACKGROUND_COLOR_BLACK == 1

                    for i_ch = 1:CH_LENGTH
                        if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                            if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;

                            elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;

                            elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                if index_traj ~= length(State_trajectory(1,:))
                                    SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX;
                                elseif index_traj == length(State_trajectory(1,:))
                                    SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                            [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                            SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_X % VX
                            [~,VX_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_X, index_traj) - SBEV_PARAM.RANGE.VX_RANGE));
                            SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VX_uint8-1;

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.REL_VEL_Y % VY
                            [~,VY_uint8] = min(abs(State_trajectory(TRAJ_PARAM.REL_VEL_Y, index_traj) - SBEV_PARAM.RANGE.VY_RANGE));
                            SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = VY_uint8-1;
                        end
                    end

                elseif SBEV_PARAM.BACKGROUND_COLOR_WHITE == 1
                    for i_ch = 1:CH_LENGTH
                        if SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                            if SBEV_PARAM.TRAJECTORY_POSITION == 1
                                SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;

                            elseif SBEV_PARAM.TRAJECTORY_THREAT == 1
                                SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX - (I_LAT_uint8-1);

                            elseif SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT == 1
                                if index_traj ~= length(State_trajectory(1,:))
                                    SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MIN;
                                elseif index_traj == length(State_trajectory(1,:))
                                    SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX - (I_LAT_uint8-1);
                                end
                            end

                        elseif SBEV_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                            [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - SBEV_PARAM.RANGE.I_LAT_RANGE));
                            SBEV_out(Image_Position_X,Image_Position_Y,SBEV_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = SBEV_PARAM.RGB_MAX - (I_LAT_uint8-1);
                        end
                    end
                end
            end
        end
    end
end


if Target_Exist_in_Input_SBEV == 0 && Target_Shape_Exist_Flag == 0
    SBEV_out = empty_SBEV; % delete SBEV(with only lane mark, without target)
end



% if time_index == 4236
%     figure
%     imshow(uint8(SBEV_out))
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_total, y_total] = plotLineLow(x0, y0, x1, y1)

if abs(x1 - x0 + 1) > abs(y1 - y0 + 1)
    x_total = zeros(abs(x1 - x0 + 1),1);
    y_total = x_total;
elseif abs(x1 - x0 + 1) < abs(y1 - y0 + 1)
    x_total = zeros(abs(y1 - y0 + 1),1);
    y_total = x_total;
else
    x_total = zeros(abs(x1 - x0 + 1),1);
    y_total = zeros(abs(x1 - x0 + 1),1);
end

dx = x1 - x0;
dy = y1 - y0;
yi = 1;

if dy < 0
    yi = -1;
    dy = -dy;
end

D = 2*dy - dx;
y = y0;

i_internal = 0;
for x = x0:x1
    i_internal = i_internal + 1;
%     plot(x,y,'o')
    x_total(i_internal) = x;
    y_total(i_internal) = y;
    
    if D > 0
        y = y + yi;
        D = D + (2*(dy - dx));
    else
        D = D + 2*dy;
    end
    
end
end


function [x_total, y_total] = plotLineHigh(x0, y0, x1, y1)

if abs(x1 - x0 + 1) > abs(y1 - y0 + 1)
    x_total = zeros(abs(x1 - x0 + 1),1);
    y_total = x_total;
elseif abs(x1 - x0 + 1) < abs(y1 - y0 + 1)
    x_total = zeros(abs(y1 - y0 + 1),1);
    y_total = x_total;
else
    x_total = zeros(abs(x1 - x0 + 1),1);
    y_total = zeros(abs(x1 - x0 + 1),1);
end

dx = x1 - x0;
dy = y1 - y0;
xi = 1;

if dx < 0
    xi = -1;
    dx = -dx;
end

D = 2*dx - dy;
x = x0;

i_internal = 0;
for y = y0:y1
    i_internal = i_internal + 1;
%     plot(x,y,'o')
    x_total(i_internal) = x;
    y_total(i_internal) = y;
    
    if D > 0
        x = x + xi;
        D = D + (2*(dx - dy));
    else
        D = D + 2*dx;
    end
    
end
end

function [x_total, y_total] = plotLine(x0, y0, x1, y1)

if abs(x1 - x0 + 1) > abs(y1 - y0 + 1)
    x_total = zeros(abs(x1 - x0 + 1),1);
    y_total = x_total;
elseif abs(x1 - x0 + 1) < abs(y1 - y0 + 1)
    x_total = zeros(abs(y1 - y0 + 1),1);
    y_total = x_total;
else
    x_total = zeros(abs(x1 - x0 + 1),1);
    y_total = zeros(abs(x1 - x0 + 1),1);
end

if abs(y1 - y0) < abs(x1 - x0)
    if x0 > x1
        [x_total, y_total] = plotLineLow(x1, y1, x0, y0);
    else
        [x_total, y_total] = plotLineLow(x0, y0, x1, y1);
    end
else
    if y0 > y1
        [x_total, y_total] = plotLineHigh(x1, y1, x0, y0);
    else
        [x_total, y_total]= plotLineHigh(x0, y0, x1, y1);
    end
end
end
end