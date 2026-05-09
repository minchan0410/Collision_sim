function [DSM_out, LANE_MARK_FLAG_out, EGO_SHAPE_FLAG_out] = DSMPredictionPointRG3(DSM_in, State_trajectory, laneInfoL, laneInfoR,...
    Target_X_pred, LANE_MARK_FLAG_in, EGO_SHAPE_FLAG_in, DSM_PARAM, TRAJ_PARAM, FRONT_VISION_LANE, EGO_VEHICLE, TRACKING, SAMPLE_TIME)
% DSMPredictionPointRG3 function generates DSM with predicted state as point in RGB image for RG3 data
%
% [DSM_out, LANE_MARK_FLAG_out, EGO_SHAPE_FLAG_out] = DSMPredictionPointRG3(DSM_in, State_trajectory, laneInfoL, laneInfoR,...
%    Target_X_pred, LANE_MARK_FLAG_in, EGO_SHAPE_FLAG_in, DSM_PARAM, TRAJ_PARAM, FRONT_VISION_LANE, EGO_VEHICLE, TRACKING, SAMPLE_TIME)
%
% DSM_out {double}              : generated DSM
% LANE_MARK_FLAG_out {double}   : flag for lane mark in DSM
% EGO_SHAPE_FLAG_out {double}   : flag for ego vehicle in DSM
%
% DSM_in {double}               : initialized DSM
% State_trajectory {double}     : trajectory
% laneInfoL {double}            : information for left lane mark
% laneInfoR {double}            : information for left lane mark
% Target_X_pred {double}        : predicted state of surrounding object
% LANE_MARK_FLAG_in {double}    : flag for lane mark in DSM
% EGO_SHAPE_FLAG_in {double}    : flag for ego vehicle in DSM
% DSM_PARAM {struct}            : parameters for generation of DSM
% TRAJ_PARAM {struct}           : parameters for state of trajectory
% EGO_VEHICLE {struct}          : parameters for ego vehicle
% TRACKING {struct}             : parameters for predicted state
% SAMPLE_TIME {double}          : sample time of data

Lane_Width = 3.5;

if DSM_PARAM.BACKGROUND_COLOR_BLACK == 1
    empty_DSM = zeros(DSM_PARAM.IMAGE_HEIGHT, DSM_PARAM.IMAGE_WIDTH, DSM_PARAM.IMAGE_CHANNEL);
elseif DSM_PARAM.BACKGROUND_COLOR_WHITE == 1
    empty_DSM = 255*ones(DSM_PARAM.IMAGE_HEIGHT, DSM_PARAM.IMAGE_WIDTH, DSM_PARAM.IMAGE_CHANNEL);
end

LANE_MARK_FLAG_out = 0;
EGO_SHAPE_FLAG_out = 0;
Target_Shape_Exist_Flag = 0;
Target_Trajectory_Exist_Flag = 0;
Target_Exist_in_Input_SBEV = 0;

DSM_out = DSM_in;
CH_LENGTH = length(DSM_PARAM.CHANNEL_INFO);

if ~isequal(empty_DSM, DSM_out)
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

if ( (min(target_y) >= DSM_PARAM.RANGE.Y_MIN && min(target_y) <= DSM_PARAM.RANGE.Y_MAX) || (max(target_y) >= DSM_PARAM.RANGE.Y_MIN && max(target_y) <= DSM_PARAM.RANGE.Y_MAX)) ...
        && ((min(target_x) >= DSM_PARAM.RANGE.X_MIN && min(target_x) <= DSM_PARAM.RANGE.X_MAX) || (max(target_x) >= DSM_PARAM.RANGE.X_MIN && max(target_x) <= DSM_PARAM.RANGE.X_MAX))

    Target_Shape_Exist_Flag = 1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lane Mark
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if DSM_PARAM.LANE_MARK.ON
    if LANE_MARK_FLAG_in == 0 && Target_Shape_Exist_Flag == 1
        num_left_line = 1;
        num_right_line = 1;

        HIGH = FRONT_VISION_LANE.DESCRIPTION_CONFIDENCE.PREDICTION;

        confidence_left = laneInfoL(FRONT_VISION_LANE.MEASURE.CONFIDENCE, 1);
        confidence_right = laneInfoR(FRONT_VISION_LANE.MEASURE.CONFIDENCE, 1);

        if laneInfoL(FRONT_VISION_LANE.MEASURE.VIEWRANGE, 1) > DSM_PARAM.RANGE.X_MAX
            x_lane_left = [laneInfoL(FRONT_VISION_LANE.MEASURE.VIEWRANGE_START, 1):(DSM_PARAM.RANGE.X_MAX - DSM_PARAM.RANGE.X_MIN)/(DSM_PARAM.IMAGE_HEIGHT - 1):DSM_PARAM.RANGE.X_MAX];
        else
            x_lane_left = [laneInfoL(FRONT_VISION_LANE.MEASURE.VIEWRANGE_START, 1):(DSM_PARAM.RANGE.X_MAX - DSM_PARAM.RANGE.X_MIN)/(DSM_PARAM.IMAGE_HEIGHT - 1):laneInfoL(FRONT_VISION_LANE.MEASURE.VIEWRANGE, 1)];
        end
        
        if laneInfoR(FRONT_VISION_LANE.MEASURE.VIEWRANGE, 1) > DSM_PARAM.RANGE.X_MAX
            x_lane_right = [laneInfoR(FRONT_VISION_LANE.MEASURE.VIEWRANGE_START, 1):(DSM_PARAM.RANGE.X_MAX - DSM_PARAM.RANGE.X_MIN)/(DSM_PARAM.IMAGE_HEIGHT - 1):DSM_PARAM.RANGE.X_MAX];
        else
            x_lane_right = [laneInfoR(FRONT_VISION_LANE.MEASURE.VIEWRANGE_START, 1):(DSM_PARAM.RANGE.X_MAX - DSM_PARAM.RANGE.X_MIN)/(DSM_PARAM.IMAGE_HEIGHT - 1):laneInfoR(FRONT_VISION_LANE.MEASURE.VIEWRANGE, 1)];
        end

        laneCoeff_left = [laneInfoL(FRONT_VISION_LANE.PREPROCESSING.DISTANCE, 1), laneInfoL(FRONT_VISION_LANE.PREPROCESSING.ROAD_SLOPE, 1), ...
            laneInfoL(FRONT_VISION_LANE.PREPROCESSING.CURVATURE, 1), laneInfoL(FRONT_VISION_LANE.PREPROCESSING.CURVATURE_RATE, 1)];
        
        laneCoeff_right = [laneInfoR(FRONT_VISION_LANE.PREPROCESSING.DISTANCE, 1), laneInfoR(FRONT_VISION_LANE.PREPROCESSING.ROAD_SLOPE, 1), ...
            laneInfoR(FRONT_VISION_LANE.PREPROCESSING.CURVATURE, 1), laneInfoR(FRONT_VISION_LANE.PREPROCESSING.CURVATURE_RATE, 1)];

        if confidence_left >= HIGH
            
            i = 0;
            while i < num_left_line
                tmp_lane_y = 0;
                tmp_lane_y= tmp_lane_y + laneCoeff_left(4)*x_lane_left.^3+...
                    laneCoeff_left(3)*x_lane_left.^2+...
                    laneCoeff_left(2)*x_lane_left+...
                    laneCoeff_left(1);
                
                tmp_lane_y = tmp_lane_y + i*Lane_Width;
                
                for i_line=1:length(tmp_lane_y)
                    if x_lane_left(i_line) >= DSM_PARAM.RANGE.X_MIN && x_lane_left(i_line) <= DSM_PARAM.RANGE.X_MAX...
                            && tmp_lane_y(i_line) >= DSM_PARAM.RANGE.Y_MIN && tmp_lane_y(i_line) <= DSM_PARAM.RANGE.Y_MAX
                        [~,X_LINE_uint8] = min(abs(x_lane_left(i_line) - DSM_PARAM.RANGE.X_RANGE));
                        [~,Y_LINE_uint8] = min(abs(tmp_lane_y(i_line) - DSM_PARAM.RANGE.Y_RANGE));
                        
                        if DSM_PARAM.BACKGROUND_COLOR_WHITE == 1
                            DSM_out(X_LINE_uint8, Y_LINE_uint8, DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER) = DSM_PARAM.RGB_MAX;

                            if rem(DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 1 % if remainder = 1, R ch -> current channel number +1, +2 = 0
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = DSM_PARAM.RGB_MIN;
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 2) = DSM_PARAM.RGB_MIN;
                            elseif rem(DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 2 % if remainder = 2, G ch -> current channel number -1, +1 = 0
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = DSM_PARAM.RGB_MIN;
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = DSM_PARAM.RGB_MIN;
                            elseif rem(DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 3 % if remainder = 3, B ch -> current channel number -2, -1 = 0
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 2) = DSM_PARAM.RGB_MIN;
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = DSM_PARAM.RGB_MIN;
                            end
                        end
                    end
                end
                
                i = i+1;
            end
            
            LANE_MARK_FLAG_in = 1;
        end
        

        if confidence_right >= HIGH
            i = 0;
            while i < num_right_line
                tmp_lane_y = 0;
                tmp_lane_y= tmp_lane_y + laneCoeff_right(4)*x_lane_right.^3+...
                    laneCoeff_right(3)*x_lane_right.^2+...
                    laneCoeff_right(2)*x_lane_right+...
                    laneCoeff_right(1);
                
                tmp_lane_y = tmp_lane_y - i*Lane_Width;
                
                for i_line=1:length(tmp_lane_y)
                    if x_lane_right(i_line) >= DSM_PARAM.RANGE.X_MIN && x_lane_right(i_line) <= DSM_PARAM.RANGE.X_MAX...
                            && tmp_lane_y(i_line) >= DSM_PARAM.RANGE.Y_MIN && tmp_lane_y(i_line) <= DSM_PARAM.RANGE.Y_MAX
                        [~,X_LINE_uint8] = min(abs(x_lane_right(i_line) - DSM_PARAM.RANGE.X_RANGE));
                        [~,Y_LINE_uint8] = min(abs(tmp_lane_y(i_line) - DSM_PARAM.RANGE.Y_RANGE));
                        
                        if DSM_PARAM.BACKGROUND_COLOR_WHITE == 1
                            DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER) = DSM_PARAM.RGB_MAX;

                            if rem(DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 1 % if remainder = 1, R ch -> current channel number +1, +2 = 0
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = DSM_PARAM.RGB_MIN;
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 2) = DSM_PARAM.RGB_MIN;
                            elseif rem(DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 2 % if remainder = 2, G ch -> current channel number -1, +1 = 0
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = DSM_PARAM.RGB_MIN;
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER + 1) = DSM_PARAM.RGB_MIN;
                            elseif rem(DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER,3) == 3 % if remainder = 3, B ch -> current channel number -2, -1 = 0
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 2) = DSM_PARAM.RGB_MIN;
                                DSM_out(X_LINE_uint8,Y_LINE_uint8,DSM_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER - 1) = DSM_PARAM.RGB_MIN;
                            end
                        end
                    end
                end
                i = i+1;
            end
            LANE_MARK_FLAG_in = 1;
        end

        LANE_MARK_FLAG_out = LANE_MARK_FLAG_in;

    else
        LANE_MARK_FLAG_out = LANE_MARK_FLAG_in;
    end
else
    LANE_MARK_FLAG_out = LANE_MARK_FLAG_in;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ego Shape
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if DSM_PARAM.SHAPE.EGO == 1
    if Target_Shape_Exist_Flag == 1 && EGO_SHAPE_FLAG_in == 0
        
        tmp_ego_y_vertex = [-EGO_VEHICLE.WIDTH/2, -EGO_VEHICLE.WIDTH/2,...
            EGO_VEHICLE.WIDTH/2, EGO_VEHICLE.WIDTH/2, -EGO_VEHICLE.WIDTH/2];
        tmp_ego_x_vertex = [-EGO_VEHICLE.LENGTH, 0, 0, -EGO_VEHICLE.LENGTH, -EGO_VEHICLE.LENGTH];
        
        ego_x_contour_total = zeros(200,1);
        ego_y_contour_total = zeros(200,1);
        
        i_row = 1;
        f_row = 0;
        
        for tmp_i = 1:length(tmp_ego_y_vertex) - 1
            tmp_y_vertex0 = tmp_ego_y_vertex(tmp_i);
            tmp_x_vertex0 = tmp_ego_x_vertex(tmp_i);
            
            tmp_y_vertex1 = tmp_ego_y_vertex(tmp_i+1);
            tmp_x_vertex1 = tmp_ego_x_vertex(tmp_i+1);
            
            [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - DSM_PARAM.RANGE.Y_RANGE));
            [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - DSM_PARAM.RANGE.X_RANGE));
            
            [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - DSM_PARAM.RANGE.Y_RANGE));
            [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - DSM_PARAM.RANGE.X_RANGE));
            
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
        
        if DSM_PARAM.RGB_IMAGE == 1
            if DSM_PARAM.BACKGROUND_COLOR_WHITE == 1
                for tmp_j = 1:length(pixel_info(:,1))
                    if pixel_info(tmp_j,1) ~= 0
                        DSM_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),:) = DSM_PARAM.RGB_MIN;
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target Prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if DSM_PARAM.PREDICTION.TARGET == 1 && Target_Shape_Exist_Flag == 1
    
    ROI_margin2ego = 0;
    
    if ~isequal(empty_DSM, DSM_out)
        
        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - DSM_PARAM.RANGE.I_LAT_RANGE));

        if DSM_PARAM.COLLISION_PROBABILITY.ON
            [~, Collision_Probability_uint8] = min(abs(State_trajectory(TRAJ_PARAM.COLLISION_PROBABILITY, end) - DSM_PARAM.RANGE.COLLISION_PROBABILITY_RANGE));
        end
        
        if sum(Target_X_pred) ~= 0
            predicted_x = squeeze(Target_X_pred(TRACKING.REL_POS_X,1,:)); % x
            predicted_y = squeeze(Target_X_pred(TRACKING.REL_POS_Y,1,:)); % y


            % overlap 허용
            for index_pred = 1:DSM_PARAM.PREDICTION.TARGET_PRED_WINDOW/DSM_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE

                index_pred_detail = round(index_pred*DSM_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);

                % trajectory + hollow shape(last prediction time)

                if index_pred == DSM_PARAM.PREDICTION.TARGET_PRED_WINDOW/DSM_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE

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

                    if ~( all(target_y >= DSM_PARAM.RANGE.Y_MIN) && all(target_y <= DSM_PARAM.RANGE.Y_MAX) && all(target_x >= DSM_PARAM.RANGE.X_MIN) && all(target_x <= DSM_PARAM.RANGE.X_MAX) )

                        vertex_total = zeros(4, 4);

                        vertex_total(1, :) = target_y(1:4) >= DSM_PARAM.RANGE.Y_MIN;
                        vertex_total(2, :) = target_y(1:4) <= DSM_PARAM.RANGE.Y_MAX;
                        vertex_total(3, :) = target_x(1:4) >= DSM_PARAM.RANGE.X_MIN;
                        vertex_total(4, :) = target_x(1:4) <= DSM_PARAM.RANGE.X_MAX;

                        vertex_out_flag = all(vertex_total);

                        if nnz(vertex_out_flag) == 3 % only one vertex out of ROI
                            ONLY_ONE_VERTEX_ROI_OUT_FLAG = 1;
                        elseif nnz(vertex_out_flag) == 2 % two vertex out of ROI
                            TWO_VERTEX_ROI_OUT_FLAG = 1;
                        elseif nnz(vertex_out_flag) == 1 % three vertex out of ROI
                            THREE_VERTEX_ROI_OUT_FLAG = 1;
                        end
                    end

                    if ( (min(target_y) >= DSM_PARAM.RANGE.Y_MIN && min(target_y) <= DSM_PARAM.RANGE.Y_MAX) || (max(target_y) >= DSM_PARAM.RANGE.Y_MIN && max(target_y) <= DSM_PARAM.RANGE.Y_MAX)) ...
                            && ((min(target_x) >= DSM_PARAM.RANGE.X_MIN && min(target_x) <= DSM_PARAM.RANGE.X_MAX) || (max(target_x) >= DSM_PARAM.RANGE.X_MIN && max(target_x) <= DSM_PARAM.RANGE.X_MAX))

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

                                if ~(tmp_y_vertex0 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= DSM_PARAM.RANGE.Y_MAX &&...
                                        tmp_x_vertex0 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= DSM_PARAM.RANGE.X_MAX)

                                    if (tmp_y_vertex1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= DSM_PARAM.RANGE.Y_MAX &&... % next and before vertex in ROI
                                            tmp_x_vertex1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= DSM_PARAM.RANGE.X_MAX) && ...
                                            (tmp_y_vertex_1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= DSM_PARAM.RANGE.Y_MAX &&...
                                            tmp_x_vertex_1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= DSM_PARAM.RANGE.X_MAX)

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

                                        if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                            y_cross_1 = DSM_PARAM.RANGE.Y_MIN;
                                            x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                                        elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                            y_cross_1 = DSM_PARAM.RANGE.Y_MAX;
                                            x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                                        elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                            x_cross_1 = DSM_PARAM.RANGE.X_MIN;
                                            y_cross_1 = m_1*x_cross_1 + tmp_base_1;

                                        elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                            x_cross_1 = DSM_PARAM.RANGE.X_MAX;
                                            y_cross_1 = m_1*x_cross_1 + tmp_base_1;
                                        end


                                        % current ~ next vertex
                                        m_2 = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                        tmp_base_2 = tmp_y_vertex1 - m_2*tmp_x_vertex1;

                                        if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                            y_cross_2 = DSM_PARAM.RANGE.Y_MIN;
                                            x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                                        elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                            y_cross_2 = DSM_PARAM.RANGE.Y_MAX;
                                            x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                                        elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                            x_cross_2 = DSM_PARAM.RANGE.X_MIN;
                                            y_cross_2 = m_2*x_cross_2 + tmp_base_2;

                                        elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                            x_cross_2 = DSM_PARAM.RANGE.X_MAX;
                                            y_cross_2 = m_2*x_cross_2 + tmp_base_2;
                                        end

                                        target_y_correction(tmp_i) = y_cross_1;
                                        target_x_correction(tmp_i) = x_cross_1;

                                        target_y_correction(tmp_i+1) = y_cross_2;
                                        target_x_correction(tmp_i+1) = x_cross_2;


                                    elseif (tmp_y_vertex_1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= DSM_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                            tmp_x_vertex_1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= DSM_PARAM.RANGE.X_MAX)

                                        if tmp_x_vertex0 == tmp_x_vertex_1
                                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                                x_cross = tmp_x_vertex0;

                                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                                x_cross = tmp_x_vertex0;
                                            end

                                        elseif tmp_y_vertex0 == tmp_y_vertex_1
                                            if tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                                y_cross = tmp_y_vertex0;

                                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                                x_cross = DSM_PARAM.RANGE.X_MAX;
                                                y_cross = tmp_y_vertex0;
                                            end
                                        else
                                            m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                            tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                                x_cross = (y_cross - tmp_base)/m;

                                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                                x_cross = (y_cross - tmp_base)/m;

                                            elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                                y_cross = m*x_cross + tmp_base;

                                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                                x_cross = DSM_PARAM.RANGE.X_MAX;
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

                                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - DSM_PARAM.RANGE.Y_RANGE));
                                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - DSM_PARAM.RANGE.X_RANGE));

                                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - DSM_PARAM.RANGE.Y_RANGE));
                                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - DSM_PARAM.RANGE.X_RANGE));

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

                                if ~(tmp_y_vertex0 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= DSM_PARAM.RANGE.Y_MAX &&...
                                        tmp_x_vertex0 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= DSM_PARAM.RANGE.X_MAX)

                                    if (tmp_y_vertex1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= DSM_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                                            tmp_x_vertex1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= DSM_PARAM.RANGE.X_MAX)

                                        if tmp_x_vertex0 == tmp_x_vertex1
                                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                                x_cross = tmp_x_vertex0;

                                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                                x_cross = tmp_x_vertex0;
                                            end

                                        elseif tmp_y_vertex0 == tmp_y_vertex1
                                            if tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                                y_cross = tmp_y_vertex0;
                                                x_cross = DSM_PARAM.RANGE.X_MIN;

                                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                                y_cross = tmp_y_vertex0;
                                                x_cross = DSM_PARAM.RANGE.X_MAX;
                                            end
                                        else
                                            m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                            tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                                x_cross = (y_cross - tmp_base)/m;

                                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                                x_cross = (y_cross - tmp_base)/m;

                                            elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                                y_cross = m*x_cross + tmp_base;

                                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                                x_cross = DSM_PARAM.RANGE.X_MAX;
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


                                    elseif (tmp_y_vertex_1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= DSM_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                            tmp_x_vertex_1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= DSM_PARAM.RANGE.X_MAX)

                                        if tmp_x_vertex0 == tmp_x_vertex_1
                                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                                x_cross = tmp_x_vertex0;

                                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                                x_cross = tmp_x_vertex0;
                                            end

                                        elseif tmp_y_vertex0 == tmp_y_vertex_1
                                            if tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                                y_cross = tmp_y_vertex0;

                                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                                x_cross = DSM_PARAM.RANGE.X_MAX;
                                                y_cross = tmp_y_vertex0;
                                            end
                                        else
                                            m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                            tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                                x_cross = (y_cross - tmp_base)/m;

                                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                                x_cross = (y_cross - tmp_base)/m;

                                            elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                                y_cross = m*x_cross + tmp_base;

                                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                                x_cross = DSM_PARAM.RANGE.X_MAX;
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

                                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - DSM_PARAM.RANGE.Y_RANGE));
                                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - DSM_PARAM.RANGE.X_RANGE));

                                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - DSM_PARAM.RANGE.Y_RANGE));
                                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - DSM_PARAM.RANGE.X_RANGE));

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

                                if ~(tmp_y_vertex0 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= DSM_PARAM.RANGE.Y_MAX &&...
                                        tmp_x_vertex0 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= DSM_PARAM.RANGE.X_MAX)

                                    if (tmp_y_vertex1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= DSM_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                                            tmp_x_vertex1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= DSM_PARAM.RANGE.X_MAX)

                                        if tmp_x_vertex0 == tmp_x_vertex1
                                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                                x_cross = tmp_x_vertex0;

                                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                                x_cross = tmp_x_vertex0;
                                            end

                                        elseif tmp_y_vertex0 == tmp_y_vertex1
                                            if tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                                y_cross = tmp_y_vertex0;
                                                x_cross = DSM_PARAM.RANGE.X_MIN;

                                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                                y_cross = tmp_y_vertex0;
                                                x_cross = DSM_PARAM.RANGE.X_MAX;
                                            end
                                        else
                                            m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                                            tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                                x_cross = (y_cross - tmp_base)/m;

                                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                                x_cross = (y_cross - tmp_base)/m;

                                            elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                                y_cross = m*x_cross + tmp_base;

                                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                                x_cross = DSM_PARAM.RANGE.X_MAX;
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


                                    elseif (tmp_y_vertex_1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= DSM_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                                            tmp_x_vertex_1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= DSM_PARAM.RANGE.X_MAX)

                                        if tmp_x_vertex0 == tmp_x_vertex_1
                                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                                x_cross = tmp_x_vertex0;

                                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                                x_cross = tmp_x_vertex0;
                                            end

                                        elseif tmp_y_vertex0 == tmp_y_vertex_1
                                            if tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                                y_cross = tmp_y_vertex0;

                                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                                x_cross = DSM_PARAM.RANGE.X_MAX;
                                                y_cross = tmp_y_vertex0;
                                            end
                                        else
                                            m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                                            tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                                x_cross = (y_cross - tmp_base)/m;

                                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                                x_cross = (y_cross - tmp_base)/m;

                                            elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                                y_cross = m*x_cross + tmp_base;

                                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                                x_cross = DSM_PARAM.RANGE.X_MAX;
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
                                if ( target_y_correction(vertex_index_beforeCurrentNext_all_out) >= DSM_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= DSM_PARAM.RANGE.Y_MAX ) || ...
                                        ( target_x_correction(vertex_index_beforeCurrentNext_all_out) >= DSM_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= DSM_PARAM.RANGE.X_MAX )

                                    if vertex_index_beforeCurrentNext_all_out == 1
                                        target_y_correction(1) = target_y_correction(4);
                                        target_x_correction(1) = target_x_correction(4);
                                    else
                                        target_y_correction(vertex_index_beforeCurrentNext_all_out) = target_y_correction(vertex_index_beforeCurrentNext_all_out-1);
                                        target_x_correction(vertex_index_beforeCurrentNext_all_out) = target_x_correction(vertex_index_beforeCurrentNext_all_out-1);
                                    end

                                    % 모두 벗어나는 경우
                                elseif ~(target_y_correction(vertex_index_beforeCurrentNext_all_out) >= DSM_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= DSM_PARAM.RANGE.Y_MAX &&...
                                        target_x_correction(vertex_index_beforeCurrentNext_all_out) >= DSM_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= DSM_PARAM.RANGE.X_MAX)

                                    if target_y_correction(vertex_index_beforeCurrentNext_all_out) < DSM_PARAM.RANGE.Y_MIN
                                        target_y_correction(vertex_index_beforeCurrentNext_all_out) = DSM_PARAM.RANGE.Y_MIN;
                                    elseif target_y_correction(vertex_index_beforeCurrentNext_all_out) > DSM_PARAM.RANGE.Y_MAX
                                        target_y_correction(vertex_index_beforeCurrentNext_all_out) = DSM_PARAM.RANGE.Y_MAX;
                                    end

                                    if target_x_correction(vertex_index_beforeCurrentNext_all_out) < DSM_PARAM.RANGE.X_MIN
                                        target_x_correction(vertex_index_beforeCurrentNext_all_out) = DSM_PARAM.RANGE.X_MIN;
                                    elseif target_x_correction(vertex_index_beforeCurrentNext_all_out) > DSM_PARAM.RANGE.X_MAX
                                        target_x_correction(vertex_index_beforeCurrentNext_all_out) = DSM_PARAM.RANGE.X_MAX;
                                    end
                                end
                            end

                            for tmp_i = 1:length(target_y_correction) - 1
                                tmp_y_vertex0 = target_y_correction(tmp_i);
                                tmp_x_vertex0 = target_x_correction(tmp_i);

                                tmp_y_vertex1 = target_y_correction(tmp_i+1);
                                tmp_x_vertex1 = target_x_correction(tmp_i+1);

                                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - DSM_PARAM.RANGE.Y_RANGE));
                                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - DSM_PARAM.RANGE.X_RANGE));

                                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - DSM_PARAM.RANGE.Y_RANGE));
                                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - DSM_PARAM.RANGE.X_RANGE));

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

                                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - DSM_PARAM.RANGE.Y_RANGE));
                                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - DSM_PARAM.RANGE.X_RANGE));

                                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - DSM_PARAM.RANGE.Y_RANGE));
                                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - DSM_PARAM.RANGE.X_RANGE));

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
                        if DSM_PARAM.RGB_IMAGE == 1
                            if DSM_PARAM.BACKGROUND_COLOR_WHITE == 1
                                for i_ch = 1:CH_LENGTH
                                    if DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                        for tmp_j = 1:length(x_contour_total)
                                            DSM_out(x_contour_total(tmp_j), y_contour_total(tmp_j), DSM_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = DSM_PARAM.RGB_MIN;
                                        end

                                    elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                        for tmp_j = 1:length(x_contour_total)
                                            DSM_out(x_contour_total(tmp_j), y_contour_total(tmp_j), DSM_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;
                                        end

                                    elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY
                                        for tmp_j = 1:length(x_contour_total)
                                            DSM_out(x_contour_total(tmp_j), y_contour_total(tmp_j), DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;
                                        end

                                    elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA
                                        for tmp_j = 1:length(x_contour_total)
                                            DSM_out(x_contour_total(tmp_j), y_contour_total(tmp_j), DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = DSM_PARAM.RGB_MIN;
                                        end
                                    end
                                end
                            end
                        end
                    end
                else
                    target_y = predicted_y(index_pred_detail);
                    target_x = predicted_x(index_pred_detail);

                    if target_y >= DSM_PARAM.RANGE.Y_MIN && target_y <= DSM_PARAM.RANGE.Y_MAX ...
                            && target_x >= DSM_PARAM.RANGE.X_MIN && target_x <= DSM_PARAM.RANGE.X_MAX

                        [~,Image_Position_X] = min(abs(target_x - DSM_PARAM.RANGE.X_RANGE));
                        [~,Image_Position_Y] = min(abs(target_y - DSM_PARAM.RANGE.Y_RANGE));

                        if DSM_PARAM.RGB_IMAGE == 1
                            if DSM_PARAM.BACKGROUND_COLOR_WHITE == 1

                                for i_ch = 1:CH_LENGTH
                                    if DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                                        DSM_out(Image_Position_X,Image_Position_Y, DSM_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = DSM_PARAM.RGB_MIN;

                                    elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                                        DSM_out(Image_Position_X,Image_Position_Y, DSM_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER) = I_LAT_uint8-1;

                                    elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY
                                        DSM_out(Image_Position_X,Image_Position_Y, DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;

                                    elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA
                                        DSM_out(Image_Position_X,Image_Position_Y, DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = DSM_PARAM.RGB_MIN;
                                    end
                                end
                            end
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
if Target_Shape_Exist_Flag == 1
    
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

    if ~( all(target_y >= DSM_PARAM.RANGE.Y_MIN) && all(target_y <= DSM_PARAM.RANGE.Y_MAX) && all(target_x >= DSM_PARAM.RANGE.X_MIN) && all(target_x <= DSM_PARAM.RANGE.X_MAX) )
        vertex_total = zeros(4, 4);

        vertex_total(1, :) = target_y(1:4) >= DSM_PARAM.RANGE.Y_MIN;
        vertex_total(2, :) = target_y(1:4) <= DSM_PARAM.RANGE.Y_MAX;
        vertex_total(3, :) = target_x(1:4) >= DSM_PARAM.RANGE.X_MIN;
        vertex_total(4, :) = target_x(1:4) <= DSM_PARAM.RANGE.X_MAX;

        vertex_out_flag = all(vertex_total);

        if nnz(vertex_out_flag) == 3 % only one vertex out of ROI
            ONLY_ONE_VERTEX_ROI_OUT_FLAG = 1;
        elseif nnz(vertex_out_flag) == 2 % two vertex out of ROI
            TWO_VERTEX_ROI_OUT_FLAG = 1;
        elseif nnz(vertex_out_flag) == 1 % three vertex out of ROI
            THREE_VERTEX_ROI_OUT_FLAG = 1;
        end
    end
    
    
    if ( (min(target_y) >= DSM_PARAM.RANGE.Y_MIN && min(target_y) <= DSM_PARAM.RANGE.Y_MAX) || (max(target_y) >= DSM_PARAM.RANGE.Y_MIN && max(target_y) <= DSM_PARAM.RANGE.Y_MAX)) ...
            && ((min(target_x) >= DSM_PARAM.RANGE.X_MIN && min(target_x) <= DSM_PARAM.RANGE.X_MAX) || (max(target_x) >= DSM_PARAM.RANGE.X_MIN && max(target_x) <= DSM_PARAM.RANGE.X_MAX))

        Target_Shape_Exist_Flag = 1;

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

                if ~(tmp_y_vertex0 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= DSM_PARAM.RANGE.Y_MAX &&...
                        tmp_x_vertex0 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= DSM_PARAM.RANGE.X_MAX)

                    if (tmp_y_vertex1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= DSM_PARAM.RANGE.Y_MAX &&... % next and before vertex in ROI
                            tmp_x_vertex1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= DSM_PARAM.RANGE.X_MAX) && ...
                            (tmp_y_vertex_1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= DSM_PARAM.RANGE.Y_MAX &&...
                            tmp_x_vertex_1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= DSM_PARAM.RANGE.X_MAX)

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

                        if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                            y_cross_1 = DSM_PARAM.RANGE.Y_MIN;
                            x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                        elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                            y_cross_1 = DSM_PARAM.RANGE.Y_MAX;
                            x_cross_1 = (y_cross_1 - tmp_base_1)/m_1;

                        elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                            x_cross_1 = DSM_PARAM.RANGE.X_MIN;
                            y_cross_1 = m_1*x_cross_1 + tmp_base_1;

                        elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                            x_cross_1 = DSM_PARAM.RANGE.X_MAX;
                            y_cross_1 = m_1*x_cross_1 + tmp_base_1;
                        end


                        % current ~ next vertex
                        m_2 = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                        tmp_base_2 = tmp_y_vertex1 - m_2*tmp_x_vertex1;

                        if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                            y_cross_2 = DSM_PARAM.RANGE.Y_MIN;
                            x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                        elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                            y_cross_2 = DSM_PARAM.RANGE.Y_MAX;
                            x_cross_2 = (y_cross_2 - tmp_base_2)/m_2;

                        elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                            x_cross_2 = DSM_PARAM.RANGE.X_MIN;
                            y_cross_2 = m_2*x_cross_2 + tmp_base_2;

                        elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                            x_cross_2 = DSM_PARAM.RANGE.X_MAX;
                            y_cross_2 = m_2*x_cross_2 + tmp_base_2;
                        end

                        target_y_correction(tmp_i) = y_cross_1;
                        target_x_correction(tmp_i) = x_cross_1;

                        target_y_correction(tmp_i+1) = y_cross_2;
                        target_x_correction(tmp_i+1) = x_cross_2;


                    elseif (tmp_y_vertex_1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= DSM_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                            tmp_x_vertex_1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= DSM_PARAM.RANGE.X_MAX)

                        if tmp_x_vertex0 == tmp_x_vertex_1
                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                x_cross = tmp_x_vertex0;

                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                x_cross = tmp_x_vertex0;
                            end

                        elseif tmp_y_vertex0 == tmp_y_vertex_1
                            if tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                y_cross = tmp_y_vertex0;

                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                x_cross = DSM_PARAM.RANGE.X_MAX;
                                y_cross = tmp_y_vertex0;
                            end
                        else
                            m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                            tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                y_cross = m*x_cross + tmp_base;

                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                x_cross = DSM_PARAM.RANGE.X_MAX;
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

                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - DSM_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - DSM_PARAM.RANGE.X_RANGE));

                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - DSM_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - DSM_PARAM.RANGE.X_RANGE));

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

                if ~(tmp_y_vertex0 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= DSM_PARAM.RANGE.Y_MAX &&...
                        tmp_x_vertex0 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= DSM_PARAM.RANGE.X_MAX)

                    if (tmp_y_vertex1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= DSM_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                            tmp_x_vertex1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= DSM_PARAM.RANGE.X_MAX)

                        if tmp_x_vertex0 == tmp_x_vertex1
                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                x_cross = tmp_x_vertex0;

                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                x_cross = tmp_x_vertex0;
                            end

                        elseif tmp_y_vertex0 == tmp_y_vertex1
                            if tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                y_cross = tmp_y_vertex0;
                                x_cross = DSM_PARAM.RANGE.X_MIN;

                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                y_cross = tmp_y_vertex0;
                                x_cross = DSM_PARAM.RANGE.X_MAX;
                            end
                        else
                            m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                            tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                y_cross = m*x_cross + tmp_base;

                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                x_cross = DSM_PARAM.RANGE.X_MAX;
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


                    elseif (tmp_y_vertex_1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= DSM_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                            tmp_x_vertex_1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= DSM_PARAM.RANGE.X_MAX)

                        if tmp_x_vertex0 == tmp_x_vertex_1
                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                x_cross = tmp_x_vertex0;

                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                x_cross = tmp_x_vertex0;
                            end

                        elseif tmp_y_vertex0 == tmp_y_vertex_1
                            if tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                y_cross = tmp_y_vertex0;

                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                x_cross = DSM_PARAM.RANGE.X_MAX;
                                y_cross = tmp_y_vertex0;
                            end
                        else
                            m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                            tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                y_cross = m*x_cross + tmp_base;

                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                x_cross = DSM_PARAM.RANGE.X_MAX;
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

                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - DSM_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - DSM_PARAM.RANGE.X_RANGE));

                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - DSM_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - DSM_PARAM.RANGE.X_RANGE));

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

                if ~(tmp_y_vertex0 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex0 <= DSM_PARAM.RANGE.Y_MAX &&...
                        tmp_x_vertex0 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex0 <= DSM_PARAM.RANGE.X_MAX)

                    if (tmp_y_vertex1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex1 <= DSM_PARAM.RANGE.Y_MAX &&... % next vertex in ROI
                            tmp_x_vertex1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex1 <= DSM_PARAM.RANGE.X_MAX)

                        if tmp_x_vertex0 == tmp_x_vertex1
                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                x_cross = tmp_x_vertex0;

                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                x_cross = tmp_x_vertex0;
                            end

                        elseif tmp_y_vertex0 == tmp_y_vertex1
                            if tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                y_cross = tmp_y_vertex0;
                                x_cross = DSM_PARAM.RANGE.X_MIN;

                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                y_cross = tmp_y_vertex0;
                                x_cross = DSM_PARAM.RANGE.X_MAX;
                            end
                        else
                            m = (tmp_y_vertex1 - tmp_y_vertex0)/(tmp_x_vertex1 - tmp_x_vertex0);
                            tmp_base = tmp_y_vertex1 - m*tmp_x_vertex1;

                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                y_cross = m*x_cross + tmp_base;

                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                x_cross = DSM_PARAM.RANGE.X_MAX;
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


                    elseif (tmp_y_vertex_1 >= DSM_PARAM.RANGE.Y_MIN && tmp_y_vertex_1 <= DSM_PARAM.RANGE.Y_MAX &&... % before vertex in ROI
                            tmp_x_vertex_1 >= DSM_PARAM.RANGE.X_MIN && tmp_x_vertex_1 <= DSM_PARAM.RANGE.X_MAX)

                        if tmp_x_vertex0 == tmp_x_vertex_1
                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                x_cross = tmp_x_vertex0;

                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                x_cross = tmp_x_vertex0;
                            end

                        elseif tmp_y_vertex0 == tmp_y_vertex_1
                            if tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                y_cross = tmp_y_vertex0;

                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                x_cross = DSM_PARAM.RANGE.X_MAX;
                                y_cross = tmp_y_vertex0;
                            end
                        else
                            m = (tmp_y_vertex0 - tmp_y_vertex_1)/(tmp_x_vertex0 - tmp_x_vertex_1);
                            tmp_base = tmp_y_vertex_1 - m*tmp_x_vertex_1;

                            if tmp_y_vertex0 < DSM_PARAM.RANGE.Y_MIN
                                y_cross = DSM_PARAM.RANGE.Y_MIN;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_y_vertex0 > DSM_PARAM.RANGE.Y_MAX
                                y_cross = DSM_PARAM.RANGE.Y_MAX;
                                x_cross = (y_cross - tmp_base)/m;

                            elseif tmp_x_vertex0 < DSM_PARAM.RANGE.X_MIN
                                x_cross = DSM_PARAM.RANGE.X_MIN;
                                y_cross = m*x_cross + tmp_base;

                            elseif tmp_x_vertex0 > DSM_PARAM.RANGE.X_MAX
                                x_cross = DSM_PARAM.RANGE.X_MAX;
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
                if ( target_y_correction(vertex_index_beforeCurrentNext_all_out) >= DSM_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= DSM_PARAM.RANGE.Y_MAX ) || ...
                        ( target_x_correction(vertex_index_beforeCurrentNext_all_out) >= DSM_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= DSM_PARAM.RANGE.X_MAX )

                    if vertex_index_beforeCurrentNext_all_out == 1
                        target_y_correction(1) = target_y_correction(4);
                        target_x_correction(1) = target_x_correction(4);
                    else
                        target_y_correction(vertex_index_beforeCurrentNext_all_out) = target_y_correction(vertex_index_beforeCurrentNext_all_out-1);
                        target_x_correction(vertex_index_beforeCurrentNext_all_out) = target_x_correction(vertex_index_beforeCurrentNext_all_out-1);
                    end

                    % 모두 벗어나는 경우
                elseif ~(target_y_correction(vertex_index_beforeCurrentNext_all_out) >= DSM_PARAM.RANGE.Y_MIN && target_y_correction(vertex_index_beforeCurrentNext_all_out) <= DSM_PARAM.RANGE.Y_MAX &&...
                        target_x_correction(vertex_index_beforeCurrentNext_all_out) >= DSM_PARAM.RANGE.X_MIN && target_x_correction(vertex_index_beforeCurrentNext_all_out) <= DSM_PARAM.RANGE.X_MAX)

                    if target_y_correction(vertex_index_beforeCurrentNext_all_out) < DSM_PARAM.RANGE.Y_MIN
                        target_y_correction(vertex_index_beforeCurrentNext_all_out) = DSM_PARAM.RANGE.Y_MIN;
                    elseif target_y_correction(vertex_index_beforeCurrentNext_all_out) > DSM_PARAM.RANGE.Y_MAX
                        target_y_correction(vertex_index_beforeCurrentNext_all_out) = DSM_PARAM.RANGE.Y_MAX;
                    end

                    if target_x_correction(vertex_index_beforeCurrentNext_all_out) < DSM_PARAM.RANGE.X_MIN
                        target_x_correction(vertex_index_beforeCurrentNext_all_out) = DSM_PARAM.RANGE.X_MIN;
                    elseif target_x_correction(vertex_index_beforeCurrentNext_all_out) > DSM_PARAM.RANGE.X_MAX
                        target_x_correction(vertex_index_beforeCurrentNext_all_out) = DSM_PARAM.RANGE.X_MAX;
                    end
                end
            end

            for tmp_i = 1:length(target_y_correction) - 1
                tmp_y_vertex0 = target_y_correction(tmp_i);
                tmp_x_vertex0 = target_x_correction(tmp_i);

                tmp_y_vertex1 = target_y_correction(tmp_i+1);
                tmp_x_vertex1 = target_x_correction(tmp_i+1);

                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - DSM_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - DSM_PARAM.RANGE.X_RANGE));

                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - DSM_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - DSM_PARAM.RANGE.X_RANGE));

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

                [~,Image_Position_Y0] = min(abs(tmp_y_vertex0 - DSM_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X0] = min(abs(tmp_x_vertex0 - DSM_PARAM.RANGE.X_RANGE));

                [~,Image_Position_Y1] = min(abs(tmp_y_vertex1 - DSM_PARAM.RANGE.Y_RANGE));
                [~,Image_Position_X1] = min(abs(tmp_x_vertex1 - DSM_PARAM.RANGE.X_RANGE));

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

        if DSM_PARAM.RGB_IMAGE == 1
            if DSM_PARAM.BACKGROUND_COLOR_WHITE == 1
                for i_ch = 1:CH_LENGTH
                    if DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                        for tmp_j = 1:length(pixel_info(:,1))
                            if pixel_info(tmp_j,1) ~= 0
                                DSM_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = DSM_PARAM.RGB_MAX;
                            else
                                break
                            end
                        end

                    elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                        [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, end) - DSM_PARAM.RANGE.I_LAT_RANGE));

                        for tmp_j = 1:length(pixel_info(:,1))
                            if pixel_info(tmp_j,1) ~= 0
                                DSM_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;
                            else
                                break
                            end
                        end

                    elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY
                        [~, Collision_Probability_uint8] = min(abs(State_trajectory(TRAJ_PARAM.COLLISION_PROBABILITY, end) - DSM_PARAM.RANGE.COLLISION_PROBABILITY_RANGE));

                        for tmp_j = 1:length(pixel_info(:,1))
                            if pixel_info(tmp_j,1) ~= 0
                                DSM_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;
                            else
                                break
                            end
                        end

                    elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA
                        for tmp_j = 1:length(pixel_info(:,1))
                            if pixel_info(tmp_j,1) ~= 0
                                DSM_out(pixel_info(tmp_j,1),pixel_info(tmp_j,2):pixel_info(tmp_j,3),DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = DSM_PARAM.RGB_MIN;
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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trajectory (수정, 빈칸 찾고 한번에 연산하게 수정)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if DSM_PARAM.TRAJECTORY.ON == 1 && Target_Shape_Exist_Flag == 1
    for index_traj = 1:length(State_trajectory(1,:))
        if index_traj == length(State_trajectory(1,:))
           a = 1; 
        end
        
        if norm([State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj), State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj)],2) ~= 0 ...
                && State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj) >= DSM_PARAM.RANGE.X_MIN && State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj) <= DSM_PARAM.RANGE.X_MAX ...
                && State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj) >= DSM_PARAM.RANGE.Y_MIN && State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj) <= DSM_PARAM.RANGE.Y_MAX
            
            [~,Image_Position_X] = min(abs(State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj) - DSM_PARAM.RANGE.X_RANGE));
            [~,Image_Position_Y] = min(abs(State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj) - DSM_PARAM.RANGE.Y_RANGE));

            Target_Trajectory_Exist_Flag = 1;

            if DSM_PARAM.COLLISION_PROBABILITY.ON
                [~, Collision_Probability_uint8] = min(abs(State_trajectory(TRAJ_PARAM.COLLISION_PROBABILITY, index_traj) - DSM_PARAM.RANGE.COLLISION_PROBABILITY_RANGE));
            end
            
            if DSM_PARAM.RGB_IMAGE == 1
                if DSM_PARAM.BACKGROUND_COLOR_WHITE == 1

                    for i_ch = 1:CH_LENGTH
                        if DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                            DSM_out(Image_Position_X,Image_Position_Y,DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = DSM_PARAM.RGB_MIN;

                        elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.I_LAT % I_LAT
                            [~,I_LAT_uint8] = min(abs(State_trajectory(TRAJ_PARAM.I_LAT, index_traj) - DSM_PARAM.RANGE.I_LAT_RANGE));
                            DSM_out(Image_Position_X,Image_Position_Y,DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = I_LAT_uint8-1;

                        elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.COLLISION_PROBABILITY
                            DSM_out(Image_Position_X, Image_Position_Y, DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = Collision_Probability_uint8 - 1;

                        elseif DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE == TRAJ_PARAM.NA
                            DSM_out(Image_Position_X,Image_Position_Y,DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = DSM_PARAM.RGB_MIN;
                        end
                    end

                end
            end
        end
    end
end

if Target_Exist_in_Input_SBEV == 0 && Target_Shape_Exist_Flag == 0
    DSM_out = empty_DSM; % delete SBEV(with only lane mark, without target)
end








% if time_index == 4236
%     figure
%     imshow(uint8(BEV_Window_out))
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