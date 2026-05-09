%% Surrounding target prediction
% if TARGET_PRED_KF_CV
% 
% end


if TARGET_PRED_KF_CA == 1
    TRACKING_STATE_NUMBER = 6; % [y vy ay x vx ax]
    
    TRACKING.REL_POS_Y = 1;
    TRACKING.REL_VEL_Y = 2;
    TRACKING.REL_ACC_Y = 3;
    TRACKING.REL_POS_X = 4;
    TRACKING.REL_VEL_X = 5;
    TRACKING.REL_ACC_X = 6;

    TRACKING.WIDTH = 7;
    TRACKING.LENGTH = 8;
    TRACKING.HEADING_ANGLE = 9;
    TRACKING.SHAPE = 10;
    TRACKING.MOTION = 11;

    STATE_NUMBER = length(fieldnames(TRACKING)); % [y vy ay x vx ax] + [width, length, heading angle, classification, motion]

    
    X_est = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [y vy ay x vx ax] + [width, length, heading angle, classification, motion]
    P_est = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    X_pred = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [y vy ay x vx ax] + [width, length, heading angle, classification, motion]
    P_pred = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    z_CA = zeros(TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    X_updated = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [y vy ay x vx ax] + [width, length, heading angle, classification, motion]
    P_updated = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    X_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER); % [y vy ay x vx ax] + [width, length, heading angle, classification, motion]
    P_pred_window = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER);
    Association_Map_Total = zeros(FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    X_pred_window_SBEV = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER);
    
    % Kalman Filter - CA model parameter
    A_CA = [1, SAMPLE_TIME, 1/2*SAMPLE_TIME^2, 0, 0, 0   % y
            0, 1, SAMPLE_TIME, 0, 0, 0                   % vy
            0, 0, 1, 0, 0, 0                             % ay
            0, 0, 0, 1, SAMPLE_TIME, 1/2*SAMPLE_TIME^2   % x
            0, 0, 0, 0, 1, SAMPLE_TIME                   % vx
            0, 0, 0, 0, 0, 1];                           % ax
        
    % 원본
%     y_accel_variance_CA = 0.5;
%     x_accel_variance_CA = 1;

    y_accel_variance_CA = 0.2;
    x_accel_variance_CA = 0.4;
    
    Qy_CA = y_accel_variance_CA*[SAMPLE_TIME^5/20, SAMPLE_TIME^4/8, SAMPLE_TIME^3/6
                                 SAMPLE_TIME^4/8, SAMPLE_TIME^3/3, SAMPLE_TIME^2/2
                                 SAMPLE_TIME^3/6, SAMPLE_TIME^2/2, SAMPLE_TIME];

    Qx_CA = x_accel_variance_CA*[SAMPLE_TIME^5/20, SAMPLE_TIME^4/8, SAMPLE_TIME^3/6
                                 SAMPLE_TIME^4/8, SAMPLE_TIME^3/3, SAMPLE_TIME^2/2
                                 SAMPLE_TIME^3/6, SAMPLE_TIME^2/2, SAMPLE_TIME];

    Q_CA = blkdiag(Qy_CA, Qx_CA);
    
    A_CA_Pred = [1, TARGET_PRED_SAMPLE_RATE, 1/2*TARGET_PRED_SAMPLE_RATE, 0, 0, 0   % y
                 0, 1, TARGET_PRED_SAMPLE_RATE, 0, 0, 0                             % vy
                 0, 0, 1, 0, 0, 0                                                   % ay
                 0, 0, 0, 1, TARGET_PRED_SAMPLE_RATE, 1/2*TARGET_PRED_SAMPLE_RATE   % x
                 0, 0, 0, 0, 1, TARGET_PRED_SAMPLE_RATE                             % vx
                 0, 0, 0, 0, 0, 1];                                                 % ax

    Qy_CA_Pred = y_accel_variance_CA*[TARGET_PRED_SAMPLE_RATE^5/20, TARGET_PRED_SAMPLE_RATE^4/8, TARGET_PRED_SAMPLE_RATE^3/6
                                 TARGET_PRED_SAMPLE_RATE^4/8, TARGET_PRED_SAMPLE_RATE^3/3, TARGET_PRED_SAMPLE_RATE^2/2
                                 TARGET_PRED_SAMPLE_RATE^3/6, TARGET_PRED_SAMPLE_RATE^2/2, TARGET_PRED_SAMPLE_RATE];

    Qx_CA_Pred = x_accel_variance_CA*[TARGET_PRED_SAMPLE_RATE^5/20, TARGET_PRED_SAMPLE_RATE^4/8, TARGET_PRED_SAMPLE_RATE^3/6
                                 TARGET_PRED_SAMPLE_RATE^4/8, TARGET_PRED_SAMPLE_RATE^3/3, TARGET_PRED_SAMPLE_RATE^2/2
                                 TARGET_PRED_SAMPLE_RATE^3/6, TARGET_PRED_SAMPLE_RATE^2/2, TARGET_PRED_SAMPLE_RATE];

    Q_CA_Pred = blkdiag(Qy_CA_Pred, Qx_CA_Pred);

    
    TRACKING.RESIDUAL.DEFAULT_VALUE = 300;
    TRACKING.GATING.INPUT_NUMBER = 4; % y, x, vy, vx
    

    tmp_residual_total = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(TRACKING.GATING.INPUT_NUMBER, FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    Association_Map_Updated = zeros(FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    H_CA = [1 0 0 0 0 0 % y
            0 1 0 0 0 0 % vy
            0 0 0 0 0 0 % ay
            0 0 0 1 0 0 % x
            0 0 0 0 1 0 % vx
            0 0 0 0 0 1]; % ax
    
%     R_CA = 0.5*eye(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);
    R_CA = blkdiag(5^2*eye(TRACKING_STATE_NUMBER-1, TRACKING_STATE_NUMBER-1), 0.3);
    
    GATING.Y_MIN                           = -2;
    GATING.Y_MAX                           = 2;
    GATING.X_MIN                           = -3.5;
    GATING.X_MAX                           = 3.5;
    GATING.VY_MIN                          = -1.5;
    GATING.VY_MAX                          = 1.5;
    GATING.VX_MIN                          = -1.5;
    GATING.VX_MAX                          = 1.5;

    % collision probability
    collision_probability_total = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % prediction window, track_number, length(sim_time)
    collision_probability_final = zeros(length(sim_time), FUSION_TRACK.TRACK_NUMBER);
    
    for index_time = Test_start_index:SBEV_Gen_Sample_Rate/SAMPLE_TIME:Test_end_index

        tmp_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(TRACKING.GATING.INPUT_NUMBER, FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);
        tmp_norm_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);
        

        if index_time >= 8
            a = 1;
        end


        if index_time == 1676
            a = 1;
        end

        if index_time == 1953
            a = 1;
        end

        if index_time >= 1172
            a = 1;
        end

        if index_time == 1122 % LK_BWD_ST_13
            a = 1;
        end

        if index_time >= 2140 % 처음 FST 25가 ROI에 들어올 때
            a = 1;
        end

        if index_time >= 762
            a = 1;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Tracking for error covariance
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Prediction
        if index_time == 1
            Association_Map_k_1 = zeros(FUSION_TRACK.TRACK_NUMBER, 1);
            Fusion_Track_k_1 = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            P_Fusion_Track_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
        else
            Association_Map_k_1 = Association_Map_Total(:, index_time - 1);
            Fusion_Track_k_1 = X_est(:, :, index_time - 1);
            P_Fusion_Track_k_1 = P_est(:,:,:, index_time - 1);
        end
        
        for track_number = 1:FUSION_TRACK.TRACK_NUMBER
            if Association_Map_k_1(track_number,1) ~= 0
                X_pred(1:6, track_number, index_time) = A_CA * Fusion_Track_k_1(1:6, track_number);

                X_pred(TRACKING.WIDTH, track_number, index_time) = Fusion_Track_k_1(TRACKING.WIDTH, track_number); % width
                X_pred(TRACKING.LENGTH, track_number, index_time) = Fusion_Track_k_1(TRACKING.LENGTH, track_number); % length
                X_pred(TRACKING.HEADING_ANGLE, track_number, index_time) = Fusion_Track_k_1(TRACKING.HEADING_ANGLE, track_number); % heading angle
                X_pred(TRACKING.SHAPE, track_number, index_time) = Fusion_Track_k_1(TRACKING.SHAPE, track_number); % classification
                X_pred(TRACKING.MOTION, track_number, index_time) = Fusion_Track_k_1(TRACKING.MOTION, track_number); % motion
                
                P_pred(:, :, track_number, index_time) = A_CA * P_Fusion_Track_k_1(:, :, track_number) * A_CA' + Q_CA;
            end
        end
        
        % Correction
        for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
            if track_number_k_1 == 28
                a = 1;
            end

            if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0
                for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                    if track_number == 25
                        a = 1;
                    end

                    if norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)], 2) ~= 0
                        
                        if Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time) == Fusion_Track_k_1(TRACKING.SHAPE, track_number_k_1)
                            
                            tmp_residual(1, track_number, track_number_k_1) = X_pred(TRACKING.REL_POS_Y, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time); % y
                            tmp_residual(2, track_number, track_number_k_1) = X_pred(TRACKING.REL_POS_X, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time); % x
                            tmp_residual(3, track_number, track_number_k_1) = X_pred(TRACKING.REL_VEL_Y, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time); % vy
                            tmp_residual(4, track_number, track_number_k_1) = X_pred(TRACKING.REL_VEL_X, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time); % vx

                            tmp_residual_total(:, track_number, track_number_k_1, index_time) = tmp_residual(:, track_number, track_number_k_1);
                            
                            tmp_norm_residual(track_number_k_1, track_number) = norm(tmp_residual(:, track_number, track_number_k_1),2);
                        end
                    end
                end
            end
        end
        
        for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
            if track_number_k_1 == 28
                a = 1;
            end

            if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0
                
                [~, sorted_track_number] = sort(tmp_norm_residual(track_number_k_1,:));
                [~, sorted_track_number_k_1] = sort(tmp_norm_residual(:, sorted_track_number(1)));
                
                if sorted_track_number_k_1(1) == track_number_k_1 && ...
                        tmp_residual(1, sorted_track_number(1), track_number_k_1) > GATING.Y_MIN && tmp_residual(1, sorted_track_number(1), track_number_k_1) < GATING.Y_MAX && ...
                        tmp_residual(2, sorted_track_number(1), track_number_k_1) > GATING.X_MIN && tmp_residual(2, sorted_track_number(1), track_number_k_1) < GATING.X_MAX

                    z_CA(:,track_number_k_1,index_time) = [Fusion_Track([FUSION_TRACK.TRACKING.REL_POS_Y, FUSION_TRACK.TRACKING.REL_VEL_Y], sorted_track_number(1), index_time); 0;
                        Fusion_Track([FUSION_TRACK.TRACKING.REL_POS_X, FUSION_TRACK.TRACKING.REL_VEL_X, FUSION_TRACK.MEASURE.REL_ACC_X], sorted_track_number(1), index_time)]...
                        - H_CA*X_pred(1:6, track_number_k_1, index_time);
                    
                    X_updated(TRACKING.WIDTH, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, sorted_track_number(1), index_time); % width
                    X_updated(TRACKING.LENGTH, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, sorted_track_number(1), index_time); % length
                    X_updated(TRACKING.HEADING_ANGLE, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, sorted_track_number(1), index_time); % heading angle
                    
                    X_updated(TRACKING.SHAPE, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, sorted_track_number(1), index_time); % classification
                    X_updated(TRACKING.MOTION, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, sorted_track_number(1), index_time); % motion attribute

                    Association_Map_Total(track_number_k_1, index_time) = sorted_track_number(1);
                else
                    X_updated(TRACKING.WIDTH, track_number_k_1, index_time) = X_pred(TRACKING.WIDTH, track_number_k_1, index_time); % width
                    X_updated(TRACKING.LENGTH, track_number_k_1, index_time) = X_pred(TRACKING.LENGTH, track_number_k_1, index_time); % length
                    X_updated(TRACKING.HEADING_ANGLE, track_number_k_1, index_time) = X_pred(TRACKING.HEADING_ANGLE, track_number_k_1, index_time); % heading angle
                    
                    X_updated(TRACKING.SHAPE, track_number_k_1, index_time) = X_pred(TRACKING.SHAPE, track_number_k_1, index_time); % classification
                    X_updated(TRACKING.MOTION, track_number_k_1, index_time) = X_pred(TRACKING.MOTION, track_number_k_1, index_time); % motion attribute                    
                end

                S_CA = H_CA*P_pred(:, :, track_number_k_1, index_time)*H_CA' + R_CA;
                K_CA = P_pred(:, :, track_number_k_1, index_time)*H_CA'*inv(S_CA);

                X_updated(1:6, track_number_k_1, index_time) = X_pred(1:6, track_number_k_1, index_time) + K_CA * z_CA(:, track_number_k_1, index_time);
                P_updated(:, :, track_number_k_1, index_time) = P_pred(:, :, track_number_k_1, index_time) - K_CA * H_CA * P_pred(:, :, track_number_k_1, index_time);
            end
        end
        
        Track_Assigned_Flag = 0;
        % Track Management
        % Maintenance
        for track_number = 1:FUSION_TRACK.TRACK_NUMBER
            if norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)], 2) ~= 0
                
                for updated_track_number = 1:FUSION_TRACK.TRACK_NUMBER
                    if Association_Map_Total(updated_track_number, index_time) ~= 0
                        if Association_Map_Total(updated_track_number, index_time) == track_number
                            if X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.VEHICLE_CANDIDATE || ...
                                    X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.VEHICLE_CONFIRMED || ...
                                    X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.PEDESTRIAN_CANDIDATE || ...
                                    X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.PEDESTRIAN_CONFIRMED || ...
                                    X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.MOTOR_BIKE_CANDIDATE || ...
                                    X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.MOTOR_BIKE_CONFIRMED || ...
                                    X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.BICYCLE_CANDIDATE || ...
                                    X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.BICYCLE_CONFIRMED || ...
                                    X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.TRUCK_CANDIDATE || ...
                                    X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.TRUCK_CONFIRMED                                    
                                
                                Track_Assigned_Flag = 1;
                                break
                            end
                        end
                    end
                end
                
                if Track_Assigned_Flag == 1
                    X_est(:, track_number, index_time) = X_updated(:, track_number, index_time);
                    P_est(:, :, track_number, index_time) = P_updated(:, :, track_number, index_time);
                    Track_Assigned_Flag = 0;
                end
            end
        end
        
        % Creation
        for track_number = 1:FUSION_TRACK.TRACK_NUMBER
            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                % SBEV ROI
                target_y_vertex = zeros(1,5);
                target_x_vertex = zeros(1,5);

                tmp_target_y_vertex = [-Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2,...
                    Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, ...
                    -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2];

                tmp_target_x_vertex = [0, Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), 0, 0];

                target_y_vertex_rot = tmp_target_x_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) + tmp_target_y_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                target_x_vertex_rot = tmp_target_x_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - tmp_target_y_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                target_y_vertex = target_y_vertex_rot + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                target_x_vertex = target_x_vertex_rot + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);


                if ( (min(target_y_vertex) >= Y_MIN && min(target_y_vertex) <= Y_MAX) || (max(target_y_vertex) >= Y_MIN && max(target_y_vertex) <= Y_MAX) ) ...
                        && ( (min(target_x_vertex) >= X_MIN && min(target_x_vertex) <= X_MAX) || (max(target_x_vertex) >= X_MIN && max(target_x_vertex) <= X_MAX) )

                    for updated_track_number = 1:FUSION_TRACK.TRACK_NUMBER
                        if Association_Map_Total(updated_track_number, index_time) ~= 0
                            if track_number == Association_Map_Total(updated_track_number, index_time)
                                Track_Assigned_Flag = 1;
                                break
                            end
                        end
                    end

                    if Track_Assigned_Flag == 0

                        if sum(Association_Map_Total(track_number, index_time)) == 0

                            if track_number == 25
                                a = 1;
                            end

                            Association_Map_Total(track_number, index_time) = track_number;

                            % [y vy ay x vx ax]
                            X_est(TRACKING.REL_POS_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                            X_est(TRACKING.REL_VEL_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);
                            X_est(TRACKING.REL_ACC_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_Y, track_number, index_time);
                            X_est(TRACKING.REL_POS_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                            X_est(TRACKING.REL_VEL_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                            X_est(TRACKING.REL_ACC_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_X, track_number, index_time);

                            X_est(TRACKING.WIDTH, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_est(TRACKING.LENGTH, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_est(TRACKING.HEADING_ANGLE, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                            X_est(TRACKING.SHAPE, track_number, index_time) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_est(TRACKING.MOTION, track_number, index_time) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                            P_est(:,:,track_number, index_time) = eye(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);
                        end

                    end
                    Track_Assigned_Flag = 0;
                end
            end
        end
        
        % Deletion
        for i_X_est = 1:FUSION_TRACK.TRACK_NUMBER
            if i_X_est == 25
                a = 1;
            end

            if sum(Association_Map_Total(i_X_est, index_time)) ~= 0
                for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                    if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                        % SBEV ROI
                        target_y_vertex = zeros(1,5);
                        target_x_vertex = zeros(1,5);

                        tmp_target_y_vertex = [-Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2,...
                            Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, ...
                            -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2];

                        tmp_target_x_vertex = [0, Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), 0, 0];

                        target_y_vertex_rot = tmp_target_x_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) + tmp_target_y_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                        target_x_vertex_rot = tmp_target_x_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - tmp_target_y_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                        target_y_vertex = target_y_vertex_rot + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                        target_x_vertex = target_x_vertex_rot + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);

                        if ( (min(target_y_vertex) >= Y_MIN && min(target_y_vertex) <= Y_MAX) || (max(target_y_vertex) >= Y_MIN && max(target_y_vertex) <= Y_MAX)) ...
                                && ((min(target_x_vertex) >= X_MIN && min(target_x_vertex) <= X_MAX) || (max(target_x_vertex) >= X_MIN && max(target_x_vertex) <= X_MAX))
                            
                            if Association_Map_Total(i_X_est, index_time) == track_number
                                
                                Fusion_Object_Exist_Flag = 1;
                                break
                            end
                        end
                    end
                end
                
                if Fusion_Object_Exist_Flag == 0

                    if i_X_est == 25
                        a = 1;
                    end
                    
                    X_est(:, i_X_est, index_time) = zeros(STATE_NUMBER, 1);
                    P_est(:,:,i_X_est, index_time) = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, 1);
                    
                    Association_Map_Total(i_X_est, index_time) = 0;
                end
                Fusion_Object_Exist_Flag = 0;
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Prediction
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic
        for track_number = 1:FUSION_TRACK.TRACK_NUMBER

            if track_number == 3
                a = 1;
            end

            collision_probability_max = 0;

            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                % ROI
                target_y_vertex = zeros(1,5);
                target_x_vertex = zeros(1,5);

                tmp_target_y_vertex = [-Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2,...
                    Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, ...
                    -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2];

                tmp_target_x_vertex = [0, Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), 0, 0];

                target_y_vertex_rot = tmp_target_x_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) + tmp_target_y_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                target_x_vertex_rot = tmp_target_x_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - tmp_target_y_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                target_y_vertex = target_y_vertex_rot + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                target_x_vertex = target_x_vertex_rot + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);

                if ( (min(target_y_vertex) >= Y_MIN && min(target_y_vertex) <= Y_MAX) || (max(target_y_vertex) >= Y_MIN && max(target_y_vertex) <= Y_MAX) ) ...
                    && ( (min(target_x_vertex) >= X_MIN && min(target_x_vertex) <= X_MAX) || (max(target_x_vertex) >= X_MIN && max(target_x_vertex) <= X_MAX) ) ...
                    && sum(P_est(:, :, track_number, index_time), 'all') ~= 0

                    Prediction_On(index_time, 1) = 1;

                    for index_pred = 1:TARGET_PRED_WINDOW/SAMPLE_TIME
                        if index_pred == 1
                            X_pred_window(1:6, index_time, index_pred, track_number) = A_CA * Fusion_Track([FUSION_TRACK.TRACKING.REL_POS_Y, FUSION_TRACK.TRACKING.REL_VEL_Y, FUSION_TRACK.MEASURE.REL_ACC_Y, ...
                                FUSION_TRACK.TRACKING.REL_POS_X, FUSION_TRACK.TRACKING.REL_VEL_X, FUSION_TRACK.MEASURE.REL_ACC_X], track_number, index_time);

                            X_pred_window(TRACKING.WIDTH, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                            X_pred_window(TRACKING.SHAPE, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_pred_window(TRACKING.MOTION, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);


                            P_pred_window(:, :, index_time, index_pred, track_number) = A_CA * P_est(:, :, track_number, index_time) * A_CA' + Q_CA;
                        else

                            if index_pred == TARGET_PRED_WINDOW/SAMPLE_TIME
                                a = 1;
                            end
                            X_pred_window(1:6, index_time, index_pred, track_number) = A_CA * X_pred_window(1:6, index_time, index_pred - 1, track_number);

                            X_pred_window(TRACKING.WIDTH, index_time, index_pred, track_number) = X_pred_window(TRACKING.WIDTH, index_time, index_pred - 1, track_number);
                            X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number) = X_pred_window(TRACKING.LENGTH, index_time, index_pred - 1, track_number);
                            X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number) = X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred - 1, track_number);
                            X_pred_window(TRACKING.SHAPE, index_time, index_pred, track_number) = X_pred_window(TRACKING.SHAPE, index_time, index_pred - 1, track_number);
                            X_pred_window(TRACKING.MOTION, index_time, index_pred, track_number) = X_pred_window(TRACKING.MOTION, index_time, index_pred - 1, track_number);


                            P_pred_window(:, :, index_time, index_pred, track_number) = A_CA * P_pred_window(:, :, index_time, index_pred - 1, track_number) * A_CA' + Q_CA;
                        end

                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % Collision Probability
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if Collision_Probability_Switch == 1
                            if index_pred == 1
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = round(tmp_index*TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)

                                tmp_P_pred_window = P_pred_window([TRACKING.REL_POS_X, TRACKING.REL_POS_Y], [TRACKING.REL_POS_X, TRACKING.REL_POS_Y], index_time, index_pred, track_number); % [xx xy; yx yy]

                                tmp_sigma_x = sqrt(tmp_P_pred_window(1, 1));
                                tmp_sigma_y = sqrt(tmp_P_pred_window(2, 2));

                                tmp_y_f = EGO_VEHICLE.WIDTH/2 +...
                                          Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) +....
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_y_i = -EGO_VEHICLE.WIDTH/2 -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                                tmp_cdf_y_f = normcdf(tmp_y_f, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);
                                tmp_cdf_y_i = normcdf(tmp_y_i, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);

                                tmp_x_f = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_x_i = -EGO_VEHICLE.LENGTH -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time))*sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                                tmp_cdf_x_f = normcdf(tmp_x_f, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);
                                tmp_cdf_x_i = normcdf(tmp_x_i, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);

                                tmp_cdf_y_i_to_y_f = tmp_cdf_y_f - tmp_cdf_y_i;
                                tmp_cdf_x_i_to_x_f = tmp_cdf_x_f - tmp_cdf_x_i;

                                tmp_collision_probability = tmp_cdf_y_i_to_y_f * tmp_cdf_x_i_to_x_f;

                                collision_probability_total(index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number, index_time) = tmp_collision_probability; % prediction window, track_number, length(sim_time)

                                if tmp_collision_probability > collision_probability_max
                                    collision_probability_max = tmp_collision_probability;
                                end

                                X_pred_window_SBEV(:, index_time, round( index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000 ), track_number) = X_pred_window(:, index_time, index_pred, track_number);
                            end
                        else
                            if index_pred == 1 %TARGET_PRED_WINDOW/SAMPLE_TIME
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = round(tmp_index*TARGET_PRED_SAMPLE_RATE*10/(SAMPLE_TIME *100) *10);
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)
                                X_pred_window_SBEV(:, index_time, round( index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000 ), track_number) = X_pred_window(:, index_time, index_pred, track_number);
                            end
                        end
                    end

                    if Collision_Probability_Switch == 1
                        collision_probability_final(index_time, track_number) = collision_probability_max;
                    end
                end
            end
        end
        tmp_Execution_Time_for_prediction = toc;

        if Evaluation_of_Prediction_Switch
            if Prediction_On(index_time, 1) == 1
                Execution_Time_Total(index_time, 1) = tmp_Execution_Time_for_prediction;
                tmp_Execution_Time_for_prediction = 0;
            end
        end

        if Evaluation_Collision_Probability_Switch
            if Prediction_On(index_time, 1) == 1
                Collision_Probability(index_time, 1) = max( collision_probability_final(index_time, :) );

                if Collision_Probability(index_time, 1) >= COLLISION_PROBABILITY.THRESHOLD
                    Predict_Collision(index_time, 1) = COLLISION.PRECRASH;
                else
                    Predict_Collision(index_time, 1) = COLLISION.SAFE;
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Generate Timeseries Annotation
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if impact_section(Data_index,1) ~= 0 % precrash
                    if index_time >= Annotation_start_index && index_time <= Annotation_end_index
                        time_GT(index_time,1) = COLLISION.PRECRASH;
                    else
                        time_GT(index_time,1) = COLLISION.SAFE;
                    end

                else % safe
                    time_GT(index_time,1) = COLLISION.SAFE;
                end
            end
        end
    end
end


if TARGET_PRED_EKF_CTRV

    TRACKING_STATE_NUMBER = 5; % [x, y, vx, vy, heading angular rate]'
    
    TRACKING.REL_POS_X = 1;
    TRACKING.REL_POS_Y = 2;
    TRACKING.REL_VEL_X = 3;
    TRACKING.REL_VEL_Y = 4;
    TRACKING.HEADING_ANGLE_RATE = 5;
    
    % 추후 수정 필요
    TRACKING.HEADING_ANGLE = 6;    
    TRACKING.WIDTH = 7;
    TRACKING.LENGTH = 8;    
    TRACKING.SHAPE = 9;
    TRACKING.MOTION = 10;

    STATE_NUMBER = length(fieldnames(TRACKING)); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'

    X_est = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_est = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    X_pred = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_pred = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    z_CTRV = zeros(TRACKING_STATE_NUMBER - 1, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy]'
    X_updated = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_updated = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    X_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_pred_window = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER);

    Association_Map_Total = zeros(FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    X_pred_window_SBEV = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER);


    x_variance_CTRV = 0.4;
    y_variance_CTRV = 0.2;
    w_variance_CTRV = 3.16*10^-4;

    Q_CTRV = [x_variance_CTRV*SAMPLE_TIME^4/4, 0, x_variance_CTRV*SAMPLE_TIME^3/2, 0, 0
              0, y_variance_CTRV*SAMPLE_TIME^4/4, 0, y_variance_CTRV*SAMPLE_TIME^3/2, 0
              x_variance_CTRV*SAMPLE_TIME^3/2, 0, x_variance_CTRV*SAMPLE_TIME, 0, 0
              0, y_variance_CTRV*SAMPLE_TIME^3, 0, y_variance_CTRV*SAMPLE_TIME, 0
              0, 0, 0, 0, w_variance_CTRV];

%     std_x_processNoise_in_ref = 0.1;
%     std_y_processNoise_in_ref = 0.1;
%     std_vx_processNoise_in_ref = 0.1;
%     std_vy_processNoise_in_ref = 0.1;
%     std_w_processNoise_in_ref = 3.16*10^-4;
% 
%     Q_CTRV = [SAMPLE_TIME^2/2 0 0
%               0 SAMPLE_TIME^2/2 0
%               SAMPLE_TIME 0 0
%               0 SAMPLE_TIME 0
%               0 0 SAMPLE_TIME]*w_CTRV;

    TRACKING.RESIDUAL.DEFAULT_VALUE = 300;
    TRACKING.GATING.INPUT_NUMBER = 4; % y, x, vy, vx
    
    
    Association_Map_Updated = zeros(FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    H_CTRV = [1 0 0 0 0    % x
              0 1 0 0 0    % y
              0 0 1 0 0    % vx
              0 0 0 1 0];  % vy
                   
%     R_CTRV = 0.5*eye(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);    
%     R_CTRV = 0.1*eye(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);
%     R_CTRV = 0.01*eye(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);

%     x_e_CTRV = 0.01;
%     y_e_CTRV = 0.01;
%     vx_e_CTRV = 0.1;
%     vy_e_CTRV = 0.1;

    x_e_CTRV = 0.1;
    y_e_CTRV = 0.1;
    vx_e_CTRV = 0.1;
    vy_e_CTRV = 0.1;


%     w_e_CTRV = 0.8;
% 
%     R_CTRV = blkdiag(x_e_CTRV, y_e_CTRV, vx_e_CTRV, vy_e_CTRV, w_e_CTRV);

%     x_e_CTRV = 1;
%     y_e_CTRV = 1;
%     vx_e_CTRV = 2;
%     vy_e_CTRV = 2;
%     w_e_CTRV = 15*pi/180;

%     R_CTRV = blkdiag(x_e_CTRV, y_e_CTRV, vx_e_CTRV, vy_e_CTRV, w_e_CTRV);
    R_CTRV = diag([x_e_CTRV, y_e_CTRV, vx_e_CTRV, vy_e_CTRV]);



    GATING.Y_MIN                           = -2;
    GATING.Y_MAX                           = 2;
    GATING.X_MIN                           = -3.5;
    GATING.X_MAX                           = 3.5;
    GATING.VY_MIN                          = -1.5;
    GATING.VY_MAX                          = 1.5;
    GATING.VX_MIN                          = -1.5;
    GATING.VX_MAX                          = 1.5;


    % collision probability
    collision_probability_total = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % prediction window, track_number, length(sim_time)
    collision_probability_final = zeros(length(sim_time), FUSION_TRACK.TRACK_NUMBER);


    for index_time = Test_start_index:SBEV_Gen_Sample_Rate/SAMPLE_TIME:Test_end_index

        tmp_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(TRACKING.GATING.INPUT_NUMBER, FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);
        tmp_norm_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);

        if index_time >= 1224
            a = 1;
        end

        if index_time >= 1501
            a = 1;
        end
        if index_time >= 1670
            a = 1;
        end

        if index_time >= 1676
            a = 1;
        end
        if index_time >= 1646
            a = 1;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Tracking for error covariance
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Prediction
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if index_time == 1
                Association_Map_k_1 = zeros(FUSION_TRACK.TRACK_NUMBER, 1);
                Fusion_Track_k_1 = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
                P_Fusion_Track_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            else
                Association_Map_k_1 = Association_Map_Total(:, index_time - 1);
                Fusion_Track_k_1 = X_est(:, :, index_time - 1);
                P_Fusion_Track_k_1 = P_est(:,:,:, index_time - 1);
            end

            for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                if Association_Map_k_1(track_number,1) ~= 0
                    
                    if abs( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) ) < 0.001

                        Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) = 0.001;
                    end

                    % [x, y, vx, vy, heading angular rate]'
                    X_pred(TRACKING.REL_POS_X, track_number, index_time) = Fusion_Track_k_1(TRACKING.REL_POS_X, track_number) + ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) - ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * ( 1 - cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) ) ;

                    X_pred(TRACKING.REL_POS_Y, track_number, index_time) = Fusion_Track_k_1(TRACKING.REL_POS_Y, track_number) + ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * ( 1 - cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) ) + ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME );

                    X_pred(TRACKING.REL_VEL_X, track_number, index_time) = Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number) * cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) - ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number) * sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME );

                    X_pred(TRACKING.REL_VEL_Y, track_number, index_time) = Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number) * sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) + ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number) * cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME );

                    X_pred(TRACKING.HEADING_ANGLE_RATE, track_number, index_time) = Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number);

                    dx_dvx = sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number);
                    dx_dvy = - ( 1 - cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) ) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number);
                    dx_dw = Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) * SAMPLE_TIME - ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number)^2 * sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) - ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) * SAMPLE_TIME + ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number)^2 * ( 1 - cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) );


                    dy_dvx = ( 1 - cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) ) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number);
                    dy_dvy = sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number);
                    dy_dw = Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) * SAMPLE_TIME - ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number)^2 * ( 1 - cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) ) + ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) * SAMPLE_TIME - ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number) / Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number)^2 * sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME );

                    dvx_dvx = cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME );
                    dvx_dvy = - sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME );
                    dvx_dw = - Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number) * sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) * SAMPLE_TIME - ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number) * cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) * SAMPLE_TIME;

                    dvy_dvx = sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME );
                    dvy_dvy = cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME );
                    dvy_dw = Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number) * cos( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) * SAMPLE_TIME - ...
                        Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number) * sin( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME ) * SAMPLE_TIME;


                    J_A_CTRV = [1, 0, dx_dvx, dx_dvy, dx_dw
                        0, 1, dy_dvx, dy_dvy, dy_dw
                        0, 0, dvx_dvx, dvx_dvy, dvx_dw
                        0, 0, dvy_dvx, dvy_dvy, dvy_dw
                        0, 0, 0,       0,     1];
                  

                    X_pred(TRACKING.HEADING_ANGLE, track_number, index_time) = Fusion_Track_k_1(TRACKING.HEADING_ANGLE, track_number) + Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME;
                    X_pred(TRACKING.WIDTH, track_number, index_time) = Fusion_Track_k_1(TRACKING.WIDTH, track_number); % width
                    X_pred(TRACKING.LENGTH, track_number, index_time) = Fusion_Track_k_1(TRACKING.LENGTH, track_number); % length                    
                    X_pred(TRACKING.SHAPE, track_number, index_time) = Fusion_Track_k_1(TRACKING.SHAPE, track_number); % classification
                    X_pred(TRACKING.MOTION, track_number, index_time) = Fusion_Track_k_1(TRACKING.MOTION, track_number); % motion

                    P_pred(:, :, track_number, index_time) = J_A_CTRV * P_Fusion_Track_k_1(:, :, track_number) * J_A_CTRV' + Q_CTRV;
                end
            end

            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Correction
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
                if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0
                    for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                        if norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)], 2) ~= 0

                            if Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time) == Fusion_Track_k_1(TRACKING.SHAPE, track_number_k_1)

                                tmp_residual(1, track_number, track_number_k_1) = X_pred(TRACKING.REL_POS_Y, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time); % y
                                tmp_residual(2, track_number, track_number_k_1) = X_pred(TRACKING.REL_POS_X, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time); % x
                                tmp_residual(3, track_number, track_number_k_1) = X_pred(TRACKING.REL_VEL_Y, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time); % vy
                                tmp_residual(4, track_number, track_number_k_1) = X_pred(TRACKING.REL_VEL_X, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time); % vx
                                
                                tmp_norm_residual(track_number_k_1, track_number) = norm(tmp_residual(:, track_number, track_number_k_1),2);
                            end
                        end
                    end
                end
            end

            for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
                if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0

                    [~, sorted_track_number] = sort(tmp_norm_residual(track_number_k_1,:));
                    [~, sorted_track_number_k_1] = sort(tmp_norm_residual(:, sorted_track_number(1)));

                    if sorted_track_number_k_1(1) == track_number_k_1 && ...
                            tmp_residual(1, sorted_track_number(1), track_number_k_1) > GATING.Y_MIN && tmp_residual(1, sorted_track_number(1), track_number_k_1) < GATING.Y_MAX && ...
                            tmp_residual(2, sorted_track_number(1), track_number_k_1) > GATING.X_MIN && tmp_residual(2, sorted_track_number(1), track_number_k_1) < GATING.X_MAX

                        z_CTRV(:,track_number_k_1,index_time) = [Fusion_Track([FUSION_TRACK.TRACKING.REL_POS_X, FUSION_TRACK.TRACKING.REL_POS_Y, FUSION_TRACK.TRACKING.REL_VEL_X, FUSION_TRACK.TRACKING.REL_VEL_Y], sorted_track_number(1), index_time)]...
                            - H_CTRV*X_pred(1:TRACKING_STATE_NUMBER, track_number_k_1, index_time);

                        X_updated(TRACKING.HEADING_ANGLE, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, sorted_track_number(1), index_time); % heading angle
                        X_updated(TRACKING.WIDTH, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, sorted_track_number(1), index_time); % width
                        X_updated(TRACKING.LENGTH, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, sorted_track_number(1), index_time); % length
                        X_updated(TRACKING.SHAPE, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, sorted_track_number(1), index_time); % classification
                        X_updated(TRACKING.MOTION, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, sorted_track_number(1), index_time); % motion attribute

                        Association_Map_Total(track_number_k_1, index_time) = sorted_track_number(1);
                    else
                        X_updated(TRACKING.HEADING_ANGLE, track_number_k_1, index_time) = X_pred(TRACKING.HEADING_ANGLE, track_number_k_1, index_time); % heading angle
                        X_updated(TRACKING.WIDTH, track_number_k_1, index_time) = X_pred(TRACKING.WIDTH, track_number_k_1, index_time); % width
                        X_updated(TRACKING.LENGTH, track_number_k_1, index_time) = X_pred(TRACKING.LENGTH, track_number_k_1, index_time); % length
                        X_updated(TRACKING.SHAPE, track_number_k_1, index_time) = X_pred(TRACKING.SHAPE, track_number_k_1, index_time); % classification
                        X_updated(TRACKING.MOTION, track_number_k_1, index_time) = X_pred(TRACKING.MOTION, track_number_k_1, index_time); % motion attribute
                    end

                    S_CTRV = H_CTRV*P_pred(:, :, track_number_k_1, index_time)*H_CTRV' + R_CTRV;
                    K_CTRV = P_pred(:, :, track_number_k_1, index_time)*H_CTRV'*inv(S_CTRV);

                    X_updated(1:TRACKING_STATE_NUMBER, track_number_k_1, index_time) = X_pred(1:TRACKING_STATE_NUMBER, track_number_k_1, index_time) + K_CTRV * z_CTRV(:, track_number_k_1, index_time);
                    P_updated(:, :, track_number_k_1, index_time) = P_pred(:, :, track_number_k_1, index_time) - K_CTRV * H_CTRV * P_pred(:, :, track_number_k_1, index_time);
                end
            end


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Track Management
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Maintenance
            Track_Assigned_Flag = 0;

            for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                if norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)], 2) ~= 0

                    for updated_track_number = 1:FUSION_TRACK.TRACK_NUMBER
                        if Association_Map_Total(updated_track_number, index_time) ~= 0
                            if Association_Map_Total(updated_track_number, index_time) == track_number
                                if X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.VEHICLE_CANDIDATE || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.VEHICLE_CONFIRMED || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.PEDESTRIAN_CANDIDATE || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.PEDESTRIAN_CONFIRMED || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.MOTOR_BIKE_CANDIDATE || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.MOTOR_BIKE_CONFIRMED || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.BICYCLE_CANDIDATE || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.BICYCLE_CONFIRMED || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.TRUCK_CANDIDATE || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.TRUCK_CONFIRMED

                                    Track_Assigned_Flag = 1;
                                    break
                                end
                            end
                        end
                    end

                    if Track_Assigned_Flag == 1
                        X_est(:, track_number, index_time) = X_updated(:, track_number, index_time);
                        P_est(:, :, track_number, index_time) = P_updated(:, :, track_number, index_time);
                        Track_Assigned_Flag = 0;
                    end
                end
            end


            % Creation
            for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                    % SBEV ROI
                    if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                            && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX

                        for updated_track_number = 1:FUSION_TRACK.TRACK_NUMBER
                            if Association_Map_Total(updated_track_number, index_time) ~= 0
                                if track_number == Association_Map_Total(updated_track_number, index_time)
                                    Track_Assigned_Flag = 1;
                                    break
                                end
                            end
                        end

                        if Track_Assigned_Flag == 0

                            if sum(Association_Map_Total(track_number, index_time)) == 0

                                Association_Map_Total(track_number, index_time) = track_number;

                                % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
                                X_est(TRACKING.REL_POS_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                                X_est(TRACKING.REL_POS_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                                X_est(TRACKING.REL_VEL_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                                X_est(TRACKING.REL_VEL_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);

                                X_est(TRACKING.HEADING_ANGLE_RATE, track_number, index_time) = 0.002;

                                X_est(TRACKING.HEADING_ANGLE, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                                X_est(TRACKING.WIDTH, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                                X_est(TRACKING.LENGTH, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                                X_est(TRACKING.SHAPE, track_number, index_time) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                                X_est(TRACKING.MOTION, track_number, index_time) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                                P_est(:,:,track_number, index_time) = eye(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);
                            end                            
                        end
                        Track_Assigned_Flag = 0;
                    end
                end
            end


            % Deletion
            Fusion_Object_Exist_Flag = 0;
            for i_X_est = 1:FUSION_TRACK.TRACK_NUMBER
                if sum(Association_Map_Total(i_X_est, index_time)) ~= 0
                    for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                        if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                            % SBEV ROI
                            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                                    && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX

                                if Association_Map_Total(i_X_est, index_time) == track_number

                                    Fusion_Object_Exist_Flag = 1;
                                    break
                                end
                            end
                        end
                    end

                    if Fusion_Object_Exist_Flag == 0

                        X_est(:, i_X_est, index_time) = zeros(STATE_NUMBER, 1);
                        P_est(:,:,i_X_est, index_time) = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, 1);

                        Association_Map_Total(i_X_est, index_time) = 0;
                    end
                    Fusion_Object_Exist_Flag = 0;
                end
            end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Prediction
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic
        for track_number = 1:FUSION_TRACK.TRACK_NUMBER

            collision_probability_max = 0;

            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                % ROI
                if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                        && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX &&...
                        sum(P_est(:, :, track_number, index_time), 'all') ~= 0

                    Prediction_On(index_time, 1) = 1;

                    for index_pred = 1:TARGET_PRED_WINDOW/SAMPLE_TIME
                        if index_pred == 1

                            if abs( X_est(TRACKING.HEADING_ANGLE_RATE, track_number, index_time) ) < 0.001
                                tmp_heading_angle_rate = 0.001;
                            else
                                tmp_heading_angle_rate = X_est(TRACKING.HEADING_ANGLE_RATE, track_number, index_time);
                            end

                            % [x, y, vx, vy, heading angular rate]'
                            X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) + ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) ;

                            X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) + ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) + ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                                                                                        Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) + ...
                                                                                                        Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) * cos( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred, track_number) = tmp_heading_angle_rate;

                            X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time) +...
                                                                                                            X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred, track_number) * SAMPLE_TIME;
                            

                            dx_dvx = sin( tmp_heading_angle_rate * SAMPLE_TIME ) / tmp_heading_angle_rate;
                            dx_dvy = - ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) / tmp_heading_angle_rate;
                            dx_dw = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate^2 * sin( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME + ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate^2 * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) );


                            dy_dvx = ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) / tmp_heading_angle_rate;
                            dy_dvy = sin( tmp_heading_angle_rate * SAMPLE_TIME ) / tmp_heading_angle_rate;
                            dy_dw = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate^2 * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) + ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate^2 * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            dvx_dvx = cos( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvx_dvy = - sin( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvx_dw = - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME;

                            dvy_dvx = sin( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvy_dvy = cos( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvy_dw = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME;


                            J_A_CTRV = [1, 0, dx_dvx, dx_dvy, dx_dw
                                0, 1, dy_dvx, dy_dvy, dy_dw
                                0, 0, dvx_dvx, dvx_dvy, dvx_dw
                                0, 0, dvy_dvx, dvy_dvy, dvy_dw
                                0, 0, 0,       0,     1];

                            X_pred_window(TRACKING.WIDTH, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_pred_window(TRACKING.SHAPE, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_pred_window(TRACKING.MOTION, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);


                            P_pred_window(:, :, index_time, index_pred, track_number) = J_A_CTRV * P_est(:, :, track_number, index_time) * J_A_CTRV' + Q_CTRV;

                            tmp_heading_angle_rate = 0;

                        else
                            if abs( X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred - 1, track_number) ) < 0.001
                                tmp_heading_angle_rate = 0.001;
                            else
                                tmp_heading_angle_rate = X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred - 1, track_number);
                            end

                            % [x, y, vx, vy, heading angular rate]'
                            X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) = X_pred_window(TRACKING.REL_POS_X, index_time, index_pred - 1, track_number) + ...
                                X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) ;

                            X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) = X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred - 1, track_number) + ...
                                X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) + ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred, track_number) = X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                                                                                        X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred, track_number) = X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) + ...
                                                                                                        X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) * cos( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred, track_number) = tmp_heading_angle_rate;

                            X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number) = X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred - 1, track_number) +...
                                                                                                            X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred - 1, track_number) * SAMPLE_TIME;


                            dx_dvx = sin( tmp_heading_angle_rate * SAMPLE_TIME ) / tmp_heading_angle_rate;
                            dx_dvy = - ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) / tmp_heading_angle_rate;
                            dx_dw = X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred -1, track_number) / tmp_heading_angle_rate * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate^2 * sin( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME + ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate^2 * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) );


                            dy_dvx = ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) / tmp_heading_angle_rate;
                            dy_dvy = sin( tmp_heading_angle_rate * SAMPLE_TIME ) / tmp_heading_angle_rate;
                            dy_dw = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate^2 * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) + ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate^2 * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            dvx_dvx = cos( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvx_dvy = - sin( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvx_dw = - X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME;

                            dvy_dvx = sin( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvy_dvy = cos( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvy_dw = X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME;


                            J_A_CTRV = [1, 0, dx_dvx, dx_dvy, dx_dw
                                0, 1, dy_dvx, dy_dvy, dy_dw
                                0, 0, dvx_dvx, dvx_dvy, dvx_dw
                                0, 0, dvy_dvx, dvy_dvy, dvy_dw
                                0, 0, 0,       0,     1];

                            X_pred_window(TRACKING.WIDTH, index_time, index_pred, track_number) = X_pred_window(TRACKING.WIDTH, index_time, index_pred - 1, track_number);
                            X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number) = X_pred_window(TRACKING.LENGTH, index_time, index_pred - 1, track_number);
                            X_pred_window(TRACKING.SHAPE, index_time, index_pred, track_number) = X_pred_window(TRACKING.SHAPE, index_time, index_pred - 1, track_number);
                            X_pred_window(TRACKING.MOTION, index_time, index_pred, track_number) = X_pred_window(TRACKING.MOTION, index_time, index_pred - 1, track_number);


                            P_pred_window(:, :, index_time, index_pred, track_number) = J_A_CTRV * P_pred_window(:, :, index_time, index_pred - 1, track_number) * J_A_CTRV' + Q_CTRV;
                        end

                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % Collision Probability
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if Collision_Probability_Switch == 1
                            if index_pred == 1
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = round(tmp_index*TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)

                                tmp_P_pred_window = P_pred_window([TRACKING.REL_POS_X, TRACKING.REL_POS_Y], [TRACKING.REL_POS_X, TRACKING.REL_POS_Y], index_time, index_pred, track_number); % [xx xy; yx yy]

                                tmp_sigma_x = sqrt(tmp_P_pred_window(1, 1));
                                tmp_sigma_y = sqrt(tmp_P_pred_window(2, 2));

                                tmp_y_f = EGO_VEHICLE.WIDTH/2 +...
                                          Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) +....
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_y_i = -EGO_VEHICLE.WIDTH/2 -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                                tmp_cdf_y_f = normcdf(tmp_y_f, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);
                                tmp_cdf_y_i = normcdf(tmp_y_i, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);

                                tmp_x_f = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_x_i = -EGO_VEHICLE.LENGTH -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time))*sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                                tmp_cdf_x_f = normcdf(tmp_x_f, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);
                                tmp_cdf_x_i = normcdf(tmp_x_i, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);

                                tmp_cdf_y_i_to_y_f = tmp_cdf_y_f - tmp_cdf_y_i;
                                tmp_cdf_x_i_to_x_f = tmp_cdf_x_f - tmp_cdf_x_i;

                                tmp_collision_probability = tmp_cdf_y_i_to_y_f * tmp_cdf_x_i_to_x_f;

                                collision_probability_total(index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number, index_time) = tmp_collision_probability; % prediction window, track_number, length(sim_time)

                                if tmp_collision_probability > collision_probability_max
                                    collision_probability_max = tmp_collision_probability;
                                end
                            end
                        else
                            if index_pred == 1 %TARGET_PRED_WINDOW/SAMPLE_TIME
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = tmp_index*TARGET_PRED_SAMPLE_RATE*10/(SAMPLE_TIME *100) *10;
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)
                                X_pred_window_SBEV(:, index_time, index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number) = X_pred_window(:, index_time, index_pred, track_number);
                            end
                        end
                    end

                    if Collision_Probability_Switch == 1
                        collision_probability_final(index_time, track_number) = collision_probability_max;
                    end
                end
            end
        end
        tmp_Execution_Time_for_prediction = toc;

        if Evaluation_of_Prediction_Switch
            if Prediction_On(index_time, 1) == 1
                Execution_Time_Total(index_time, 1) = tmp_Execution_Time_for_prediction;
                tmp_Execution_Time_for_prediction = 0;
            end
        end

        if Evaluation_Collision_Probability_Switch
            if Prediction_On(index_time, 1) == 1
                Collision_Probability(index_time, 1) = max( collision_probability_final(index_time, :) );

                if Collision_Probability(index_time, 1) >= COLLISION_PROBABILITY.THRESHOLD
                    Predict_Collision(index_time, 1) = COLLISION.PRECRASH;
                else
                    Predict_Collision(index_time, 1) = COLLISION.SAFE;
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Generate Timeseries Annotation
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if impact_section(Data_index,1) ~= 0 % precrash
                    if index_time >= Annotation_start_index && index_time <= Annotation_end_index
                        time_GT(index_time,1) = COLLISION.PRECRASH;
                    else
                        time_GT(index_time,1) = COLLISION.SAFE;
                    end

                else % safe
                    time_GT(index_time,1) = COLLISION.SAFE;
                end
            end
        end
    end
end


% if TARGET_PRED_EKF_CTRA
% 
% end


if TARGET_PRED_UKF_CTRV

    TRACKING_STATE_NUMBER = 5; % [x, y, vx, vy, heading angular rate]'
    
    TRACKING.REL_POS_X = 1;
    TRACKING.REL_POS_Y = 2;
    TRACKING.REL_VEL_X = 3;
    TRACKING.REL_VEL_Y = 4;
    TRACKING.HEADING_ANGLE_RATE = 5;

    % 추후 수정 필요
    TRACKING.HEADING_ANGLE = 6;
    TRACKING.WIDTH = 7;
    TRACKING.LENGTH = 8;
    TRACKING.SHAPE = 9;
    TRACKING.MOTION = 10;

    STATE_NUMBER = length(fieldnames(TRACKING)); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'

    X_est = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_est = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    X_sigma = zeros(TRACKING_STATE_NUMBER, 2*TRACKING_STATE_NUMBER + 1, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy, heading angular rate]'
    W = zeros(2*TRACKING_STATE_NUMBER + 1, FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    X_sigma_pred = zeros(TRACKING_STATE_NUMBER, 2*TRACKING_STATE_NUMBER + 1, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy, heading angular rate]'
    X_pred = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_pred = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    z_CTRV = zeros(TRACKING_STATE_NUMBER - 1, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy]'
    X_updated = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_updated = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    X_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_pred_window = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER);

    Association_Map_Total = zeros(FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    X_pred_window_SBEV = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER);


    KAPPA = 3 - TRACKING_STATE_NUMBER;
    

    x_variance_CTRV = 0.4;
    y_variance_CTRV = 0.2;
    w_variance_CTRV = 3.16*10^-4;

    Q_CTRV = [x_variance_CTRV*SAMPLE_TIME^4/4, 0, x_variance_CTRV*SAMPLE_TIME^3/2, 0, 0
              0, y_variance_CTRV*SAMPLE_TIME^4/4, 0, y_variance_CTRV*SAMPLE_TIME^3/2, 0
              x_variance_CTRV*SAMPLE_TIME^3/2, 0, x_variance_CTRV*SAMPLE_TIME, 0, 0
              0, y_variance_CTRV*SAMPLE_TIME^3, 0, y_variance_CTRV*SAMPLE_TIME, 0
              0, 0, 0, 0, w_variance_CTRV];
    
    TRACKING.RESIDUAL.DEFAULT_VALUE = 300;
    TRACKING.GATING.INPUT_NUMBER = 4; % y, x, vy, vx
    
    
    Association_Map_Updated = zeros(FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    
    H_CTRV = [1 0 0 0 0    % x
              0 1 0 0 0    % y
              0 0 1 0 0    % vx
              0 0 0 1 0];  % vy
               

%     x_e_CTRV = 1;
%     y_e_CTRV = 1;
%     vx_e_CTRV = 2;
%     vy_e_CTRV = 2;
%     w_e_CTRV = 15*pi/180;
% 
% %     R_CTRV = blkdiag(x_e_CTRV, y_e_CTRV, vx_e_CTRV, vy_e_CTRV, w_e_CTRV);
%     R_CTRV = ( diag([x_e_CTRV, y_e_CTRV, vx_e_CTRV, vy_e_CTRV])*SAMPLE_TIME ).^2;

    x_e_CTRV = 0.1;
    y_e_CTRV = 0.1;
    vx_e_CTRV = 0.1;
    vy_e_CTRV = 0.1;
    R_CTRV = diag([x_e_CTRV, y_e_CTRV, vx_e_CTRV, vy_e_CTRV]);


    GATING.Y_MIN                           = -2;
    GATING.Y_MAX                           = 2;
    GATING.X_MIN                           = -3.5;
    GATING.X_MAX                           = 3.5;
    GATING.VY_MIN                          = -1.5;
    GATING.VY_MAX                          = 1.5;
    GATING.VX_MIN                          = -1.5;
    GATING.VX_MAX                          = 1.5;


    % collision probability
    collision_probability_total = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % prediction window, track_number, length(sim_time)
    collision_probability_final = zeros(length(sim_time), FUSION_TRACK.TRACK_NUMBER);

    for index_time = Test_start_index:SBEV_Gen_Sample_Rate/SAMPLE_TIME:Test_end_index

        tmp_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(TRACKING.GATING.INPUT_NUMBER, FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);
        tmp_norm_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);

        if index_time >= 1223
            a = 1;
        end

        if index_time >= 1233
            a = 1;
        end
        if index_time >= 1644
            a = 1;
        end

        if index_time >= 1676
            a = 1;
        end
        if index_time >= 1722
            a = 1;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Tracking for error covariance
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Prediction
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if index_time == 1
                Association_Map_k_1 = zeros(FUSION_TRACK.TRACK_NUMBER, 1);
                Fusion_Track_k_1 = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
                P_Fusion_Track_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            else
                Association_Map_k_1 = Association_Map_Total(:, index_time - 1);
                Fusion_Track_k_1 = X_est(:, :, index_time - 1);
                P_Fusion_Track_k_1 = P_est(:,:,:, index_time - 1);
            end

            for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                if Association_Map_k_1(track_number,1) ~= 0
                    
                    if abs( Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) ) < 0.001
                        Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) = 0.001;
                    end

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Sampling Sigma Point
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    L = real( sqrtm( (TRACKING_STATE_NUMBER + KAPPA) * P_Fusion_Track_k_1(:, :, track_number) ) );
                    
                    tmp_i = 0;
                    while tmp_i <= 2*TRACKING_STATE_NUMBER
                        if tmp_i == 0
                            X_sigma(:, 1, track_number, index_time) = Fusion_Track_k_1(1:TRACKING_STATE_NUMBER, track_number);
                            W(tmp_i + 1, track_number, index_time) = KAPPA / (KAPPA + TRACKING_STATE_NUMBER);
                        elseif tmp_i <= TRACKING_STATE_NUMBER
                            X_sigma(:, tmp_i + 1, track_number, index_time) = Fusion_Track_k_1(1:TRACKING_STATE_NUMBER, track_number) + L(tmp_i, :)';
                            W(tmp_i + 1, track_number, index_time) = 0.5 / (KAPPA + TRACKING_STATE_NUMBER);
                        else
                            X_sigma(:, tmp_i + 1, track_number, index_time) =  Fusion_Track_k_1(1:TRACKING_STATE_NUMBER, track_number) - L(tmp_i - TRACKING_STATE_NUMBER, :)';
                            W(tmp_i + 1, track_number, index_time) = 0.5 / (KAPPA + TRACKING_STATE_NUMBER);
                        end
                        tmp_i = tmp_i + 1;
                    end

                    % [x, y, vx, vy, heading angular rate]'
                    for tmp_i = 1:2*TRACKING_STATE_NUMBER + 1

                        X_sigma_pred(TRACKING.REL_POS_X, tmp_i, track_number, index_time) = X_sigma(TRACKING.REL_POS_X, tmp_i, track_number, index_time) + ...
                            X_sigma(TRACKING.REL_VEL_X, tmp_i, track_number, index_time) / X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * sin( X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * SAMPLE_TIME ) - ...
                            X_sigma(TRACKING.REL_VEL_Y, tmp_i, track_number, index_time) / X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * ( 1 - cos( X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * SAMPLE_TIME ) ) ;

                        X_sigma_pred(TRACKING.REL_POS_Y, tmp_i, track_number, index_time) = X_sigma(TRACKING.REL_POS_Y, tmp_i, track_number, index_time) + ...
                            X_sigma(TRACKING.REL_VEL_X, tmp_i, track_number, index_time) / X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * ( 1 - cos( X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * SAMPLE_TIME ) ) + ...
                            X_sigma(TRACKING.REL_VEL_Y, tmp_i, track_number, index_time) / X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * sin( X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * SAMPLE_TIME );

                        X_sigma_pred(TRACKING.REL_VEL_X, tmp_i, track_number, index_time) = X_sigma(TRACKING.REL_VEL_X, tmp_i, track_number, index_time) * cos( X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * SAMPLE_TIME ) - ...
                                                                                            X_sigma(TRACKING.REL_VEL_Y, tmp_i, track_number, index_time) * sin( X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * SAMPLE_TIME );

                        X_sigma_pred(TRACKING.REL_VEL_Y, tmp_i, track_number, index_time) = X_sigma(TRACKING.REL_VEL_X, tmp_i, track_number, index_time) * sin( X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * SAMPLE_TIME ) + ...
                                                                                            X_sigma(TRACKING.REL_VEL_Y, tmp_i, track_number, index_time) * cos( X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) * SAMPLE_TIME );

                        X_sigma_pred(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time) = X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i, track_number, index_time);

                        X_pred(1:TRACKING_STATE_NUMBER, track_number, index_time) = X_pred(1:TRACKING_STATE_NUMBER, track_number, index_time) +...
                                                                                    W(tmp_i, track_number, index_time) * X_sigma_pred(:, tmp_i, track_number, index_time);


                    end

                    X_pred(TRACKING.HEADING_ANGLE, track_number, index_time) = Fusion_Track_k_1(TRACKING.HEADING_ANGLE, track_number) + Fusion_Track_k_1(TRACKING.HEADING_ANGLE_RATE, track_number) * SAMPLE_TIME;
                    X_pred(TRACKING.WIDTH, track_number, index_time) = Fusion_Track_k_1(TRACKING.WIDTH, track_number); % width
                    X_pred(TRACKING.LENGTH, track_number, index_time) = Fusion_Track_k_1(TRACKING.LENGTH, track_number); % length                    
                    X_pred(TRACKING.SHAPE, track_number, index_time) = Fusion_Track_k_1(TRACKING.SHAPE, track_number); % classification
                    X_pred(TRACKING.MOTION, track_number, index_time) = Fusion_Track_k_1(TRACKING.MOTION, track_number); % motion

                    for tmp_i = 1:2*TRACKING_STATE_NUMBER + 1
                        P_pred(:, :, track_number, index_time) = P_pred(:, :, track_number, index_time) + W(tmp_i, track_number, index_time) *...
                                                                ( X_sigma_pred(:, tmp_i, track_number, index_time) - X_pred(1:TRACKING_STATE_NUMBER, track_number, index_time) ) *...
                                                                ( X_sigma_pred(:, tmp_i, track_number, index_time) - X_pred(1:TRACKING_STATE_NUMBER, track_number, index_time) )';
                    end

                    P_pred(:, :, track_number, index_time) = P_pred(:, :, track_number, index_time) + Q_CTRV;
                end
            end

            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Correction
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
                if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0
                    for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                        if norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)], 2) ~= 0

                            if Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time) == Fusion_Track_k_1(TRACKING.SHAPE, track_number_k_1)

                                tmp_residual(1, track_number, track_number_k_1) = X_pred(TRACKING.REL_POS_Y, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time); % y
                                tmp_residual(2, track_number, track_number_k_1) = X_pred(TRACKING.REL_POS_X, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time); % x
                                tmp_residual(3, track_number, track_number_k_1) = X_pred(TRACKING.REL_VEL_Y, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time); % vy
                                tmp_residual(4, track_number, track_number_k_1) = X_pred(TRACKING.REL_VEL_X, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time); % vx
                                
                                tmp_norm_residual(track_number_k_1, track_number) = norm(tmp_residual(:, track_number, track_number_k_1),2);
                            end
                        end
                    end
                end
            end

            for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
                if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0

                    [~, sorted_track_number] = sort(tmp_norm_residual(track_number_k_1,:));
                    [~, sorted_track_number_k_1] = sort(tmp_norm_residual(:, sorted_track_number(1)));

                    if sorted_track_number_k_1(1) == track_number_k_1 && ...
                            tmp_residual(1, sorted_track_number(1), track_number_k_1) > GATING.Y_MIN && tmp_residual(1, sorted_track_number(1), track_number_k_1) < GATING.Y_MAX && ...
                            tmp_residual(2, sorted_track_number(1), track_number_k_1) > GATING.X_MIN && tmp_residual(2, sorted_track_number(1), track_number_k_1) < GATING.X_MAX

                        sigma_hat = zeros(TRACKING_STATE_NUMBER - 1, 2*TRACKING_STATE_NUMBER + 1);
                        z_hat = zeros(TRACKING_STATE_NUMBER - 1, 1);
                        
                        P_zz = zeros(TRACKING_STATE_NUMBER - 1, TRACKING_STATE_NUMBER - 1);
                        P_xz = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER - 1);

                        for tmp_i = 1:2*TRACKING_STATE_NUMBER + 1
                            sigma_hat(:, tmp_i) = H_CTRV * X_sigma_pred(:, tmp_i, track_number_k_1, index_time);
                            z_hat = z_hat + W(tmp_i, track_number_k_1, index_time) * sigma_hat(:, tmp_i);
                        end

                        for tmp_i = 1:2*TRACKING_STATE_NUMBER + 1
                            P_zz = P_zz + W(tmp_i, track_number_k_1, index_time) * ( sigma_hat(:, tmp_i) - z_hat ) * ( sigma_hat(:, tmp_i) - z_hat )';
                            P_xz = P_xz + W(tmp_i, track_number_k_1, index_time) *...
                                    ( X_sigma_pred(:, tmp_i, track_number_k_1, index_time) - X_pred(1:TRACKING_STATE_NUMBER, track_number_k_1, index_time) ) *...
                                    ( sigma_hat(:, tmp_i) - z_hat )';
                        end

                        P_zz = P_zz + R_CTRV;

                        z_CTRV(:,track_number_k_1,index_time) = [Fusion_Track([FUSION_TRACK.TRACKING.REL_POS_X, FUSION_TRACK.TRACKING.REL_POS_Y, FUSION_TRACK.TRACKING.REL_VEL_X, FUSION_TRACK.TRACKING.REL_VEL_Y], sorted_track_number(1), index_time)]...
                            - H_CTRV*X_pred(1:TRACKING_STATE_NUMBER, track_number_k_1, index_time);

                        X_updated(TRACKING.HEADING_ANGLE, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, sorted_track_number(1), index_time); % heading angle
                        X_updated(TRACKING.WIDTH, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, sorted_track_number(1), index_time); % width
                        X_updated(TRACKING.LENGTH, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, sorted_track_number(1), index_time); % length
                        X_updated(TRACKING.SHAPE, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, sorted_track_number(1), index_time); % classification
                        X_updated(TRACKING.MOTION, track_number_k_1, index_time) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, sorted_track_number(1), index_time); % motion attribute

                        Association_Map_Total(track_number_k_1, index_time) = sorted_track_number(1);
                    else
                        X_updated(TRACKING.HEADING_ANGLE, track_number_k_1, index_time) = X_pred(TRACKING.HEADING_ANGLE, track_number_k_1, index_time); % heading angle
                        X_updated(TRACKING.WIDTH, track_number_k_1, index_time) = X_pred(TRACKING.WIDTH, track_number_k_1, index_time); % width
                        X_updated(TRACKING.LENGTH, track_number_k_1, index_time) = X_pred(TRACKING.LENGTH, track_number_k_1, index_time); % length
                        X_updated(TRACKING.SHAPE, track_number_k_1, index_time) = X_pred(TRACKING.SHAPE, track_number_k_1, index_time); % classification
                        X_updated(TRACKING.MOTION, track_number_k_1, index_time) = X_pred(TRACKING.MOTION, track_number_k_1, index_time); % motion attribute
                    end
                    
                    K_CTRV = P_xz/P_zz;

                    X_updated(1:TRACKING_STATE_NUMBER, track_number_k_1, index_time) = X_pred(1:TRACKING_STATE_NUMBER, track_number_k_1, index_time) + K_CTRV * z_CTRV(:, track_number_k_1, index_time);
                    P_updated(:, :, track_number_k_1, index_time) = P_pred(:, :, track_number_k_1, index_time) - K_CTRV * H_CTRV * P_pred(:, :, track_number_k_1, index_time);

                    X_updated(TRACKING.HEADING_ANGLE, track_number_k_1, index_time) = X_updated(TRACKING.HEADING_ANGLE, track_number_k_1, index_time) + X_updated(TRACKING.HEADING_ANGLE_RATE, track_number_k_1, index_time) * SAMPLE_TIME; % heading angle
                end
            end


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Track Management
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Maintenance
            Track_Assigned_Flag = 0;

            for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                if norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)], 2) ~= 0

                    for updated_track_number = 1:FUSION_TRACK.TRACK_NUMBER
                        if Association_Map_Total(updated_track_number, index_time) ~= 0
                            if Association_Map_Total(updated_track_number, index_time) == track_number
                                if X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.VEHICLE_CANDIDATE || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.VEHICLE_CONFIRMED || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.PEDESTRIAN_CANDIDATE || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.PEDESTRIAN_CONFIRMED || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.MOTOR_BIKE_CANDIDATE || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.MOTOR_BIKE_CONFIRMED || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.BICYCLE_CANDIDATE || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.BICYCLE_CONFIRMED || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.TRUCK_CANDIDATE || ...
                                        X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.TRUCK_CONFIRMED

                                    Track_Assigned_Flag = 1;
                                    break
                                end
                            end
                        end
                    end

                    if Track_Assigned_Flag == 1
                        X_est(:, track_number, index_time) = X_updated(:, track_number, index_time);
                        P_est(:, :, track_number, index_time) = P_updated(:, :, track_number, index_time);
                        Track_Assigned_Flag = 0;
                    end
                end
            end


            % Creation
            for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                    % SBEV ROI
                    if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                            && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX

                        for updated_track_number = 1:FUSION_TRACK.TRACK_NUMBER
                            if Association_Map_Total(updated_track_number, index_time) ~= 0
                                if track_number == Association_Map_Total(updated_track_number, index_time)
                                    Track_Assigned_Flag = 1;
                                    break
                                end
                            end
                        end

                        if Track_Assigned_Flag == 0

                            if sum(Association_Map_Total(track_number, index_time)) == 0

                                Association_Map_Total(track_number, index_time) = track_number;

                                % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
                                X_est(TRACKING.REL_POS_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                                X_est(TRACKING.REL_POS_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                                X_est(TRACKING.REL_VEL_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                                X_est(TRACKING.REL_VEL_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);

                                %                                         X_est(TRACKING.HEADING_ANGLE_RATE, track_number, index_time) = 0.002;

                                X_est(TRACKING.HEADING_ANGLE, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                                X_est(TRACKING.WIDTH, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                                X_est(TRACKING.LENGTH, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                                X_est(TRACKING.SHAPE, track_number, index_time) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                                X_est(TRACKING.MOTION, track_number, index_time) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

%                                 P_est(:,:,track_number, index_time) = eye(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);
                                P_est(:,:,track_number, index_time) = blkdiag(eye(TRACKING_STATE_NUMBER - 1, TRACKING_STATE_NUMBER - 1), 0.06);
                            end
                        end
                        Track_Assigned_Flag = 0;
                    end
                end
            end


            % Deletion
            Fusion_Object_Exist_Flag = 0;
            for i_X_est = 1:FUSION_TRACK.TRACK_NUMBER
                if sum(Association_Map_Total(i_X_est, index_time)) ~= 0
                    for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                        if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                            % SBEV ROI
                            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                                    && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX

                                if Association_Map_Total(i_X_est, index_time) == track_number

                                    Fusion_Object_Exist_Flag = 1;
                                    break
                                end
                            end
                        end
                    end

                    if Fusion_Object_Exist_Flag == 0

                        X_est(:, i_X_est, index_time) = zeros(STATE_NUMBER, 1);
                        P_est(:,:,i_X_est, index_time) = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, 1);

                        Association_Map_Total(i_X_est, index_time) = 0;
                    end
                    Fusion_Object_Exist_Flag = 0;
                end
            end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Prediction
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic
        for track_number = 1:FUSION_TRACK.TRACK_NUMBER

            collision_probability_max = 0;

            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                % ROI
                if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                        && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX &&...
                        sum(P_est(:, :, track_number, index_time), 'all') ~= 0

                    Prediction_On(index_time, 1) = 1;

                    for index_pred = 1:TARGET_PRED_WINDOW/SAMPLE_TIME
                        if index_pred == 1

                            if abs( X_est(TRACKING.HEADING_ANGLE_RATE, track_number, index_time) ) < 0.001
                                tmp_heading_angle_rate = 0.001;
                            else
                                tmp_heading_angle_rate = X_est(TRACKING.HEADING_ANGLE_RATE, track_number, index_time);
                            end

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Sampling Sigma Point
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            tmp_X_sigma = zeros(TRACKING_STATE_NUMBER, 2*TRACKING_STATE_NUMBER + 1);
                            tmp_W = zeros(2*TRACKING_STATE_NUMBER + 1, 1);
                            tmp_X_sigma_pred = zeros(TRACKING_STATE_NUMBER, 2*TRACKING_STATE_NUMBER + 1);
                            tmp_X_pred = zeros(STATE_NUMBER, 1);
                            tmp_P_pred = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);


                            L = real( sqrtm( (TRACKING_STATE_NUMBER + KAPPA) * P_est(:, :, track_number, index_time) ) );

                            tmp_i = 0;
                            while tmp_i <= 2*TRACKING_STATE_NUMBER
                                if tmp_i == 0
                                    tmp_X_sigma(:, 1) = [Fusion_Track([FUSION_TRACK.TRACKING.REL_POS_X, FUSION_TRACK.TRACKING.REL_POS_Y, FUSION_TRACK.TRACKING.REL_VEL_X, FUSION_TRACK.TRACKING.REL_VEL_Y], track_number, index_time); tmp_heading_angle_rate];
                                    tmp_W(tmp_i + 1, 1) = KAPPA / (KAPPA + TRACKING_STATE_NUMBER);
                                elseif tmp_i <= TRACKING_STATE_NUMBER
                                    tmp_X_sigma(:, tmp_i + 1) = [Fusion_Track([FUSION_TRACK.TRACKING.REL_POS_X, FUSION_TRACK.TRACKING.REL_POS_Y, FUSION_TRACK.TRACKING.REL_VEL_X, FUSION_TRACK.TRACKING.REL_VEL_Y], track_number, index_time); tmp_heading_angle_rate] + L(tmp_i, :)';
                                    tmp_W(tmp_i + 1, 1) = 0.5 / (KAPPA + TRACKING_STATE_NUMBER);
                                else
                                    tmp_X_sigma(:, tmp_i + 1) =  [Fusion_Track([FUSION_TRACK.TRACKING.REL_POS_X, FUSION_TRACK.TRACKING.REL_POS_Y, FUSION_TRACK.TRACKING.REL_VEL_X, FUSION_TRACK.TRACKING.REL_VEL_Y], track_number, index_time); tmp_heading_angle_rate] - L(tmp_i - TRACKING_STATE_NUMBER, :)';
                                    tmp_W(tmp_i + 1, 1) = 0.5 / (KAPPA + TRACKING_STATE_NUMBER);
                                end
                                tmp_i = tmp_i + 1;
                            end

                            % [x, y, vx, vy, heading angular rate]'
                            for tmp_i = 1:2*TRACKING_STATE_NUMBER + 1

                                tmp_X_sigma_pred(TRACKING.REL_POS_X, tmp_i) = tmp_X_sigma(TRACKING.REL_POS_X, tmp_i) + ...
                                    tmp_X_sigma(TRACKING.REL_VEL_X, tmp_i) / tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * sin( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME ) - ...
                                    tmp_X_sigma(TRACKING.REL_VEL_Y, tmp_i) / tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * ( 1 - cos( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME ) ) ;

                                tmp_X_sigma_pred(TRACKING.REL_POS_Y, tmp_i) = tmp_X_sigma(TRACKING.REL_POS_Y, tmp_i) + ...
                                    tmp_X_sigma(TRACKING.REL_VEL_X, tmp_i) / tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * ( 1 - cos( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME ) ) + ...
                                    tmp_X_sigma(TRACKING.REL_VEL_Y, tmp_i) / tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * sin( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME );

                                tmp_X_sigma_pred(TRACKING.REL_VEL_X, tmp_i) = tmp_X_sigma(TRACKING.REL_VEL_X, tmp_i) * cos( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME ) - ...
                                                                                tmp_X_sigma(TRACKING.REL_VEL_Y, tmp_i) * sin( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME );

                                tmp_X_sigma_pred(TRACKING.REL_VEL_Y, tmp_i) = tmp_X_sigma(TRACKING.REL_VEL_X, tmp_i) * sin( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME ) + ...
                                                                                tmp_X_sigma(TRACKING.REL_VEL_Y, tmp_i) * cos( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME );

                                tmp_X_sigma_pred(TRACKING.HEADING_ANGLE_RATE, tmp_i) = tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i);

                                tmp_X_pred(1:TRACKING_STATE_NUMBER, 1) = tmp_X_pred(1:TRACKING_STATE_NUMBER, 1) + tmp_W(tmp_i, 1) * tmp_X_sigma_pred(:, tmp_i);
                            end

                            tmp_X_pred(TRACKING.HEADING_ANGLE, 1) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time) + tmp_heading_angle_rate * SAMPLE_TIME;
                            tmp_X_pred(TRACKING.WIDTH, 1) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number); % width
                            tmp_X_pred(TRACKING.LENGTH, 1) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number); % length
                            tmp_X_pred(TRACKING.SHAPE, 1) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number); % classification
                            tmp_X_pred(TRACKING.MOTION, 1) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number); % motion

                            X_pred_window(:, index_time, index_pred, track_number) = tmp_X_pred;

                            for tmp_i = 1:2*TRACKING_STATE_NUMBER + 1
                                tmp_P_pred = tmp_P_pred + tmp_W(tmp_i, 1) * ( tmp_X_sigma_pred(:, tmp_i) - tmp_X_pred(1:TRACKING_STATE_NUMBER, 1) ) * ( tmp_X_sigma_pred(:, tmp_i) - tmp_X_pred(1:TRACKING_STATE_NUMBER, 1) )';
                            end

                            P_pred_window(:, :, index_time, index_pred, track_number) = tmp_P_pred + Q_CTRV;

                            tmp_heading_angle_rate = 0;

                        else
                            if abs( X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred - 1, track_number) ) < 0.001
                                tmp_heading_angle_rate = 0.001;
                            else
                                tmp_heading_angle_rate = X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred - 1, track_number);
                            end


                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Sampling Sigma Point
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            tmp_X_sigma = zeros(TRACKING_STATE_NUMBER, 2*TRACKING_STATE_NUMBER + 1);
                            tmp_W = zeros(2*TRACKING_STATE_NUMBER + 1, 1);
                            tmp_X_sigma_pred = zeros(TRACKING_STATE_NUMBER, 2*TRACKING_STATE_NUMBER + 1);
                            tmp_X_pred = zeros(STATE_NUMBER, 1);
                            tmp_P_pred = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);


                            L = real( sqrtm( (TRACKING_STATE_NUMBER + KAPPA) * P_pred_window(:, :, index_time, index_pred - 1, track_number) ) );

                            tmp_i = 0;
                            while tmp_i <= 2*TRACKING_STATE_NUMBER
                                if tmp_i == 0
                                    tmp_X_sigma(:, 1) = X_pred_window(1:TRACKING_STATE_NUMBER, index_time, index_pred - 1, track_number);
                                    tmp_W(tmp_i + 1, 1) = KAPPA / (KAPPA + TRACKING_STATE_NUMBER);
                                elseif tmp_i <= TRACKING_STATE_NUMBER
                                    tmp_X_sigma(:, tmp_i + 1) = X_pred_window(1:TRACKING_STATE_NUMBER, index_time, index_pred - 1, track_number) + L(tmp_i, :)';
                                    tmp_W(tmp_i + 1, 1) = 0.5 / (KAPPA + TRACKING_STATE_NUMBER);
                                else
                                    tmp_X_sigma(:, tmp_i + 1) =  X_pred_window(1:TRACKING_STATE_NUMBER, index_time, index_pred - 1, track_number) - L(tmp_i - TRACKING_STATE_NUMBER, :)';
                                    tmp_W(tmp_i + 1, 1) = 0.5 / (KAPPA + TRACKING_STATE_NUMBER);
                                end
                                tmp_i = tmp_i + 1;
                            end

                            % [x, y, vx, vy, heading angular rate]'
                            for tmp_i = 1:2*TRACKING_STATE_NUMBER + 1

                                tmp_X_sigma_pred(TRACKING.REL_POS_X, tmp_i) = tmp_X_sigma(TRACKING.REL_POS_X, tmp_i) + ...
                                    tmp_X_sigma(TRACKING.REL_VEL_X, tmp_i) / tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * sin( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME ) - ...
                                    tmp_X_sigma(TRACKING.REL_VEL_Y, tmp_i) / tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * ( 1 - cos( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME ) ) ;

                                tmp_X_sigma_pred(TRACKING.REL_POS_Y, tmp_i) = tmp_X_sigma(TRACKING.REL_POS_Y, tmp_i) + ...
                                    tmp_X_sigma(TRACKING.REL_VEL_X, tmp_i) / tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * ( 1 - cos( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME ) ) + ...
                                    tmp_X_sigma(TRACKING.REL_VEL_Y, tmp_i) / tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * sin( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME );

                                tmp_X_sigma_pred(TRACKING.REL_VEL_X, tmp_i) = tmp_X_sigma(TRACKING.REL_VEL_X, tmp_i) * cos( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME ) - ...
                                                                                tmp_X_sigma(TRACKING.REL_VEL_Y, tmp_i) * sin( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME );

                                tmp_X_sigma_pred(TRACKING.REL_VEL_Y, tmp_i) = tmp_X_sigma(TRACKING.REL_VEL_X, tmp_i) * sin( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME ) + ...
                                                                                tmp_X_sigma(TRACKING.REL_VEL_Y, tmp_i) * cos( tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i) * SAMPLE_TIME );

                                tmp_X_sigma_pred(TRACKING.HEADING_ANGLE_RATE, tmp_i) = tmp_X_sigma(TRACKING.HEADING_ANGLE_RATE, tmp_i);

                                tmp_X_pred(1:TRACKING_STATE_NUMBER, 1) = tmp_X_pred(1:TRACKING_STATE_NUMBER, 1) + tmp_W(tmp_i, 1) * tmp_X_sigma_pred(:, tmp_i);
                            end

                            tmp_X_pred(TRACKING.HEADING_ANGLE, 1) = X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred - 1, track_number) + tmp_heading_angle_rate * SAMPLE_TIME; % heading angle
                            tmp_X_pred(TRACKING.WIDTH, 1) = X_pred_window(TRACKING.WIDTH, index_time, index_pred - 1, track_number); % width
                            tmp_X_pred(TRACKING.LENGTH, 1) = X_pred_window(TRACKING.LENGTH, index_time, index_pred - 1, track_number); % length
                            tmp_X_pred(TRACKING.SHAPE, 1) = X_pred_window(TRACKING.SHAPE, index_time, index_pred - 1, track_number); % classification
                            tmp_X_pred(TRACKING.MOTION, 1) = X_pred_window(TRACKING.MOTION, index_time, index_pred - 1, track_number); % motion

                            X_pred_window(:, index_time, index_pred, track_number) = tmp_X_pred;


                            for tmp_i = 1:2*TRACKING_STATE_NUMBER + 1
                                tmp_P_pred = tmp_P_pred + tmp_W(tmp_i, 1) * ( tmp_X_sigma_pred(:, tmp_i) - tmp_X_pred(1:TRACKING_STATE_NUMBER, 1) ) * ( tmp_X_sigma_pred(:, tmp_i) - tmp_X_pred(1:TRACKING_STATE_NUMBER, 1) )';
                            end

                            P_pred_window(:, :, index_time, index_pred, track_number) = tmp_P_pred + Q_CTRV;

                            tmp_heading_angle_rate = 0;
                        end

                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % Collision Probability
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if Collision_Probability_Switch == 1
                            if index_pred == 1
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = round(tmp_index*TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)

                                tmp_P_pred_window = P_pred_window([TRACKING.REL_POS_X, TRACKING.REL_POS_Y], [TRACKING.REL_POS_X, TRACKING.REL_POS_Y], index_time, index_pred, track_number); % [xx xy; yx yy]

                                tmp_sigma_x = sqrt(tmp_P_pred_window(1, 1));
                                tmp_sigma_y = sqrt(tmp_P_pred_window(2, 2));

                                tmp_y_f = EGO_VEHICLE.WIDTH/2 +...
                                          Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) +....
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_y_i = -EGO_VEHICLE.WIDTH/2 -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                                tmp_cdf_y_f = normcdf(tmp_y_f, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);
                                tmp_cdf_y_i = normcdf(tmp_y_i, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);

                                tmp_x_f = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_x_i = -EGO_VEHICLE.LENGTH -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time))*sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                                tmp_cdf_x_f = normcdf(tmp_x_f, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);
                                tmp_cdf_x_i = normcdf(tmp_x_i, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);

                                tmp_cdf_y_i_to_y_f = tmp_cdf_y_f - tmp_cdf_y_i;
                                tmp_cdf_x_i_to_x_f = tmp_cdf_x_f - tmp_cdf_x_i;

                                tmp_collision_probability = tmp_cdf_y_i_to_y_f * tmp_cdf_x_i_to_x_f;

                                collision_probability_total(index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number, index_time) = tmp_collision_probability; % prediction window, track_number, length(sim_time)

                                if tmp_collision_probability > collision_probability_max
                                    collision_probability_max = tmp_collision_probability;
                                end
                            end
                        else
                            if index_pred == 1 %TARGET_PRED_WINDOW/SAMPLE_TIME
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = tmp_index*TARGET_PRED_SAMPLE_RATE*10/(SAMPLE_TIME *100) *10;
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)
                                X_pred_window_SBEV(:, index_time, index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number) = X_pred_window(:, index_time, index_pred, track_number);
                            end
                        end
                    end

                    if Collision_Probability_Switch == 1
                        collision_probability_final(index_time, track_number) = collision_probability_max;
                    end
                end
            end
        end
        tmp_Execution_Time_for_prediction = toc;

        if Evaluation_of_Prediction_Switch
            if Prediction_On(index_time, 1) == 1
                Execution_Time_Total(index_time, 1) = tmp_Execution_Time_for_prediction;
                tmp_Execution_Time_for_prediction = 0;
            end
        end

        if Evaluation_Collision_Probability_Switch
            if Prediction_On(index_time, 1) == 1
                Collision_Probability(index_time, 1) = max( collision_probability_final(index_time, :) );

                if Collision_Probability(index_time, 1) >= COLLISION_PROBABILITY.THRESHOLD
                    Predict_Collision(index_time, 1) = COLLISION.PRECRASH;
                else
                    Predict_Collision(index_time, 1) = COLLISION.SAFE;
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Generate Timeseries Annotation
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if impact_section(Data_index,1) ~= 0 % precrash
                    if index_time >= Annotation_start_index && index_time <= Annotation_end_index
                        time_GT(index_time,1) = COLLISION.PRECRASH;
                    else
                        time_GT(index_time,1) = COLLISION.SAFE;
                    end

                else % safe
                    time_GT(index_time,1) = COLLISION.SAFE;
                end
            end
        end
    end
end


% if TARGET_PRED_UKF_CTRA
% 
% end


if TARGET_PRED_IMM_EKF_CTRV_CV
    
    TRACKING.CV.REL_POS_X = 1;
    TRACKING.CV.REL_POS_Y = 2;
    TRACKING.CV.REL_VEL_X = 3;
    TRACKING.CV.REL_VEL_Y = 4;

    CV_TRACKING_STATE_NUMBER = 4; % [x, y, vx, vy]'
    TRACKING.CV_STATE_NUMBER = CV_TRACKING_STATE_NUMBER;


    TRACKING.CTRV.REL_POS_X = 1;
    TRACKING.CTRV.REL_POS_Y = 2;
    TRACKING.CTRV.REL_VEL_X = 3;
    TRACKING.CTRV.REL_VEL_Y = 4;
    TRACKING.CTRV.HEADING_ANGLE_RATE = 5;

    CTRV_TRACKING_STATE_NUMBER = 5; % [x, y, vx, vy, heading angular rate]'
    TRACKING.CTRV_STATE_NUMBER = CTRV_TRACKING_STATE_NUMBER;
    
    % 추후 수정 필요
    TRACKING.REL_POS_X = 1;
    TRACKING.REL_POS_Y = 2;
    TRACKING.REL_VEL_X = 3;
    TRACKING.REL_VEL_Y = 4;
    TRACKING.HEADING_ANGLE_RATE = 5;
    TRACKING.HEADING_ANGLE = 6;
    TRACKING.WIDTH = 7;
    TRACKING.LENGTH = 8;
    TRACKING.SHAPE = 9;
    TRACKING.MOTION = 10;

    STATE_NUMBER = 10; % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    TRACKING_STATE_NUMBER = 5; % [x, y, vx, vy, heading angular rate]'

    TRACKING.STATE_NUMBER = STATE_NUMBER; % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    TRACKING.TRACKING_STATE_NUMBER = TRACKING_STATE_NUMBER; % [x, y, vx, vy, heading angular rate]'
    
    

    
    X_est = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_est = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    Association_Map_Total = zeros(FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    X_pred_window_SBEV = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER);

    X_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_pred_window = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER);

    X_CTRV_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_CTRV_pred_window = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER);

    X_CV_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER); % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_CV_pred_window = zeros(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER);

    
    TRACKING.RESIDUAL.DEFAULT_VALUE = 300;
    TRACKING.GATING.INPUT_NUMBER = 4; % y, x, vy, vx
    

    GATING.Y_MIN                           = -2;
    GATING.Y_MAX                           = 2;
    GATING.X_MIN                           = -3.5;
    GATING.X_MAX                           = 3.5;
    GATING.VY_MIN                          = -1.5;
    GATING.VY_MAX                          = 1.5;
    GATING.VX_MIN                          = -1.5;
    GATING.VX_MAX                          = 1.5;


    MODEL_TRANSITION_PROBABILITY = [0.981 0.019; 0.019 0.981];

    MODEL_CTRV_INITIAL_PROBABILITY = 0.5;
    MODEL_CV_INITIAL_PROBILITY = 0.5;


    A_CV = [1, 0, SAMPLE_TIME, 0 % x
            0, 1, 0, SAMPLE_TIME % y
            0, 0, 1, 0           % vx
            0, 0, 0, 1];         % vy

    H_CV = eye(4,4);

    x_variance_CV = 0.4;
    y_variance_CV = 0.2;

    
    Q_CV = [x_variance_CV*SAMPLE_TIME^3/3, 0, x_variance_CV*SAMPLE_TIME/2, 0   % x
              0, y_variance_CV*SAMPLE_TIME^3/3, 0, y_variance_CV*SAMPLE_TIME/2 % y
              x_variance_CV*SAMPLE_TIME/2, 0, x_variance_CV*SAMPLE_TIME, 0     % vx
              0, y_variance_CV*SAMPLE_TIME/2, 0, y_variance_CV*SAMPLE_TIME];   % vy


    R_CV = 0.5*eye(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER);


    x_variance_CTRV = 0.4;
    y_variance_CTRV = 0.2;
    w_variance_CTRV = 3.16*10^-4;

    Q_CTRV = [x_variance_CTRV*SAMPLE_TIME^4/4, 0, x_variance_CTRV*SAMPLE_TIME^3/2, 0, 0
              0, y_variance_CTRV*SAMPLE_TIME^4/4, 0, y_variance_CTRV*SAMPLE_TIME^3/2, 0
              x_variance_CTRV*SAMPLE_TIME^3/2, 0, x_variance_CTRV*SAMPLE_TIME, 0, 0
              0, y_variance_CTRV*SAMPLE_TIME^3, 0, y_variance_CTRV*SAMPLE_TIME, 0
              0, 0, 0, 0, w_variance_CTRV];


    H_CTRV = [1 0 0 0 0    % x
              0 1 0 0 0    % y
              0 0 1 0 0    % vx
              0 0 0 1 0];  % vy

    x_e_CTRV = 0.1;
    y_e_CTRV = 0.1;
    vx_e_CTRV = 0.1;
    vy_e_CTRV = 0.1;
    R_CTRV = diag([x_e_CTRV, y_e_CTRV, vx_e_CTRV, vy_e_CTRV]);


    


    % collision probability
    collision_probability_total = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % prediction window, track_number, length(sim_time)
    collision_probability_final = zeros(length(sim_time), FUSION_TRACK.TRACK_NUMBER);

    for index_time = Test_start_index:SBEV_Gen_Sample_Rate/SAMPLE_TIME:Test_end_index

        tmp_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(TRACKING.GATING.INPUT_NUMBER, FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);
        tmp_norm_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);

        if index_time >= 1224
            a = 1;
        end

        if index_time >= 1687
            a = 1;
        end

        if index_time >= 1676
            a = 1;
        end

        if index_time >= 1823
            a = 1;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Initialization
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Y = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER); % measurement

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Tracking for error covariance
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if index_time == 1
            model_CTRV_probability_k_1 = MODEL_CTRV_INITIAL_PROBABILITY * ones(FUSION_TRACK.TRACK_NUMBER, 1);
            model_CV_probability_k_1 = MODEL_CV_INITIAL_PROBILITY * ones(FUSION_TRACK.TRACK_NUMBER, 1);


            X_CTRV_k_1 = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            P_CTRV_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);

            X_CV_k_1 = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            P_CV_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            

            Association_Map_k_1 = zeros(FUSION_TRACK.TRACK_NUMBER, 1);
            Fusion_Track_k_1 = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            P_Fusion_Track_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
        else
            model_CTRV_probability_k_1 = model_CTRV_probability_k;
            model_CV_probability_k_1 = model_CV_probability_k;

            X_CTRV_k_1 = X_CTRV_k;
            P_CTRV_k_1 = P_CTRV_k;

            X_CV_k_1 = [X_CV_k(1:CV_TRACKING_STATE_NUMBER, :); X_CTRV_k_1(TRACKING.HEADING_ANGLE_RATE, :); X_CV_k(TRACKING_STATE_NUMBER + 1:STATE_NUMBER, :)];
            P_CV_k_1 = blkdiag(P_CV_k, P_CTRV_k_1(TRACKING.HEADING_ANGLE_RATE, TRACKING.HEADING_ANGLE_RATE, :));

            Association_Map_k_1 = Association_Map_Total(:, index_time - 1);
            Fusion_Track_k_1 = X_est(:, :, index_time - 1);
            P_Fusion_Track_k_1 = P_est(:,:,:, index_time - 1);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Gating ( measurement for FST(k-1) )

        for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
            if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0
                for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                    if norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)], 2) ~= 0

                        if Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time) == Fusion_Track_k_1(TRACKING.SHAPE, track_number_k_1)

                            tmp_residual(1, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_POS_Y, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time); % y
                            tmp_residual(2, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_POS_X, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time); % x
                            tmp_residual(3, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time); % vy
                            tmp_residual(4, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time); % vx

                            tmp_norm_residual(track_number_k_1, track_number) = norm(tmp_residual(:, track_number, track_number_k_1),2);
                        end
                    end
                end
            end
        end

        for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
            if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0

                [~, sorted_track_number] = sort(tmp_norm_residual(track_number_k_1,:));
                [~, sorted_track_number_k_1] = sort(tmp_norm_residual(:, sorted_track_number(1)));

                if sorted_track_number_k_1(1) == track_number_k_1 && ...
                            tmp_residual(1, sorted_track_number(1), track_number_k_1) > GATING.Y_MIN && tmp_residual(1, sorted_track_number(1), track_number_k_1) < GATING.Y_MAX && ...
                            tmp_residual(2, sorted_track_number(1), track_number_k_1) > GATING.X_MIN && tmp_residual(2, sorted_track_number(1), track_number_k_1) < GATING.X_MAX

                    Y(TRACKING.REL_POS_X, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_POS_Y, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_VEL_X, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_VEL_Y, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, sorted_track_number(1), index_time);

                    Y(TRACKING.HEADING_ANGLE, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, sorted_track_number(1), index_time);
                    Y(TRACKING.WIDTH, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, sorted_track_number(1), index_time);
                    Y(TRACKING.LENGTH, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, sorted_track_number(1), index_time);
                    Y(TRACKING.SHAPE, track_number_k_1) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, sorted_track_number(1), index_time);
                    Y(TRACKING.MOTION, track_number_k_1) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, sorted_track_number(1), index_time);

                    Association_Map_Total(track_number_k_1, index_time) = sorted_track_number(1);
                end
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Interaction
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [model_CTRV_probability_pred, X_CTRV_k_1_mixed, P_CTRV_k_1_mixed, model_CV_probability_pred, X_CV_k_1_mixed, P_CV_k_1_mixed] = interaction(model_CTRV_probability_k_1, model_CV_probability_k_1, X_CTRV_k_1, X_CV_k_1,...
                                                                                P_CTRV_k_1, P_CV_k_1, MODEL_TRANSITION_PROBABILITY, FUSION_TRACK.TRACK_NUMBER, TRACKING);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Model Individual Filtering
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [X_CTRV_k, P_CTRV_k, lambda_CTRV] = EKF_CTRV_model_state5(X_CTRV_k_1_mixed, P_CTRV_k_1_mixed, Y, Association_Map_Total(:, index_time), H_CTRV, Q_CTRV, R_CTRV, FUSION_TRACK.TRACK_NUMBER, TRACKING, SAMPLE_TIME);

        [X_CV_k, P_CV_k, lambda_CV] = KF_CV_model_state5(X_CV_k_1_mixed, P_CV_k_1_mixed, Y, Association_Map_Total(:, index_time), A_CV, H_CV, Q_CV, R_CV, FUSION_TRACK.TRACK_NUMBER, TRACKING);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Combination
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [model_CTRV_probability_k, model_CV_probability_k, X_combined, P_combined] = combination(model_CTRV_probability_pred, X_CTRV_k, P_CTRV_k, lambda_CTRV, model_CV_probability_pred, X_CV_k, P_CV_k, lambda_CV, FUSION_TRACK.TRACK_NUMBER, TRACKING);

        X_est(:, :, index_time) = X_combined;
        P_est(:, :, :, index_time) = P_combined;

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Track Management
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Creation
        Track_Assigned_Flag = 0;

        for track_number = 1:FUSION_TRACK.TRACK_NUMBER
            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                % SBEV ROI
                if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                        && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX

                    for updated_track_number = 1:FUSION_TRACK.TRACK_NUMBER
                        if Association_Map_Total(updated_track_number, index_time) ~= 0
                            if track_number == Association_Map_Total(updated_track_number, index_time)
                                Track_Assigned_Flag = 1;
                                break
                            end
                        end
                    end

                    if Track_Assigned_Flag == 0

                        if sum(Association_Map_Total(track_number, index_time)) == 0

                            Association_Map_Total(track_number, index_time) = track_number;

                            % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
                            X_est(TRACKING.REL_POS_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                            X_est(TRACKING.REL_POS_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                            X_est(TRACKING.REL_VEL_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                            X_est(TRACKING.REL_VEL_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);

                            X_est(TRACKING.HEADING_ANGLE, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                            X_est(TRACKING.WIDTH, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_est(TRACKING.LENGTH, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_est(TRACKING.SHAPE, track_number, index_time) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_est(TRACKING.MOTION, track_number, index_time) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                            P_est(:,:,track_number, index_time) = eye(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);


                            X_CTRV_k(TRACKING.REL_POS_X, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                            X_CTRV_k(TRACKING.REL_POS_Y, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                            X_CTRV_k(TRACKING.REL_VEL_X, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                            X_CTRV_k(TRACKING.REL_VEL_Y, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);

                            X_CTRV_k(TRACKING.HEADING_ANGLE, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                            X_CTRV_k(TRACKING.WIDTH, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_CTRV_k(TRACKING.LENGTH, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_CTRV_k(TRACKING.SHAPE, track_number) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_CTRV_k(TRACKING.MOTION, track_number) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                            P_CTRV_k(:,:,track_number) = eye(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);


                            X_CV_k(TRACKING.REL_POS_X, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                            X_CV_k(TRACKING.REL_POS_Y, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                            X_CV_k(TRACKING.REL_VEL_X, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                            X_CV_k(TRACKING.REL_VEL_Y, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);

                            X_CV_k(TRACKING.HEADING_ANGLE, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                            X_CV_k(TRACKING.WIDTH, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_CV_k(TRACKING.LENGTH, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_CV_k(TRACKING.SHAPE, track_number) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_CV_k(TRACKING.MOTION, track_number) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                            P_CV_k(:,:,track_number) = eye(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER);

                            model_CTRV_probability_k(track_number, 1) = MODEL_CTRV_INITIAL_PROBABILITY;
                            model_CV_probability_k(track_number, 1) = MODEL_CV_INITIAL_PROBILITY;
                        end
                    end
                    Track_Assigned_Flag = 0;
                end
            end
        end



        % Deletion
        Fusion_Object_Exist_Flag = 0;
        for i_X_est = 1:FUSION_TRACK.TRACK_NUMBER
            if sum(Association_Map_Total(i_X_est, index_time)) ~= 0
                for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                    if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                        % SBEV ROI
                        if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                                && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX

                            if Association_Map_Total(i_X_est, index_time) == track_number

                                Fusion_Object_Exist_Flag = 1;
                                break
                            end
                        end
                    end
                end

                if Fusion_Object_Exist_Flag == 0

                    X_est(:, i_X_est, index_time) = zeros(STATE_NUMBER, 1);
                    P_est(:,:,i_X_est, index_time) = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, 1);

                    X_CTRV_k(:, i_X_est) = zeros(STATE_NUMBER, 1);
                    P_CTRV_k(:,:,i_X_est) = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, 1);

                    X_CV_k(:, i_X_est) = zeros(TRACKING.STATE_NUMBER, 1);
                    P_CV_k(:, :, i_X_est) = zeros(TRACKING.CV_STATE_NUMBER, TRACKING.CV_STATE_NUMBER, 1);


                    Association_Map_Total(i_X_est, index_time) = 0;
                end
                Fusion_Object_Exist_Flag = 0;
            end
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Prediction
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic
        for track_number = 1:FUSION_TRACK.TRACK_NUMBER

            collision_probability_max = 0;

            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                % ROI
                if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                        && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX &&...
                        sum(P_est(:, :, track_number, index_time), 'all') ~= 0


                    Prediction_On(index_time, 1) = 1;

                    for index_pred = 1:TARGET_PRED_WINDOW/SAMPLE_TIME
                        if index_pred == 1

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Initialization for prediction
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            X_CTRV_for_prediction = zeros(STATE_NUMBER, 1);
                            P_CTRV_for_prediction = zeros(CTRV_TRACKING_STATE_NUMBER, CTRV_TRACKING_STATE_NUMBER);

                            X_CV_for_prediction = zeros(STATE_NUMBER, 1);
                            P_CV_for_prediction = zeros(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER);


                            X_CTRV_for_prediction(TRACKING.REL_POS_X, 1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.REL_POS_Y, 1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.REL_VEL_X, 1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.REL_VEL_Y, 1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);

                            X_CTRV_for_prediction(TRACKING.HEADING_ANGLE, 1) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.WIDTH, 1) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.LENGTH, 1) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.SHAPE, 1) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.MOTION, 1) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                            P_CTRV_for_prediction = P_est([TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y, TRACKING.HEADING_ANGLE_RATE],...
                                                          [TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y, TRACKING.HEADING_ANGLE_RATE], track_number, index_time);


                            X_CV_for_prediction = X_CTRV_for_prediction;

                            P_CV_for_prediction = P_est([TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y],...
                                                          [TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y], track_number, index_time);

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Model Individual Filtering
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            [X_pred_CTRV, P_pred_CTRV] = predict_EKF_CTRV_model(X_CTRV_for_prediction, P_CTRV_for_prediction, Q_CTRV, TRACKING, SAMPLE_TIME);
                            [X_pred_CV, P_pred_CV] = predict_KF_CV_model(X_CV_for_prediction, P_CV_for_prediction, A_CV, Q_CV, TRACKING);

                            X_CTRV_pred_window(:, index_time, index_pred, track_number) = X_pred_CTRV;
                            P_CTRV_pred_window(:, :, index_time, index_pred, track_number) = P_pred_CTRV;

                            X_CV_pred_window(:, index_time, index_pred, track_number) = X_pred_CV;
                            P_CV_pred_window(:, :, index_time, index_pred, track_number) = P_pred_CV;

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Combination
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            model_CTRV_for_prediction = model_CTRV_probability_k(track_number, 1);
                            model_CV_for_prediction = model_CV_probability_k(track_number, 1);

                            [X_combined_for_prediction, P_combined_for_prediction] = predict_combination(model_CTRV_for_prediction, X_pred_CTRV, P_pred_CTRV, model_CV_for_prediction, X_pred_CV, P_pred_CV, TRACKING);

                            X_pred_window(:, index_time, index_pred, track_number) = X_combined_for_prediction;
                            P_pred_window(:, :, index_time, index_pred, track_number) = P_combined_for_prediction;

                        else
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Initialization for prediction
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            X_CTRV_for_prediction = zeros(STATE_NUMBER, 1);
                            P_CTRV_for_prediction = zeros(CTRV_TRACKING_STATE_NUMBER, CTRV_TRACKING_STATE_NUMBER);

                            X_CV_for_prediction = zeros(STATE_NUMBER, 1);
                            P_CV_for_prediction = zeros(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER);


                            X_CTRV_for_prediction(:, 1) = X_CTRV_pred_window(:, index_time, index_pred - 1, track_number);
                            P_CTRV_for_prediction = P_CTRV_pred_window(:, :, index_time, index_pred - 1, track_number);

                            X_CV_for_prediction = X_CV_pred_window(:, index_time, index_pred - 1, track_number);
                            P_CV_for_prediction = P_CV_pred_window(:, :, index_time, index_pred - 1, track_number);

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Model Individual Filtering
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            [X_pred_CTRV, P_pred_CTRV] = predict_EKF_CTRV_model(X_CTRV_for_prediction, P_CTRV_for_prediction, Q_CTRV, TRACKING, SAMPLE_TIME);
                            [X_pred_CV, P_pred_CV] = predict_KF_CV_model(X_CV_for_prediction, P_CV_for_prediction, A_CV, Q_CV, TRACKING);

                            X_CTRV_pred_window(:, index_time, index_pred, track_number) = X_pred_CTRV;
                            P_CTRV_pred_window(:, :, index_time, index_pred, track_number) = P_pred_CTRV;

                            X_CV_pred_window(:, index_time, index_pred, track_number) = X_pred_CV;
                            P_CV_pred_window(:, :, index_time, index_pred, track_number) = P_pred_CV;


                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Combination
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            model_CTRV_for_prediction = model_CTRV_probability_k(track_number, 1);
                            model_CV_for_prediction = model_CV_probability_k(track_number, 1);

                            [X_combined_for_prediction, P_combined_for_prediction] = predict_combination(model_CTRV_for_prediction, X_pred_CTRV, P_pred_CTRV, model_CV_for_prediction, X_pred_CV, P_pred_CV, TRACKING);

                            X_pred_window(:, index_time, index_pred, track_number) = X_combined_for_prediction;
                            P_pred_window(:, :, index_time, index_pred, track_number) = P_combined_for_prediction;
                        end


                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % Collision Probability
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if Collision_Probability_Switch == 1
                            if index_pred == 1
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = tmp_index*TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME;
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)

                                tmp_P_pred_window = P_pred_window([TRACKING.REL_POS_X, TRACKING.REL_POS_Y], [TRACKING.REL_POS_X, TRACKING.REL_POS_Y], index_time, index_pred, track_number); % [xx xy; yx yy]

                                tmp_sigma_x = sqrt(tmp_P_pred_window(1, 1));
                                tmp_sigma_y = sqrt(tmp_P_pred_window(2, 2));

                                tmp_y_f = EGO_VEHICLE.WIDTH/2 +...
                                          Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) +....
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_y_i = -EGO_VEHICLE.WIDTH/2 -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                                tmp_cdf_y_f = normcdf(tmp_y_f, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);
                                tmp_cdf_y_i = normcdf(tmp_y_i, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);

                                tmp_x_f = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_x_i = -EGO_VEHICLE.LENGTH -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time))*sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                                tmp_cdf_x_f = normcdf(tmp_x_f, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);
                                tmp_cdf_x_i = normcdf(tmp_x_i, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);

                                tmp_cdf_y_i_to_y_f = tmp_cdf_y_f - tmp_cdf_y_i;
                                tmp_cdf_x_i_to_x_f = tmp_cdf_x_f - tmp_cdf_x_i;

                                tmp_collision_probability = tmp_cdf_y_i_to_y_f * tmp_cdf_x_i_to_x_f;

                                collision_probability_total(index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number, index_time) = tmp_collision_probability; % prediction window, track_number, length(sim_time)

                                if tmp_collision_probability > collision_probability_max
                                    collision_probability_max = tmp_collision_probability;
                                end
                            end
                        else
                            if index_pred == 1 %TARGET_PRED_WINDOW/SAMPLE_TIME
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = tmp_index*TARGET_PRED_SAMPLE_RATE*10/(SAMPLE_TIME *100) *10;
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)
                                X_pred_window_SBEV(:, index_time, index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number) = X_pred_window(:, index_time, index_pred, track_number);
                            end
                        end
                    end

                    if Collision_Probability_Switch == 1
                        collision_probability_final(index_time, track_number) = collision_probability_max;
                    end
                end
            end
        end
        tmp_Execution_Time_for_prediction = toc;

        if Evaluation_of_Prediction_Switch
            if Prediction_On(index_time, 1) == 1
                Execution_Time_Total(index_time, 1) = tmp_Execution_Time_for_prediction;
                tmp_Execution_Time_for_prediction = 0;
            end
        end

        if Evaluation_Collision_Probability_Switch
            if Prediction_On(index_time, 1) == 1
                Collision_Probability(index_time, 1) = max( collision_probability_final(index_time, :) );

                if Collision_Probability(index_time, 1) >= COLLISION_PROBABILITY.THRESHOLD
                    Predict_Collision(index_time, 1) = COLLISION.PRECRASH;
                else
                    Predict_Collision(index_time, 1) = COLLISION.SAFE;
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Generate Timeseries Annotation
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if impact_section(Data_index,1) ~= 0 % precrash
                    if index_time >= Annotation_start_index && index_time <= Annotation_end_index
                        time_GT(index_time,1) = COLLISION.PRECRASH;
                    else
                        time_GT(index_time,1) = COLLISION.SAFE;
                    end

                else % safe
                    time_GT(index_time,1) = COLLISION.SAFE;
                end
            end
        end
    end
end


if TARGET_PRED_IMM_EKF_CTRV_CV_CA

    TRACKING.CV.REL_POS_X = 1;
    TRACKING.CV.REL_POS_Y = 2;
    TRACKING.CV.REL_VEL_X = 3;
    TRACKING.CV.REL_VEL_Y = 4;

    CV_TRACKING_STATE_NUMBER = 4; % [x, y, vx, vy]'
    TRACKING.CV_STATE_NUMBER = CV_TRACKING_STATE_NUMBER;


    TRACKING.CA.REL_POS_X = 1;
    TRACKING.CA.REL_POS_Y = 2;
    TRACKING.CA.REL_VEL_X = 3;
    TRACKING.CA.REL_VEL_Y = 4;
    TRACKING.CA.REL_ACC_X = 5;
    TRACKING.CA.REL_ACC_Y = 6;

    CA_TRACKING_STATE_NUMBER = 6; % [x, y, vx, vy, ax, ay]'
    TRACKING.CA_STATE_NUMBER = CA_TRACKING_STATE_NUMBER;


    TRACKING.CTRV.REL_POS_X = 1;
    TRACKING.CTRV.REL_POS_Y = 2;
    TRACKING.CTRV.REL_VEL_X = 3;
    TRACKING.CTRV.REL_VEL_Y = 4;
    TRACKING.CTRV.HEADING_ANGLE_RATE = 5;

    CTRV_TRACKING_STATE_NUMBER = 5; % [x, y, vx, vy, heading angular rate]'
    TRACKING.CTRV_STATE_NUMBER = CTRV_TRACKING_STATE_NUMBER;


    TRACKING.REL_POS_X = 1;
    TRACKING.REL_POS_Y = 2;
    TRACKING.REL_VEL_X = 3;
    TRACKING.REL_VEL_Y = 4;
    TRACKING.REL_ACC_X = 5;
    TRACKING.REL_ACC_Y = 6;
    TRACKING.HEADING_ANGLE_RATE = 7;
    TRACKING.HEADING_ANGLE = 8;
    TRACKING.WIDTH = 9;
    TRACKING.LENGTH = 10;
    TRACKING.SHAPE = 11;
    TRACKING.MOTION = 12;

    STATE_NUMBER = 12;    
    TRACKING.STATE_NUMBER = STATE_NUMBER;

    TRACKING_STATE_NUMBER = 7; % [x, y, vx, vy, ax, ay, heading angular rate]'
    TRACKING.TRACKING_STATE_NUMBER = TRACKING_STATE_NUMBER;


    X_est = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy, ax, ay, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_est = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    Association_Map_Total = zeros(FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    X_pred_window_SBEV = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER);

    X_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER); % [x, y, vx, vy, ax, ay, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_pred_window = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER);

    X_CTRV_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER); % [x, y, vx, vy, ax, ay, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_CTRV_pred_window = zeros(CTRV_TRACKING_STATE_NUMBER, CTRV_TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER);

    X_CV_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER); % [x, y, vx, vy, ax, ay, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_CV_pred_window = zeros(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER);

    X_CA_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER); % [x, y, vx, vy, ax, ay, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_CA_pred_window = zeros(CA_TRACKING_STATE_NUMBER, CA_TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER);


    P_est_CTRV = zeros(CTRV_TRACKING_STATE_NUMBER, CTRV_TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    P_est_CV = zeros(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));
    P_est_CA = zeros(CA_TRACKING_STATE_NUMBER, CA_TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));



%     MODEL_TRANSITION_PROBABILITY = [0.99,  0.001, 0.009
%                                     0.001, 0.99,  0.009
%                                     0.005, 0.005, 0.99];

    MODEL_TRANSITION_PROBABILITY = [0.99,  0.005, 0.005
                                    0.005, 0.99,  0.005
                                    0.005, 0.005, 0.99];


    MODEL_CTRV_INITIAL_PROBABILITY = 0.333;
    MODEL_CV_INITIAL_PROBILITY = 0.333;
    MODEL_CA_INITIAL_PROBILITY = 0.333;


    x_variance_CTRV = 0.4;
%     x_variance_CTRV = 0.2;
    y_variance_CTRV = 0.2;
%     y_variance_CTRV = 0.1;
    
%     w_variance_CTRV = 3.16*10^-4;
    w_variance_CTRV = 4*10^-4;
    
%     w_variance_CTRV = 1;
%     w_variance_CTRV = 0.1;

    Q_CTRV = [x_variance_CTRV*SAMPLE_TIME^4/4, 0, x_variance_CTRV*SAMPLE_TIME^3/2, 0, 0
              0, y_variance_CTRV*SAMPLE_TIME^4/4, 0, y_variance_CTRV*SAMPLE_TIME^3/2, 0
              x_variance_CTRV*SAMPLE_TIME^3/2, 0, x_variance_CTRV*SAMPLE_TIME, 0, 0
              0, y_variance_CTRV*SAMPLE_TIME^3, 0, y_variance_CTRV*SAMPLE_TIME, 0
              0, 0, 0, 0, w_variance_CTRV];


    H_CTRV = [1 0 0 0 0    % x
              0 1 0 0 0    % y
              0 0 1 0 0    % vx
              0 0 0 1 0];  % vy

%     x_e_CTRV = 0.2;
%     y_e_CTRV = 0.2;
%     vx_e_CTRV = 0.2;
%     vy_e_CTRV = 0.2;

    x_e_CTRV = 5^2;
    y_e_CTRV = 5^2;
    vx_e_CTRV = 5^2;
    vy_e_CTRV = 5^2;
    R_CTRV = diag([x_e_CTRV, y_e_CTRV, vx_e_CTRV, vy_e_CTRV]);



    A_CV = [1, 0, SAMPLE_TIME, 0 % x
            0, 1, 0, SAMPLE_TIME % y
            0, 0, 1, 0           % vx
            0, 0, 0, 1];         % vy

    H_CV = eye(4,4);

%     x_variance_CV = 0.1;
%     y_variance_CV = 0.1;

    x_variance_CV = 0.1;
    y_variance_CV = 0.1;

%     x_variance_CV = 0.4;
%     y_variance_CV = 0.4;

    
    
    Q_CV = [x_variance_CV*SAMPLE_TIME^3/3, 0, x_variance_CV*SAMPLE_TIME/2, 0   % x
              0, y_variance_CV*SAMPLE_TIME^3/3, 0, y_variance_CV*SAMPLE_TIME/2 % y
              x_variance_CV*SAMPLE_TIME/2, 0, x_variance_CV*SAMPLE_TIME, 0     % vx
              0, y_variance_CV*SAMPLE_TIME/2, 0, y_variance_CV*SAMPLE_TIME];   % vy


%     R_CV = 0.5*eye(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER);
%     R_CV = 0.5*eye(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER);
    R_CV = 5^2*eye(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER);



    A_CA = [1, 0, SAMPLE_TIME, 0, 1/2*SAMPLE_TIME^2, 0 % x
            0, 1, 0, SAMPLE_TIME, 0, 1/2*SAMPLE_TIME^2 % y
            0, 0, 1, 0, SAMPLE_TIME, 0                 % vx
            0, 0, 0, 1, 0, SAMPLE_TIME                 % vy
            0, 0, 0, 0, 1, 0                           % ax
            0, 0, 0, 0, 0, 1];                         % ay

    H_CA = [1 0 0 0 0 0   % x
            0 1 0 0 0 0   % y
            0 0 1 0 0 0   % vx
            0 0 0 1 0 0   % vy
            0 0 0 0 1 0]; % ax 

%     H_CA = [1 0 0 0 0 0     % x
%             0 1 0 0 0 0     % y
%             0 0 1 0 0 0     % vx
%             0 0 0 1 0 0];   % vy
            

    x_variance_CA = 0.1;
    y_variance_CA = 0.1;


    Q_CA = [x_variance_CA*SAMPLE_TIME^5/20, 0, x_variance_CA*SAMPLE_TIME^4/8, 0, x_variance_CA*SAMPLE_TIME^3/6, 0 % x
            0, y_variance_CA*SAMPLE_TIME^5/20, 0, y_variance_CA*SAMPLE_TIME^4/8, 0, y_variance_CA*SAMPLE_TIME^3/6 % y
            x_variance_CA*SAMPLE_TIME^4/8, 0, x_variance_CA*SAMPLE_TIME^3/3, 0, x_variance_CA*SAMPLE_TIME^2/2, 0  % vx
            0, y_variance_CA*SAMPLE_TIME^4/8, 0, y_variance_CA*SAMPLE_TIME^3/3, 0, y_variance_CA*SAMPLE_TIME^2/2  % vy
            x_variance_CA*SAMPLE_TIME^3/6, 0, x_variance_CA*SAMPLE_TIME^2/2, 0, x_variance_CA*SAMPLE_TIME, 0      % ax
            0, y_variance_CA*SAMPLE_TIME^3/6, 0, y_variance_CA*SAMPLE_TIME^2/2, 0, y_variance_CA*SAMPLE_TIME];    % ay


%     R_CA = 0.5*eye(CA_TRACKING_STATE_NUMBER - 1, CA_TRACKING_STATE_NUMBER - 1);
%     R_CA = 0.2*eye(CA_TRACKING_STATE_NUMBER - 1, CA_TRACKING_STATE_NUMBER - 1);

    R_CA = blkdiag(5^2*eye(CA_TRACKING_STATE_NUMBER - 2, CA_TRACKING_STATE_NUMBER - 2), 0.3);


    TRACKING.RESIDUAL.DEFAULT_VALUE = 300;
    TRACKING.GATING.INPUT_NUMBER = 4; % y, x, vy, vx
    

    tmp_residual_total = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(TRACKING.GATING.INPUT_NUMBER, FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    GATING.Y_MIN                           = -2;
    GATING.Y_MAX                           = 2;
    GATING.X_MIN                           = -3.5;
    GATING.X_MAX                           = 3.5;
    GATING.VY_MIN                          = -1.5;
    GATING.VY_MAX                          = 1.5;
    GATING.VX_MIN                          = -1.5;
    GATING.VX_MAX                          = 1.5;

    % collision probability
    collision_probability_total = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % prediction window, track_number, length(sim_time)
    collision_probability_final = zeros(length(sim_time), FUSION_TRACK.TRACK_NUMBER);


    % execution time
    execution_time_gating_total = zeros(length(sim_time), 1);

    execution_time_interaction = zeros(length(sim_time), 1);

    execution_time_CV_filtering = zeros(length(sim_time), 1);
    execution_time_CA_filtering = zeros(length(sim_time), 1);
    execution_time_CTRV_filtering = zeros(length(sim_time), 1);

    execution_time_combination = zeros(length(sim_time), 1);


    model_CTRV_probability_total = zeros(length(sim_time), FUSION_TRACK.TRACK_NUMBER);
    model_CV_probability_total = zeros(length(sim_time), FUSION_TRACK.TRACK_NUMBER);
    model_CA_probability_total = zeros(length(sim_time), FUSION_TRACK.TRACK_NUMBER);




    for index_time = Test_start_index:SBEV_Gen_Sample_Rate/SAMPLE_TIME:Test_end_index

        tmp_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(TRACKING.GATING.INPUT_NUMBER, FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);
        tmp_norm_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);

        if index_time >= 1173
            a = 1;
        end

        if index_time >= 1220
            a = 1;
        end

        if index_time >= 1248
            a = 1;
        end

        if index_time >= 1299
            a = 1;
        end

        if index_time >= 1687
            a = 1;
        end

        if index_time >= 1676
            a = 1;
        end

        if index_time >= 1448
            a = 1;
        end

        if index_time >= 596
            a = 1;
        end

        if index_time >= 2123
            a = 1;
        end

        if index_time >= 2538
            a = 1;
        end

        if index_time >= 2558
            a = 1;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Initialization
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Y = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER); % measurement

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Tracking for error covariance
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if index_time == 1
            model_CTRV_probability_k_1 = MODEL_CTRV_INITIAL_PROBABILITY * ones(FUSION_TRACK.TRACK_NUMBER, 1);
            model_CV_probability_k_1 = MODEL_CV_INITIAL_PROBILITY * ones(FUSION_TRACK.TRACK_NUMBER, 1);
            model_CA_probability_k_1 = MODEL_CA_INITIAL_PROBILITY * ones(FUSION_TRACK.TRACK_NUMBER, 1);

            X_CTRV_k_1 = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            P_CTRV_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER); % [x, y, vx, vy, ax, ay, heading angular rate]' X [x, y, vx, vy, ax, ay, heading angular rate]'

            X_CV_k_1 = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            P_CV_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);

            X_CA_k_1 = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            P_CA_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);

            Association_Map_k_1 = zeros(FUSION_TRACK.TRACK_NUMBER, 1);
            Fusion_Track_k_1 = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            P_Fusion_Track_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
        else
            model_CTRV_probability_k_1 = model_CTRV_probability_k;
            model_CV_probability_k_1 = model_CV_probability_k;
            model_CA_probability_k_1 = model_CA_probability_k;


            X_CTRV_k_1 = [X_CTRV_k([TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y], :);...
                            X_CA_k([TRACKING.REL_ACC_X, TRACKING.REL_ACC_Y], :);...
                            X_CTRV_k(TRACKING.HEADING_ANGLE_RATE, :);...
                            X_CTRV_k(TRACKING_STATE_NUMBER + 1:STATE_NUMBER, :)];
            
            X_CV_k_1 = [X_CV_k([TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y], :);...
                        X_CA_k([TRACKING.REL_ACC_X, TRACKING.REL_ACC_Y], :);...
                        X_CTRV_k(TRACKING.HEADING_ANGLE_RATE, :);...
                        X_CV_k(TRACKING_STATE_NUMBER + 1:STATE_NUMBER, :)];

            X_CA_k_1 = [X_CA_k(1:CA_TRACKING_STATE_NUMBER, :); X_CTRV_k(TRACKING.HEADING_ANGLE_RATE, :); X_CA_k(TRACKING_STATE_NUMBER + 1:STATE_NUMBER, :)];

            for track_number = 1:FUSION_TRACK.TRACK_NUMBER

                P_CTRV_k_1(:, :, track_number) = blkdiag(P_CTRV_k([TRACKING.CTRV.REL_POS_X, TRACKING.CTRV.REL_POS_Y, TRACKING.CTRV.REL_VEL_X, TRACKING.CTRV.REL_VEL_Y], [TRACKING.CTRV.REL_POS_X, TRACKING.CTRV.REL_POS_Y, TRACKING.CTRV.REL_VEL_X, TRACKING.CTRV.REL_VEL_Y], track_number),...
                    P_CA_k([TRACKING.CA.REL_ACC_X, TRACKING.CA.REL_ACC_Y], [TRACKING.CA.REL_ACC_X, TRACKING.CA.REL_ACC_Y], track_number),...
                    P_CTRV_k(TRACKING.CTRV.HEADING_ANGLE_RATE, TRACKING.CTRV.HEADING_ANGLE_RATE, track_number));

                P_CV_k_1(:, :, track_number) = blkdiag(P_CV_k(:, :, track_number), P_CA_k([TRACKING.CA.REL_ACC_X, TRACKING.CA.REL_ACC_Y], [TRACKING.CA.REL_ACC_X, TRACKING.CA.REL_ACC_Y], track_number),...
                                                        P_CTRV_k(TRACKING.CTRV.HEADING_ANGLE_RATE, TRACKING.CTRV.HEADING_ANGLE_RATE, track_number));

                P_CA_k_1(:, :, track_number) = blkdiag(P_CA_k(:, :, track_number), P_CTRV_k(TRACKING.CTRV.HEADING_ANGLE_RATE, TRACKING.CTRV.HEADING_ANGLE_RATE, track_number));

            end

            Association_Map_k_1 = Association_Map_Total(:, index_time - 1);
            Fusion_Track_k_1 = X_est(:, :, index_time - 1);
            P_Fusion_Track_k_1 = P_est(:,:,:, index_time - 1);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Gating ( measurement for FST(k-1) )

        tic
        for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
            if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0
                for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                    if norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)], 2) ~= 0

                        if Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time) == Fusion_Track_k_1(TRACKING.SHAPE, track_number_k_1)

                            tmp_residual(1, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_POS_Y, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time); % y
                            tmp_residual(2, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_POS_X, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time); % x
                            tmp_residual(3, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time); % vy
                            tmp_residual(4, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time); % vx

                            tmp_residual_total(:, track_number, track_number_k_1, index_time) = tmp_residual(:, track_number, track_number_k_1);

                            tmp_norm_residual(track_number_k_1, track_number) = norm(tmp_residual(:, track_number, track_number_k_1),2);
                        end
                    end
                end
            end
        end

        for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
            if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0

                [~, sorted_track_number] = sort(tmp_norm_residual(track_number_k_1,:));
                [~, sorted_track_number_k_1] = sort(tmp_norm_residual(:, sorted_track_number(1)));

                if sorted_track_number_k_1(1) == track_number_k_1 && ...
                            tmp_residual(1, sorted_track_number(1), track_number_k_1) > GATING.Y_MIN && tmp_residual(1, sorted_track_number(1), track_number_k_1) < GATING.Y_MAX && ...
                            tmp_residual(2, sorted_track_number(1), track_number_k_1) > GATING.X_MIN && tmp_residual(2, sorted_track_number(1), track_number_k_1) < GATING.X_MAX

                    Y(TRACKING.REL_POS_X, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_POS_Y, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_VEL_X, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_VEL_Y, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_ACC_X, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_X, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_ACC_Y, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_Y, sorted_track_number(1), index_time);

                    Y(TRACKING.HEADING_ANGLE, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, sorted_track_number(1), index_time);
                    Y(TRACKING.WIDTH, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, sorted_track_number(1), index_time);
                    Y(TRACKING.LENGTH, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, sorted_track_number(1), index_time);
                    Y(TRACKING.SHAPE, track_number_k_1) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, sorted_track_number(1), index_time);
                    Y(TRACKING.MOTION, track_number_k_1) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, sorted_track_number(1), index_time);

                    Association_Map_Total(track_number_k_1, index_time) = sorted_track_number(1);
                end
            end
        end
        execution_time_gating_total(index_time, 1) = toc;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Interaction
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic
        [model_CTRV_probability_pred, X_CTRV_k_1_mixed, P_CTRV_k_1_mixed, model_CV_probability_pred, X_CV_k_1_mixed, P_CV_k_1_mixed, model_CA_probability_pred, X_CA_k_1_mixed, P_CA_k_1_mixed] =...
            interaction3model(model_CTRV_probability_k_1, model_CV_probability_k_1, model_CA_probability_k_1, X_CTRV_k_1, X_CV_k_1, X_CA_k_1, ...
                                                                                P_CTRV_k_1, P_CV_k_1, P_CA_k_1, MODEL_TRANSITION_PROBABILITY, FUSION_TRACK.TRACK_NUMBER, TRACKING);

%         [model_CTRV_probability_pred, X_CTRV_k_1_mixed, P_CTRV_k_1_mixed, model_CV_probability_pred, X_CV_k_1_mixed, P_CV_k_1_mixed, model_CA_probability_pred, X_CA_k_1_mixed, P_CA_k_1_mixed] =...
%             interaction3model_mtx(model_CTRV_probability_k_1, model_CV_probability_k_1, model_CA_probability_k_1, X_CTRV_k_1, X_CV_k_1, X_CA_k_1, ...
%                                                                                 P_CTRV_k_1, P_CV_k_1, P_CA_k_1, MODEL_TRANSITION_PROBABILITY, FUSION_TRACK.TRACK_NUMBER, TRACKING);


        execution_time_interaction(index_time, 1) = toc;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Model Individual Filtering
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic

        [X_CV_k, P_CV_k, lambda_CV] = KF_CV_model_state7(X_CV_k_1_mixed, P_CV_k_1_mixed, Y, Association_Map_Total(:, index_time), A_CV, H_CV, Q_CV, R_CV, FUSION_TRACK.TRACK_NUMBER, TRACKING);

        execution_time_CV_filtering(index_time, 1) = toc;

        tic

        [X_CA_k, P_CA_k, lambda_CA] = KF_CA_model_state7(X_CA_k_1_mixed, P_CA_k_1_mixed, Y, Association_Map_Total(:, index_time), A_CA, H_CA, Q_CA, R_CA, FUSION_TRACK.TRACK_NUMBER, TRACKING);

        execution_time_CA_filtering(index_time, 1) = toc;

        tic
        
        [X_CTRV_k, P_CTRV_k, lambda_CTRV] = EKF_CTRV_model_state7(X_CTRV_k_1_mixed, P_CTRV_k_1_mixed, Y, Association_Map_Total(:, index_time), H_CTRV, Q_CTRV, R_CTRV, FUSION_TRACK.TRACK_NUMBER, TRACKING, SAMPLE_TIME);

        execution_time_CTRV_filtering(index_time, 1) = toc;

        P_est_CV(:,:,:,index_time) = P_CV_k;
        P_est_CA(:,:,:,index_time) = P_CA_k;        
        P_est_CTRV(:,:,:,index_time) = P_CTRV_k;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Combination
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic
        [model_CTRV_probability_k, model_CV_probability_k, model_CA_probability_k, X_combined, P_combined] =...
            combination3model(model_CTRV_probability_pred, X_CTRV_k, P_CTRV_k, lambda_CTRV, model_CV_probability_pred, X_CV_k, P_CV_k, lambda_CV,...
            model_CA_probability_pred, X_CA_k, P_CA_k, lambda_CA, FUSION_TRACK.TRACK_NUMBER, TRACKING);

        execution_time_combination(index_time, 1) = toc;

        X_est(:, :, index_time) = X_combined;
        P_est(:, :, :, index_time) = P_combined;

        model_CTRV_probability_total(index_time, :) = model_CTRV_probability_k;
        model_CV_probability_total(index_time, :) = model_CV_probability_k;
        model_CA_probability_total(index_time, :) = model_CA_probability_k;

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Track Management
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Creation
        Track_Assigned_Flag = 0;
        for track_number = 1:FUSION_TRACK.TRACK_NUMBER
            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                % SBEV ROI
                if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                        && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX

                    for updated_track_number = 1:FUSION_TRACK.TRACK_NUMBER
                        if Association_Map_Total(updated_track_number, index_time) ~= 0
                            if track_number == Association_Map_Total(updated_track_number, index_time)
                                Track_Assigned_Flag = 1;
                                break
                            end
                        end
                    end

                    if Track_Assigned_Flag == 0

                        if sum(Association_Map_Total(track_number, index_time)) == 0

                            Association_Map_Total(track_number, index_time) = track_number;

                            % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
                            X_est(TRACKING.REL_POS_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                            X_est(TRACKING.REL_POS_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                            X_est(TRACKING.REL_VEL_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                            X_est(TRACKING.REL_VEL_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);
                            X_est(TRACKING.REL_ACC_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_X, track_number, index_time);
                            X_est(TRACKING.REL_ACC_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_Y, track_number, index_time);

                            X_est(TRACKING.HEADING_ANGLE, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                            X_est(TRACKING.WIDTH, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_est(TRACKING.LENGTH, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_est(TRACKING.SHAPE, track_number, index_time) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_est(TRACKING.MOTION, track_number, index_time) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                            P_est(:,:,track_number, index_time) = eye(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);


                            X_CTRV_k(TRACKING.REL_POS_X, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                            X_CTRV_k(TRACKING.REL_POS_Y, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                            X_CTRV_k(TRACKING.REL_VEL_X, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                            X_CTRV_k(TRACKING.REL_VEL_Y, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);

                            X_CTRV_k(TRACKING.HEADING_ANGLE, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                            X_CTRV_k(TRACKING.WIDTH, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_CTRV_k(TRACKING.LENGTH, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_CTRV_k(TRACKING.SHAPE, track_number) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_CTRV_k(TRACKING.MOTION, track_number) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                            %                                     P_CTRV_k(:,:,track_number) = blkdiag(eye(CTRV_TRACKING_STATE_NUMBER - 1, CTRV_TRACKING_STATE_NUMBER - 1), 0.1);
                            P_CTRV_k(:,:,track_number) = blkdiag(eye(CTRV_TRACKING_STATE_NUMBER - 1, CTRV_TRACKING_STATE_NUMBER - 1), 0.06);
                            %                                     P_CTRV_k(:,:,track_number) = blkdiag(eye(CTRV_TRACKING_STATE_NUMBER - 1, CTRV_TRACKING_STATE_NUMBER - 1), 0.08);
                            %                                     P_CTRV_k(:,:,track_number) = blkdiag(eye(CTRV_TRACKING_STATE_NUMBER - 1, CTRV_TRACKING_STATE_NUMBER - 1), 0.02);


                            X_CV_k(TRACKING.REL_POS_X, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                            X_CV_k(TRACKING.REL_POS_Y, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                            X_CV_k(TRACKING.REL_VEL_X, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                            X_CV_k(TRACKING.REL_VEL_Y, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);

                            X_CV_k(TRACKING.HEADING_ANGLE, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                            X_CV_k(TRACKING.WIDTH, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_CV_k(TRACKING.LENGTH, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_CV_k(TRACKING.SHAPE, track_number) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_CV_k(TRACKING.MOTION, track_number) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                            P_CV_k(:,:,track_number) = eye(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER);



                            X_CA_k(TRACKING.REL_POS_X, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                            X_CA_k(TRACKING.REL_POS_Y, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                            X_CA_k(TRACKING.REL_VEL_X, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                            X_CA_k(TRACKING.REL_VEL_Y, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);
                            X_CA_k(TRACKING.REL_ACC_X, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_X, track_number, index_time);
                            X_CA_k(TRACKING.REL_ACC_Y, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_Y, track_number, index_time);

                            X_CA_k(TRACKING.HEADING_ANGLE, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                            X_CA_k(TRACKING.WIDTH, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_CA_k(TRACKING.LENGTH, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_CA_k(TRACKING.SHAPE, track_number) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_CA_k(TRACKING.MOTION, track_number) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                            P_CA_k(:,:,track_number) = eye(CA_TRACKING_STATE_NUMBER, CA_TRACKING_STATE_NUMBER);



                            model_CTRV_probability_k(track_number, 1) = MODEL_CTRV_INITIAL_PROBABILITY;
                            model_CV_probability_k(track_number, 1) = MODEL_CV_INITIAL_PROBILITY;
                            model_CA_probability_k(track_number, 1) = MODEL_CA_INITIAL_PROBILITY;
                        end
                    end
                    Track_Assigned_Flag = 0;
                end
            end
        end

        model_CTRV_probability_total(index_time, :) = model_CTRV_probability_k;
        model_CV_probability_total(index_time, :) = model_CV_probability_k;
        model_CA_probability_total(index_time, :) = model_CA_probability_k;



        % Deletion
        Fusion_Object_Exist_Flag = 0;
        for i_X_est = 1:FUSION_TRACK.TRACK_NUMBER
            if sum(Association_Map_Total(i_X_est, index_time)) ~= 0
                for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                    if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                        % SBEV ROI
                        if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                                && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX

                            if Association_Map_Total(i_X_est, index_time) == track_number

                                Fusion_Object_Exist_Flag = 1;
                                break
                            end
                        end
                    end
                end

                if Fusion_Object_Exist_Flag == 0

                    X_est(:, i_X_est, index_time) = zeros(STATE_NUMBER, 1);
                    P_est(:,:,i_X_est, index_time) = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, 1);

                    X_CTRV_k(:, i_X_est) = zeros(STATE_NUMBER, 1);
                    P_CTRV_k(:,:,i_X_est) = zeros(TRACKING.CTRV_STATE_NUMBER, TRACKING.CTRV_STATE_NUMBER, 1);

                    X_CV_k(:, i_X_est) = zeros(TRACKING.STATE_NUMBER, 1);
                    P_CV_k(:, :, i_X_est) = zeros(TRACKING.CV_STATE_NUMBER, TRACKING.CV_STATE_NUMBER, 1);

                    X_CA_k(:, i_X_est) = zeros(TRACKING.STATE_NUMBER, 1);
                    P_CA_k(:, :, i_X_est) = zeros(TRACKING.CA_STATE_NUMBER, TRACKING.CA_STATE_NUMBER, 1);


                    Association_Map_Total(i_X_est, index_time) = 0;
                end
                Fusion_Object_Exist_Flag = 0;
            end
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Prediction
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic
        for track_number = 1:FUSION_TRACK.TRACK_NUMBER

            collision_probability_max = 0;

            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                % ROI
                if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                        && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX &&...
                        sum(P_est(:, :, track_number, index_time), 'all') ~= 0


                    Prediction_On(index_time, 1) = 1;

                    for index_pred = 1:TARGET_PRED_WINDOW/SAMPLE_TIME
                        if index_pred == 1

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Initialization for prediction
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            X_CTRV_for_prediction = zeros(STATE_NUMBER, 1);
                            P_CTRV_for_prediction = zeros(CTRV_TRACKING_STATE_NUMBER, CTRV_TRACKING_STATE_NUMBER);

                            X_CV_for_prediction = zeros(STATE_NUMBER, 1);
                            P_CV_for_prediction = zeros(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER);

                            X_CA_for_prediction = zeros(STATE_NUMBER, 1);
                            P_CA_for_prediction = zeros(CA_TRACKING_STATE_NUMBER, CA_TRACKING_STATE_NUMBER);


                            X_CTRV_for_prediction(TRACKING.REL_POS_X, 1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.REL_POS_Y, 1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.REL_VEL_X, 1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.REL_VEL_Y, 1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.REL_ACC_X, 1) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_X, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.REL_ACC_Y, 1) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_Y, track_number, index_time);

                            X_CTRV_for_prediction(TRACKING.HEADING_ANGLE_RATE, 1) = X_est(TRACKING.HEADING_ANGLE_RATE, track_number, index_time);

                            X_CTRV_for_prediction(TRACKING.HEADING_ANGLE, 1) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.WIDTH, 1) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.LENGTH, 1) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.SHAPE, 1) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_CTRV_for_prediction(TRACKING.MOTION, 1) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                            P_CTRV_for_prediction = P_est([TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y, TRACKING.HEADING_ANGLE_RATE],...
                                                          [TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y, TRACKING.HEADING_ANGLE_RATE], track_number, index_time);


                            X_CV_for_prediction = X_CTRV_for_prediction;
                            P_CV_for_prediction = P_est([TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y],...
                                                          [TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y], track_number, index_time);


                            X_CA_for_prediction = X_CTRV_for_prediction;
                            P_CA_for_prediction = P_est([TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y, TRACKING.REL_ACC_X, TRACKING.REL_ACC_Y],...
                                                          [TRACKING.REL_POS_X, TRACKING.REL_POS_Y, TRACKING.REL_VEL_X, TRACKING.REL_VEL_Y, TRACKING.REL_ACC_X, TRACKING.REL_ACC_Y], track_number, index_time);

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Model Individual Filtering
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            [X_pred_CTRV, P_pred_CTRV] = predict_EKF_CTRV_model(X_CTRV_for_prediction, P_CTRV_for_prediction, Q_CTRV, TRACKING, SAMPLE_TIME);
                            [X_pred_CV, P_pred_CV] = predict_KF_CV_model(X_CV_for_prediction, P_CV_for_prediction, A_CV, Q_CV, TRACKING);
                            [X_pred_CA, P_pred_CA] = predict_KF_CA_model(X_CA_for_prediction, P_CA_for_prediction, A_CA, Q_CA, TRACKING);

                            X_CTRV_pred_window(:, index_time, index_pred, track_number) = X_pred_CTRV;
                            P_CTRV_pred_window(:, :, index_time, index_pred, track_number) = P_pred_CTRV;

                            X_CV_pred_window(:, index_time, index_pred, track_number) = X_pred_CV;
                            P_CV_pred_window(:, :, index_time, index_pred, track_number) = P_pred_CV;

                            X_CA_pred_window(:, index_time, index_pred, track_number) = X_pred_CA;
                            P_CA_pred_window(:, :, index_time, index_pred, track_number) = P_pred_CA;

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Combination
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            model_CTRV_for_prediction = model_CTRV_probability_k(track_number, 1);
                            model_CV_for_prediction = model_CV_probability_k(track_number, 1);
                            model_CA_for_prediction = model_CA_probability_k(track_number, 1);

                            [X_combined_for_prediction, P_combined_for_prediction] = predict_combination3model(model_CTRV_for_prediction, X_pred_CTRV, P_pred_CTRV,...
                                                                                                               model_CV_for_prediction, X_pred_CV, P_pred_CV,...
                                                                                                               model_CA_for_prediction, X_pred_CA, P_pred_CA, TRACKING);

                            X_pred_window(:, index_time, index_pred, track_number) = X_combined_for_prediction;
                            P_pred_window(:, :, index_time, index_pred, track_number) = P_combined_for_prediction;

                        else
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Initialization for prediction
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            X_CTRV_for_prediction = zeros(STATE_NUMBER, 1);
                            P_CTRV_for_prediction = zeros(CTRV_TRACKING_STATE_NUMBER, CTRV_TRACKING_STATE_NUMBER);

                            X_CV_for_prediction = zeros(STATE_NUMBER, 1);
                            P_CV_for_prediction = zeros(CV_TRACKING_STATE_NUMBER, CV_TRACKING_STATE_NUMBER);

                            X_CA_for_prediction = zeros(STATE_NUMBER, 1);
                            P_CA_for_prediction = zeros(CA_TRACKING_STATE_NUMBER, CA_TRACKING_STATE_NUMBER);


                            X_CTRV_for_prediction(:, 1) = X_CTRV_pred_window(:, index_time, index_pred - 1, track_number);
                            P_CTRV_for_prediction = P_CTRV_pred_window(:, :, index_time, index_pred - 1, track_number);

                            X_CV_for_prediction = X_CV_pred_window(:, index_time, index_pred - 1, track_number);
                            P_CV_for_prediction = P_CV_pred_window(:, :, index_time, index_pred - 1, track_number);

                            X_CA_for_prediction = X_CA_pred_window(:, index_time, index_pred - 1, track_number);
                            P_CA_for_prediction = P_CA_pred_window(:, :, index_time, index_pred - 1, track_number);

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Model Individual Filtering
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            [X_pred_CTRV, P_pred_CTRV] = predict_EKF_CTRV_model(X_CTRV_for_prediction, P_CTRV_for_prediction, Q_CTRV, TRACKING, SAMPLE_TIME);
                            [X_pred_CV, P_pred_CV] = predict_KF_CV_model(X_CV_for_prediction, P_CV_for_prediction, A_CV, Q_CV, TRACKING);
                            [X_pred_CA, P_pred_CA] = predict_KF_CA_model(X_CA_for_prediction, P_CA_for_prediction, A_CA, Q_CA, TRACKING);

                            X_CTRV_pred_window(:, index_time, index_pred, track_number) = X_pred_CTRV;
                            P_CTRV_pred_window(:, :, index_time, index_pred, track_number) = P_pred_CTRV;

                            X_CV_pred_window(:, index_time, index_pred, track_number) = X_pred_CV;
                            P_CV_pred_window(:, :, index_time, index_pred, track_number) = P_pred_CV;

                            X_CA_pred_window(:, index_time, index_pred, track_number) = X_pred_CA;
                            P_CA_pred_window(:, :, index_time, index_pred, track_number) = P_pred_CA;

                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            % Combination
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            model_CTRV_for_prediction = model_CTRV_probability_k(track_number, 1);
                            model_CV_for_prediction = model_CV_probability_k(track_number, 1);
                            model_CA_for_prediction = model_CA_probability_k(track_number, 1);

                            [X_combined_for_prediction, P_combined_for_prediction] = predict_combination3model(model_CTRV_for_prediction, X_pred_CTRV, P_pred_CTRV,...
                                                                                                               model_CV_for_prediction, X_pred_CV, P_pred_CV,...
                                                                                                               model_CA_for_prediction, X_pred_CA, P_pred_CA, TRACKING);

                            X_pred_window(:, index_time, index_pred, track_number) = X_combined_for_prediction;
                            P_pred_window(:, :, index_time, index_pred, track_number) = P_combined_for_prediction;

                        end

                        if ~isreal(X_combined_for_prediction)
                            a = 1;
                        end


                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % Collision Probability
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if Collision_Probability_Switch == 1
                            if index_pred == 1
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = round(tmp_index*TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)

                                tmp_P_pred_window = P_pred_window([TRACKING.REL_POS_X, TRACKING.REL_POS_Y], [TRACKING.REL_POS_X, TRACKING.REL_POS_Y], index_time, index_pred, track_number); % [xx xy; yx yy]

                                tmp_sigma_x = sqrt(tmp_P_pred_window(1, 1));
                                tmp_sigma_y = sqrt(tmp_P_pred_window(2, 2));

                                tmp_y_f = EGO_VEHICLE.WIDTH/2 +...
                                          Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) +....
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_y_i = -EGO_VEHICLE.WIDTH/2 -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

%                                 tmp_y_f = (EGO_VEHICLE.WIDTH + Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time))/2;
%                                 tmp_y_i = -(EGO_VEHICLE.WIDTH + Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time))/2;

                                tmp_cdf_y_f = normcdf(tmp_y_f, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);
                                tmp_cdf_y_i = normcdf(tmp_y_i, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);

%                                 tmp_x_f = 0;
%                                 tmp_x_i = -EGO_VEHICLE.LENGTH;

                                tmp_x_f = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_x_i = -EGO_VEHICLE.LENGTH -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time))*sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                                tmp_cdf_x_f = normcdf(tmp_x_f, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);
                                tmp_cdf_x_i = normcdf(tmp_x_i, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);

                                tmp_cdf_y_i_to_y_f = tmp_cdf_y_f - tmp_cdf_y_i;
                                tmp_cdf_x_i_to_x_f = tmp_cdf_x_f - tmp_cdf_x_i;

                                tmp_collision_probability = tmp_cdf_y_i_to_y_f * tmp_cdf_x_i_to_x_f;

                                collision_probability_total(index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number, index_time) = tmp_collision_probability; % prediction window, track_number, length(sim_time)

                                if tmp_collision_probability > collision_probability_max
                                    collision_probability_max = tmp_collision_probability;
                                end
                            end
                        else
                            if index_pred == 1 %TARGET_PRED_WINDOW/SAMPLE_TIME
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = tmp_index*TARGET_PRED_SAMPLE_RATE*10/(SAMPLE_TIME *100) *10;
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)
                                X_pred_window_SBEV(:, index_time, index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number) = X_pred_window(:, index_time, index_pred, track_number);
                            end
                        end
                    end

                    if Collision_Probability_Switch == 1
                        collision_probability_final(index_time, track_number) = collision_probability_max;
                    end
                end
            end
        end
        tmp_Execution_Time_for_prediction = toc;

        if Evaluation_of_Prediction_Switch
            if Prediction_On(index_time, 1) == 1
                Execution_Time_Total(index_time, 1) = tmp_Execution_Time_for_prediction;
                tmp_Execution_Time_for_prediction = 0;
            end
        end

        if Evaluation_Collision_Probability_Switch
            if Prediction_On(index_time, 1) == 1
                Collision_Probability(index_time, 1) = max( collision_probability_final(index_time, :) );

                if Collision_Probability(index_time, 1) >= COLLISION_PROBABILITY.THRESHOLD
                    Predict_Collision(index_time, 1) = COLLISION.PRECRASH;
                else
                    Predict_Collision(index_time, 1) = COLLISION.SAFE;
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Generate Timeseries Annotation
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if impact_section(Data_index,1) ~= 0 % precrash
                    if index_time >= Annotation_start_index && index_time <= Annotation_end_index
                        time_GT(index_time,1) = COLLISION.PRECRASH;
                    else
                        time_GT(index_time,1) = COLLISION.SAFE;
                    end

                else % safe
                    time_GT(index_time,1) = COLLISION.SAFE;
                end
            end
        end
    end
end

if TARGET_PRED_CTRV_for_IMM

    TRACKING.CTRV.REL_POS_X = 1;
    TRACKING.CTRV.REL_POS_Y = 2;
    TRACKING.CTRV.REL_VEL_X = 3;
    TRACKING.CTRV.REL_VEL_Y = 4;
    TRACKING.CTRV.HEADING_ANGLE_RATE = 5;

    CTRV_TRACKING_STATE_NUMBER = 5; % [x, y, vx, vy, heading angular rate]'
    TRACKING.CTRV_STATE_NUMBER = CTRV_TRACKING_STATE_NUMBER;

    TRACKING_STATE_NUMBER = 5; % [x, y, vx, vy, heading angular rate]'


    TRACKING.REL_POS_X = 1;
    TRACKING.REL_POS_Y = 2;
    TRACKING.REL_VEL_X = 3;
    TRACKING.REL_VEL_Y = 4;
    TRACKING.REL_ACC_X = 5;
    TRACKING.REL_ACC_Y = 6;
    TRACKING.HEADING_ANGLE_RATE = 7;
    TRACKING.HEADING_ANGLE = 8;
    TRACKING.WIDTH = 9;
    TRACKING.LENGTH = 10;
    TRACKING.SHAPE = 11;
    TRACKING.MOTION = 12;


    STATE_NUMBER = 12;    
    TRACKING.STATE_NUMBER = STATE_NUMBER;


    X_est = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % [x, y, vx, vy, ax, ay, heading angular rate]' + [heading angle, width, length, classification, motion]'
    P_est = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    X_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER); % [y vy ay x vx ax] + [width, length, heading angle, classification, motion]
    P_pred_window = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, FUSION_TRACK.TRACK_NUMBER);

    Association_Map_Total = zeros(FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    

    X_pred_window_SBEV = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER);

    x_variance_CTRV = 0.4;
    y_variance_CTRV = 0.2;
    w_variance_CTRV = 3.16*10^-4;

    Q_CTRV = [x_variance_CTRV*SAMPLE_TIME^4/4, 0, x_variance_CTRV*SAMPLE_TIME^3/2, 0, 0
              0, y_variance_CTRV*SAMPLE_TIME^4/4, 0, y_variance_CTRV*SAMPLE_TIME^3/2, 0
              x_variance_CTRV*SAMPLE_TIME^3/2, 0, x_variance_CTRV*SAMPLE_TIME, 0, 0
              0, y_variance_CTRV*SAMPLE_TIME^3, 0, y_variance_CTRV*SAMPLE_TIME, 0
              0, 0, 0, 0, w_variance_CTRV];


    H_CTRV = [1 0 0 0 0    % x
              0 1 0 0 0    % y
              0 0 1 0 0    % vx
              0 0 0 1 0];  % vy

    x_e_CTRV = 0.1;
    y_e_CTRV = 0.1;
    vx_e_CTRV = 0.1;
    vy_e_CTRV = 0.1;
    R_CTRV = diag([x_e_CTRV, y_e_CTRV, vx_e_CTRV, vy_e_CTRV]);


    TRACKING.RESIDUAL.DEFAULT_VALUE = 300;
    TRACKING.GATING.INPUT_NUMBER = 4; % y, x, vy, vx
    

    tmp_residual_total = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(TRACKING.GATING.INPUT_NUMBER, FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER, length(sim_time));

    GATING.Y_MIN                           = -2;
    GATING.Y_MAX                           = 2;
    GATING.X_MIN                           = -3.5;
    GATING.X_MAX                           = 3.5;
    GATING.VY_MIN                          = -1.5;
    GATING.VY_MAX                          = 1.5;
    GATING.VX_MIN                          = -1.5;
    GATING.VX_MAX                          = 1.5;

    % collision probability
    collision_probability_total = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, FUSION_TRACK.TRACK_NUMBER, length(sim_time)); % prediction window, track_number, length(sim_time)
    collision_probability_final = zeros(length(sim_time), FUSION_TRACK.TRACK_NUMBER);


    for index_time = Test_start_index:SBEV_Gen_Sample_Rate/SAMPLE_TIME:Test_end_index

        tmp_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(TRACKING.GATING.INPUT_NUMBER, FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);
        tmp_norm_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(FUSION_TRACK.TRACK_NUMBER, FUSION_TRACK.TRACK_NUMBER);

        if index_time >= 1173
            a = 1;
        end

        if index_time >= 1501
            a = 1;
        end
        if index_time >= 1670
            a = 1;
        end

        if index_time >= 1676
            a = 1;
        end
        if index_time >= 1646
            a = 1;
        end
        

        Y = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER); % measurement

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Tracking for error covariance
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if index_time == 1
            Association_Map_k_1 = zeros(FUSION_TRACK.TRACK_NUMBER, 1);
            Fusion_Track_k_1 = zeros(STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
            P_Fusion_Track_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, FUSION_TRACK.TRACK_NUMBER);
        else
            Association_Map_k_1 = Association_Map_Total(:, index_time - 1);
            Fusion_Track_k_1 = X_est(:, :, index_time - 1);
            P_Fusion_Track_k_1 = P_est(:,:,:, index_time - 1);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Gating ( measurement for FST(k-1) )

        for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
            if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0
                for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                    if norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)], 2) ~= 0

                        if Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time) == Fusion_Track_k_1(TRACKING.SHAPE, track_number_k_1)

                            tmp_residual(1, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_POS_Y, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time); % y
                            tmp_residual(2, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_POS_X, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time); % x
                            tmp_residual(3, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_VEL_Y, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time); % vy
                            tmp_residual(4, track_number, track_number_k_1) = Fusion_Track_k_1(TRACKING.REL_VEL_X, track_number_k_1) - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time); % vx

                            tmp_residual_total(:, track_number, track_number_k_1, index_time) = tmp_residual(:, track_number, track_number_k_1);

                            tmp_norm_residual(track_number_k_1, track_number) = norm(tmp_residual(:, track_number, track_number_k_1),2);
                        end
                    end
                end
            end
        end

        for track_number_k_1 = 1:FUSION_TRACK.TRACK_NUMBER
            if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0

                [~, sorted_track_number] = sort(tmp_norm_residual(track_number_k_1,:));
                [~, sorted_track_number_k_1] = sort(tmp_norm_residual(:, sorted_track_number(1)));

                if sorted_track_number_k_1(1) == track_number_k_1 && ...
                            tmp_residual(1, sorted_track_number(1), track_number_k_1) > GATING.Y_MIN && tmp_residual(1, sorted_track_number(1), track_number_k_1) < GATING.Y_MAX && ...
                            tmp_residual(2, sorted_track_number(1), track_number_k_1) > GATING.X_MIN && tmp_residual(2, sorted_track_number(1), track_number_k_1) < GATING.X_MAX

                    Y(TRACKING.REL_POS_X, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_POS_Y, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_VEL_X, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_VEL_Y, track_number_k_1) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_ACC_X, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_X, sorted_track_number(1), index_time);
                    Y(TRACKING.REL_ACC_Y, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_Y, sorted_track_number(1), index_time);

                    Y(TRACKING.HEADING_ANGLE, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, sorted_track_number(1), index_time);
                    Y(TRACKING.WIDTH, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, sorted_track_number(1), index_time);
                    Y(TRACKING.LENGTH, track_number_k_1) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, sorted_track_number(1), index_time);
                    Y(TRACKING.SHAPE, track_number_k_1) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, sorted_track_number(1), index_time);
                    Y(TRACKING.MOTION, track_number_k_1) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, sorted_track_number(1), index_time);

                    Association_Map_Total(track_number_k_1, index_time) = sorted_track_number(1);
                end
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EKF CTRV Filtering
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [X_CTRV_k, P_CTRV_k, lambda_CTRV] = EKF_CTRV_model_state7(Fusion_Track_k_1, P_Fusion_Track_k_1, Y, Association_Map_Total(:, index_time), H_CTRV, Q_CTRV, R_CTRV, FUSION_TRACK.TRACK_NUMBER, TRACKING, SAMPLE_TIME);


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Track Management
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Maintenance
        Track_Assigned_Flag = 0;

        for track_number = 1:FUSION_TRACK.TRACK_NUMBER
            if norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)], 2) ~= 0

                for updated_track_number = 1:FUSION_TRACK.TRACK_NUMBER
                    if Association_Map_Total(updated_track_number, index_time) ~= 0
                        if Association_Map_Total(updated_track_number, index_time) == track_number
                            if X_CTRV_k(TRACKING.SHAPE, updated_track_number) == SHAPE.PEDESTRIAN_CANDIDATE || ...
                                    X_CTRV_k(TRACKING.SHAPE, updated_track_number) == SHAPE.PEDESTRIAN_CONFIRMED || ...
                                    X_CTRV_k(TRACKING.SHAPE, updated_track_number) == SHAPE.BICYCLE_CANDIDATE || ...
                                    X_CTRV_k(TRACKING.SHAPE, updated_track_number) == SHAPE.BICYCLE_CONFIRMED || ...
                                    X_CTRV_k(TRACKING.SHAPE, updated_track_number) == SHAPE.E_SCOOTER_CANDIDATE || ...
                                    X_CTRV_k(TRACKING.SHAPE, updated_track_number) == SHAPE.E_SCOOTER_CONFIRMED || ...
                                    X_CTRV_k(TRACKING.SHAPE, updated_track_number) == SHAPE.VEHICLE_CONFIRMED

                                Track_Assigned_Flag = 1;
                                break
                            end
                        end
                    end
                end

                if Track_Assigned_Flag == 1
                    X_est(:, track_number, index_time) = X_CTRV_k(:, track_number);
                    P_est(:, :, track_number, index_time) = P_CTRV_k(:, :, track_number);
                    Track_Assigned_Flag = 0;
                end
            end
        end


        % Creation
        for track_number = 1:FUSION_TRACK.TRACK_NUMBER
            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                % SBEV ROI
                if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                        && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX

                    for updated_track_number = 1:FUSION_TRACK.TRACK_NUMBER
                        if Association_Map_Total(updated_track_number, index_time) ~= 0
                            if track_number == Association_Map_Total(updated_track_number, index_time)
                                Track_Assigned_Flag = 1;
                                break
                            end
                        end
                    end

                    if Track_Assigned_Flag == 0
                        for init_track_number = 1:FUSION_TRACK.TRACK_NUMBER
                            if sum(Association_Map_Total(init_track_number, index_time)) == 0
                                if Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time) == Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, init_track_number, index_time)

                                    Association_Map_Total(init_track_number, index_time) = track_number;

                                    % [x, y, vx, vy, heading angular rate]' + [heading angle, width, length, classification, motion]'
                                    X_est(TRACKING.REL_POS_X, init_track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
                                    X_est(TRACKING.REL_POS_Y, init_track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
                                    X_est(TRACKING.REL_VEL_X, init_track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time);
                                    X_est(TRACKING.REL_VEL_Y, init_track_number, index_time) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time);

                                    X_est(TRACKING.HEADING_ANGLE_RATE, init_track_number, index_time) = 0.002;

                                    X_est(TRACKING.HEADING_ANGLE, init_track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
                                    X_est(TRACKING.WIDTH, init_track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                                    X_est(TRACKING.LENGTH, init_track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                                    X_est(TRACKING.SHAPE, init_track_number, index_time) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                                    X_est(TRACKING.MOTION, init_track_number, index_time) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);

                                    P_est(:,:,init_track_number, index_time) = eye(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER);
                                    break
                                end
                            end
                        end
                    end
                    Track_Assigned_Flag = 0;
                end
            end
        end


        % Deletion
        Fusion_Object_Exist_Flag = 0;
        for i_X_est = 1:FUSION_TRACK.TRACK_NUMBER
            if sum(Association_Map_Total(i_X_est, index_time)) ~= 0
                for track_number = 1:FUSION_TRACK.TRACK_NUMBER
                    if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                        % SBEV ROI
                        if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                                && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX

                            if Association_Map_Total(i_X_est, index_time) == track_number

                                Fusion_Object_Exist_Flag = 1;
                                break
                            end
                        end
                    end
                end

                if Fusion_Object_Exist_Flag == 0

                    X_est(:, i_X_est, index_time) = zeros(STATE_NUMBER, 1);
                    P_est(:,:,i_X_est, index_time) = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, 1);

                    Association_Map_Total(i_X_est, index_time) = 0;
                end
                Fusion_Object_Exist_Flag = 0;
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Prediction
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic
        for track_number = 1:FUSION_TRACK.TRACK_NUMBER

            collision_probability_max = 0;

            if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time)^2 ~= 0
                % ROI
                if Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                        && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX

                    Prediction_On(index_time, 1) = 1;

                    for index_pred = 1:TARGET_PRED_WINDOW/SAMPLE_TIME
                        if index_pred == 1

                            if abs( X_est(TRACKING.HEADING_ANGLE_RATE, track_number, index_time) ) < 0.001
                                tmp_heading_angle_rate = 0.001;
                            else
                                tmp_heading_angle_rate = X_est(TRACKING.HEADING_ANGLE_RATE, track_number, index_time);
                            end

                            % [x, y, vx, vy, heading angular rate]'
                            X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) + ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) ;

                            X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) + ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) + ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                                                                                        Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) + ...
                                                                                                        Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) * cos( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred, track_number) = tmp_heading_angle_rate;

                            X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time) +...
                                                                                                            X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred, track_number) * SAMPLE_TIME;
                            

                            dx_dvx = sin( tmp_heading_angle_rate * SAMPLE_TIME ) / tmp_heading_angle_rate;
                            dx_dvy = - ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) / tmp_heading_angle_rate;
                            dx_dw = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate^2 * sin( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME + ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate^2 * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) );


                            dy_dvx = ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) / tmp_heading_angle_rate;
                            dy_dvy = sin( tmp_heading_angle_rate * SAMPLE_TIME ) / tmp_heading_angle_rate;
                            dy_dw = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate^2 * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) + ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) / tmp_heading_angle_rate^2 * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            dvx_dvx = cos( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvx_dvy = - sin( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvx_dw = - Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME;

                            dvy_dvx = sin( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvy_dvy = cos( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvy_dw = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME;


                            J_A_CTRV = [1, 0, dx_dvx, dx_dvy, dx_dw
                                0, 1, dy_dvx, dy_dvy, dy_dw
                                0, 0, dvx_dvx, dvx_dvy, dvx_dw
                                0, 0, dvy_dvx, dvy_dvy, dvy_dw
                                0, 0, 0,       0,     1];

                            X_pred_window(TRACKING.WIDTH, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time);
                            X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time);
                            X_pred_window(TRACKING.SHAPE, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time);
                            X_pred_window(TRACKING.MOTION, index_time, index_pred, track_number) = Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time);


                            P_pred_window(:, :, index_time, index_pred, track_number) = J_A_CTRV * P_est(:, :, track_number, index_time) * J_A_CTRV' + Q_CTRV;

                            tmp_heading_angle_rate = 0;

                        else
                            if abs( X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred - 1, track_number) ) < 0.001
                                tmp_heading_angle_rate = 0.001;
                            else
                                tmp_heading_angle_rate = X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred - 1, track_number);
                            end

                            % [x, y, vx, vy, heading angular rate]'
                            X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) = X_pred_window(TRACKING.REL_POS_X, index_time, index_pred - 1, track_number) + ...
                                X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) ;

                            X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) = X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred - 1, track_number) + ...
                                X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) + ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred, track_number) = X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                                                                                        X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred, track_number) = X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) + ...
                                                                                                        X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) * cos( tmp_heading_angle_rate * SAMPLE_TIME );

                            X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred, track_number) = tmp_heading_angle_rate;

                            X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number) = X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred - 1, track_number) +...
                                                                                                            X_pred_window(TRACKING.HEADING_ANGLE_RATE, index_time, index_pred - 1, track_number) * SAMPLE_TIME;


                            dx_dvx = sin( tmp_heading_angle_rate * SAMPLE_TIME ) / tmp_heading_angle_rate;
                            dx_dvy = - ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) / tmp_heading_angle_rate;
                            dx_dw = X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred -1, track_number) / tmp_heading_angle_rate * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate^2 * sin( tmp_heading_angle_rate * SAMPLE_TIME ) - ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME + ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate^2 * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) );


                            dy_dvx = ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) / tmp_heading_angle_rate;
                            dy_dvy = sin( tmp_heading_angle_rate * SAMPLE_TIME ) / tmp_heading_angle_rate;
                            dy_dw = Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time) / tmp_heading_angle_rate * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate^2 * ( 1 - cos( tmp_heading_angle_rate * SAMPLE_TIME ) ) + ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) / tmp_heading_angle_rate^2 * sin( tmp_heading_angle_rate * SAMPLE_TIME );

                            dvx_dvx = cos( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvx_dvy = - sin( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvx_dw = - X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME;

                            dvy_dvx = sin( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvy_dvy = cos( tmp_heading_angle_rate * SAMPLE_TIME );
                            dvy_dw = X_pred_window(TRACKING.REL_VEL_X, index_time, index_pred - 1, track_number) * cos( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME - ...
                                X_pred_window(TRACKING.REL_VEL_Y, index_time, index_pred - 1, track_number) * sin( tmp_heading_angle_rate * SAMPLE_TIME ) * SAMPLE_TIME;


                            J_A_CTRV = [1, 0, dx_dvx, dx_dvy, dx_dw
                                0, 1, dy_dvx, dy_dvy, dy_dw
                                0, 0, dvx_dvx, dvx_dvy, dvx_dw
                                0, 0, dvy_dvx, dvy_dvy, dvy_dw
                                0, 0, 0,       0,     1];

                            X_pred_window(TRACKING.WIDTH, index_time, index_pred, track_number) = X_pred_window(TRACKING.WIDTH, index_time, index_pred - 1, track_number);
                            X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number) = X_pred_window(TRACKING.LENGTH, index_time, index_pred - 1, track_number);
                            X_pred_window(TRACKING.SHAPE, index_time, index_pred, track_number) = X_pred_window(TRACKING.SHAPE, index_time, index_pred - 1, track_number);
                            X_pred_window(TRACKING.MOTION, index_time, index_pred, track_number) = X_pred_window(TRACKING.MOTION, index_time, index_pred - 1, track_number);


                            P_pred_window(:, :, index_time, index_pred, track_number) = J_A_CTRV * P_pred_window(:, :, index_time, index_pred - 1, track_number) * J_A_CTRV' + Q_CTRV;
                        end

                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % Collision Probability
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if Collision_Probability_Switch == 1
                            if index_pred == 1
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = round(tmp_index*TARGET_PRED_SAMPLE_RATE/SAMPLE_TIME);
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)

                                tmp_P_pred_window = P_pred_window([TRACKING.REL_POS_X, TRACKING.REL_POS_Y], [TRACKING.REL_POS_X, TRACKING.REL_POS_Y], index_time, index_pred, track_number); % [xx xy; yx yy]

                                tmp_sigma_x = sqrt(tmp_P_pred_window(1, 1));
                                tmp_sigma_y = sqrt(tmp_P_pred_window(2, 2));

                                tmp_y_f = EGO_VEHICLE.WIDTH/2 +...
                                          Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) +....
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_y_i = -EGO_VEHICLE.WIDTH/2 -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                                tmp_cdf_y_f = normcdf(tmp_y_f, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);
                                tmp_cdf_y_i = normcdf(tmp_y_i, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);

                                tmp_x_f = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                          Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                                tmp_x_i = -EGO_VEHICLE.LENGTH -...
                                           Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                           Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time))*sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                                tmp_cdf_x_f = normcdf(tmp_x_f, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);
                                tmp_cdf_x_i = normcdf(tmp_x_i, X_pred_window(TRACKING.REL_POS_X, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*cos(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_x);

                                tmp_cdf_y_i_to_y_f = tmp_cdf_y_f - tmp_cdf_y_i;
                                tmp_cdf_x_i_to_x_f = tmp_cdf_x_f - tmp_cdf_x_i;

                                tmp_collision_probability = tmp_cdf_y_i_to_y_f * tmp_cdf_x_i_to_x_f;

                                collision_probability_total(index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number, index_time) = tmp_collision_probability; % prediction window, track_number, length(sim_time)

                                if tmp_collision_probability > collision_probability_max
                                    collision_probability_max = tmp_collision_probability;
                                end
                            end
                        else
                            if index_pred == 1 %TARGET_PRED_WINDOW/SAMPLE_TIME
                                sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                                for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                    sample_time_total_for_collision_probability(tmp_index) = tmp_index*TARGET_PRED_SAMPLE_RATE*10/(SAMPLE_TIME *100) *10;
                                end
                            end

                            if ismember(index_pred, sample_time_total_for_collision_probability)
                                X_pred_window_SBEV(:, index_time, index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number) = X_pred_window(:, index_time, index_pred, track_number);
                            end
                        end
                    end

                    if Collision_Probability_Switch == 1
                        collision_probability_final(index_time, track_number) = collision_probability_max;
                    end
                end
            end
        end
        tmp_Execution_Time_for_prediction = toc;

        if Evaluation_of_Prediction_Switch
            if Prediction_On(index_time, 1) == 1
                Execution_Time_Total(index_time, 1) = tmp_Execution_Time_for_prediction;
                tmp_Execution_Time_for_prediction = 0;
            end
        end

        if Evaluation_Collision_Probability_Switch
            if Prediction_On(index_time, 1) == 1
                Collision_Probability(index_time, 1) = max( collision_probability_final(index_time, :) );

                if Collision_Probability(index_time, 1) >= COLLISION_PROBABILITY.THRESHOLD
                    Predict_Collision(index_time, 1) = COLLISION.PRECRASH;
                else
                    Predict_Collision(index_time, 1) = COLLISION.SAFE;
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Generate Timeseries Annotation
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if impact_section(Data_index,1) ~= 0 % precrash
                    if index_time >= Annotation_start_index && index_time <= Annotation_end_index
                        time_GT(index_time,1) = COLLISION.PRECRASH;
                    else
                        time_GT(index_time,1) = COLLISION.SAFE;
                    end

                else % safe
                    time_GT(index_time,1) = COLLISION.SAFE;
                end
            end
        end
    end
end


if TARGET_PRED_KF_CV == 0 && TARGET_PRED_KF_CA == 0 && TARGET_PRED_EKF_CTRV == 0 && TARGET_PRED_EKF_CTRA == 0 &&...
        TARGET_PRED_UKF_CTRV == 0 && TARGET_PRED_UKF_CTRA == 0 && TARGET_PRED_IMM_EKF_CTRV_CV == 0 && TARGET_PRED_IMM_EKF_CTRV_CV_CA == 0

    X_pred_window = 0;
    P_pred_window = 0;
end



if TARGET_PRED_IMM_UKF == 1
    Prob_ctrv_ini=0.5;
    Prob_cv_ini=0.5;
    
    Q_CTRV_IMM                                              = (diag([1 1 1*pi/180 2 15*pi/180])*SAMPLE_TIME).^2;
    R_CTRV_IMM                                              = (diag([5 5 30*pi/180])).^2;
    Q_CV_IMM                                                = (diag([1 1 1.5*pi/180 2 2])*SAMPLE_TIME).^2;
    R_CV_IMM                                                = (diag([5 5 30*pi/180])).^2;
    flag=0;
    x_ctrv_tmp=zeros(5,length(sim_time),10);
    P_ctrv_tmp=zeros(5,5,length(sim_time),10);
    x_cv_tmp=zeros(5,length(sim_time),10);
    P_cv_tmp=zeros(5,5,length(sim_time),10);
    X_pred=zeros(5,length(sim_time),10,FUSION_TRACK.TRACK_NUMBER); % [y x yaw v yawrate]
    P_pred=zeros(5,5,length(sim_time),10,FUSION_TRACK.TRACK_NUMBER);
    prob_matrix = [ 0.981 0.019; 0.019 0.981];
    
    for track_number = 1:FUSION_TRACK.TRACK_NUMBER
        for index_time = 1:length(sim_time)
            old_flag=flag;
            if (Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                    && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX)
                flag=1;
            else
                flag=0;
            end
            
            if old_flag==1
                old_flag=old_flag;
            end
            x=Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time);
            y=Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time);
            v=-norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time)],2);
            
            theta=-Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time);
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
            
            if  (Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) >= X_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_X, track_number, index_time) <= X_MAX ...
                    && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) >= Y_MIN && Fusion_Track(FUSION_TRACK.TRACKING.REL_POS_Y, track_number, index_time) <= Y_MAX)
                
                for pred_length=1:10
                    ts=pred_length*0.2;
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
else
    X_pred = zeros(5,length(sim_time),10,FUSION_TRACK.TRACK_NUMBER); % [y x yaw v yawrate]
    P_pred = zeros(5,5,length(sim_time),10,FUSION_TRACK.TRACK_NUMBER);
end
% %% Ego vehicle prediction
% % TJ_X(index_time,predict sample,Fallback strategy)
% 
% if Add_Ego_Prediction_Switch == 1
%     TJ_X = zeros(length(sim_time),10,7); % ACC DEC ESL ESR ELCL ELCR ESS
%     TJ_Y = zeros(length(sim_time),10,7);
%     
%     if IMAGE_CHANNEL == 24
%         for index_time = 1:length(sim_time)
%             v_e = norm([Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_Y, track_number, index_time) Fusion_Track(FUSION_TRACK.TRACKING.REL_VEL_X, track_number, index_time)],2);
%             ey_ini = 0;
%             WR_ESR = 0.9 - ey_ini;
%             WR_ELCR = 3.5 - ey_ini;
%             t_comp_esr = sqrt((4*pi/3)*WR_ESR - ey_ini);
%             t_comp_elcr = sqrt((4*pi/3)*WR_ELCR - ey_ini);
%             
%             AEB_act = 1;
%             
%             if AEB_act > 0
%                 DEC_param = 0;
%             else
%                 DEC_param = -6;
%             end
%             
%             ACC_param = 4;
%             DEC2_param = -10;
%             for i = 1:EGO_PRED_WINDOW/EGO_PRED_SAMPLE_RATE
%                 t = i*EGO_PRED_SAMPLE_RATE;
%                 %ESL
%                 TJ_Y(index_time,i,3) = (3/2)*(t_comp_esr/(2*pi))^2*...
%                     sin((2*pi/t_comp_esr)*(t))...
%                     - (3*t_comp_esr/(4*pi))*(t) + ey_ini;
%                 TJ_X(index_time,i,3) = v_e*t + 0.5*DEC_param*t.^2;
%                 %ELCL
%                 TJ_Y(index_time,i,5)=(3/2)*(t_comp_elcr/(2*pi))^2*...
%                     sin((2*pi/t_comp_elcr)*(t))...
%                     - (3*t_comp_elcr/(4*pi))*(t) + ey_ini;
%                 TJ_X(index_time,i,5) = v_e*t + 0.5*DEC_param*t.^2;
%                 
%                 TJ_X(index_time,i,1) = v_e*t + 0.5*ACC_param*t.^2;
%                 TJ_X(index_time,i,2) = v_e*t + 0.5*DEC2_param*t.^2;
%             end
%             
%         end
%         TJ_Y(:,:,1) = 0; % ACC Y
%         TJ_Y(:,:,2) = 0; % DEC Y
%         
%         TJ_X(:,:,4) = TJ_X(:,:,3); % ESL X
%         TJ_Y(:,:,4) = -TJ_Y(:,:,3); % ESL Y
%         
%         TJ_X(:,:,6) = TJ_X(:,:,5); % ELCL X
%         TJ_Y(:,:,6) = -TJ_Y(:,:,5); % ELCL Y
%         
%         TJ_X(:,:,7) = TJ_X(:,:,5); % ESS X
%         TJ_Y(:,:,7) = TJ_Y(:,:,5); % ESS Y
%         
%         TJ_Y(:,:,1) = 0; % ACC Y
%         TJ_Y(:,:,2) = 0; % DEC Y
%     else
%         error('Parameter IMAGE_CHANNEL is not 24')
%     end
% else
%     TJ_X = zeros(length(sim_time),10,7); % ACC DEC ESL ESR ELCL ELCR ESS
%     TJ_Y = zeros(length(sim_time),10,7);
% end

