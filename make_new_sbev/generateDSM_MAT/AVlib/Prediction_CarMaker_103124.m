%% Surrounding target prediction



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


X_est = zeros(STATE_NUMBER, Traffic_Number, length(sim_time)); % [y vy ay x vx ax] + [width, length, heading angle, classification, motion]
P_est = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, Traffic_Number, length(sim_time));

X_pred = zeros(STATE_NUMBER, Traffic_Number, length(sim_time)); % [y vy ay x vx ax] + [width, length, heading angle, classification, motion]
P_pred = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, Traffic_Number, length(sim_time));

z_CA = zeros(TRACKING_STATE_NUMBER, Traffic_Number, length(sim_time));
X_updated = zeros(STATE_NUMBER, Traffic_Number, length(sim_time)); % [y vy ay x vx ax] + [width, length, heading angle, classification, motion]
P_updated = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, Traffic_Number, length(sim_time));

X_pred_window = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, Traffic_Number); % [y vy ay x vx ax] + [width, length, heading angle, classification, motion]
P_pred_window = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/SAMPLE_TIME, Traffic_Number);
Association_Map_Total = zeros(Traffic_Number, length(sim_time));

X_pred_window_SBEV = zeros(STATE_NUMBER, length(sim_time), TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, Traffic_Number); % state(y x vy vx ay ax) + classification + motion

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


tmp_residual_total = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(TRACKING.GATING.INPUT_NUMBER, Traffic_Number, Traffic_Number, length(sim_time));

Association_Map_Updated = zeros(Traffic_Number, length(sim_time));

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
collision_probability_total = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, Traffic_Number, length(sim_time)); % prediction window, track_number, length(sim_time)
collision_probability_final = zeros(length(sim_time), Traffic_Number);

for index_time = 1:Test_end_index

    tmp_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(TRACKING.GATING.INPUT_NUMBER, Traffic_Number, Traffic_Number);
    tmp_norm_residual = TRACKING.RESIDUAL.DEFAULT_VALUE * ones(Traffic_Number, Traffic_Number);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Tracking for error covariance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Prediction
    if index_time == 1
        Association_Map_k_1 = zeros(Traffic_Number, 1);
        Fusion_Track_k_1 = zeros(STATE_NUMBER, Traffic_Number);
        P_Fusion_Track_k_1 = zeros(TRACKING_STATE_NUMBER, TRACKING_STATE_NUMBER, Traffic_Number);
    else
        Association_Map_k_1 = Association_Map_Total(:, index_time - 1);
        Fusion_Track_k_1 = X_est(:, :, index_time - 1);
        P_Fusion_Track_k_1 = P_est(:,:,:, index_time - 1);
    end

    for track_number = 1:Traffic_Number
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
    for track_number_k_1 = 1:Traffic_Number
        if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0
            for track_number = 1:Traffic_Number
                if norm([Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time)], 2) ~= 0

                    if Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time) == Fusion_Track_k_1(TRACKING.SHAPE, track_number_k_1)

                        tmp_residual(1, track_number, track_number_k_1) = X_pred(TRACKING.REL_POS_Y, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time); % y
                        tmp_residual(2, track_number, track_number_k_1) = X_pred(TRACKING.REL_POS_X, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time); % x
                        tmp_residual(3, track_number, track_number_k_1) = X_pred(TRACKING.REL_VEL_Y, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_Y, track_number, index_time); % vy
                        tmp_residual(4, track_number, track_number_k_1) = X_pred(TRACKING.REL_VEL_X, track_number_k_1, index_time) - Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_X, track_number, index_time); % vx

                        tmp_residual_total(:, track_number, track_number_k_1, index_time) = tmp_residual(:, track_number, track_number_k_1);

                        tmp_norm_residual(track_number_k_1, track_number) = norm(tmp_residual(:, track_number, track_number_k_1),2);
                    end
                end
            end
        end
    end

    for track_number_k_1 = 1:Traffic_Number
        if sum(Association_Map_k_1(track_number_k_1, 1)) ~= 0

            [~, sorted_track_number] = sort(tmp_norm_residual(track_number_k_1,:));
            [~, sorted_track_number_k_1] = sort(tmp_norm_residual(:, sorted_track_number(1)));

            if sorted_track_number_k_1(1) == track_number_k_1 && ...
                    tmp_residual(1, sorted_track_number(1), track_number_k_1) > GATING.Y_MIN && tmp_residual(1, sorted_track_number(1), track_number_k_1) < GATING.Y_MAX && ...
                    tmp_residual(2, sorted_track_number(1), track_number_k_1) > GATING.X_MIN && tmp_residual(2, sorted_track_number(1), track_number_k_1) < GATING.X_MAX

                z_CA(:,track_number_k_1,index_time) = [Fusion_Track([FUSION_TRACK.MEASURE.REL_POS_Y, FUSION_TRACK.MEASURE.REL_VEL_Y], sorted_track_number(1), index_time); 0;
                    Fusion_Track([FUSION_TRACK.MEASURE.REL_POS_X, FUSION_TRACK.MEASURE.REL_VEL_X, FUSION_TRACK.MEASURE.REL_ACC_X], sorted_track_number(1), index_time)]...
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
    for track_number = 1:Traffic_Number
        if norm([Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time)], 2) ~= 0

            for updated_track_number = 1:Traffic_Number
                if Association_Map_Total(updated_track_number, index_time) ~= 0
                    if Association_Map_Total(updated_track_number, index_time) == track_number
                        if X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.PEDESTRIAN_CANDIDATE || ...
                                X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.PEDESTRIAN_CONFIRMED || ...
                                X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.BICYCLE_CANDIDATE || ...
                                X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.BICYCLE_CONFIRMED || ...
                                X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.E_SCOOTER_CANDIDATE || ...
                                X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.E_SCOOTER_CONFIRMED || ...
                                X_updated(TRACKING.SHAPE, updated_track_number, index_time) == SHAPE.VEHICLE_CONFIRMED

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
    for track_number = 1:Traffic_Number
        if Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time)^2 ~= 0
            % SBEV ROI

            target_y_vertex = zeros(1,5);
            target_x_vertex = zeros(1,5);

            tmp_target_y_vertex = [-Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2,...
                Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, ...
                -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2];

            tmp_target_x_vertex = [0, Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), 0, 0];

            target_y_vertex_rot = tmp_target_x_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) + tmp_target_y_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
            target_x_vertex_rot = tmp_target_x_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - tmp_target_y_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

            target_y_vertex = target_y_vertex_rot + Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time);
            target_x_vertex = target_x_vertex_rot + Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time);

            if ( (min(target_y_vertex) >= Y_MIN && min(target_y_vertex) <= Y_MAX) || (max(target_y_vertex) >= Y_MIN && max(target_y_vertex) <= Y_MAX) ) ...
                    && ( (min(target_x_vertex) >= X_MIN && min(target_x_vertex) <= X_MAX) || (max(target_x_vertex) >= X_MIN && max(target_x_vertex) <= X_MAX) )

                for updated_track_number = 1:Traffic_Number
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

                        % [y vy ay x vx ax]
                        X_est(TRACKING.REL_POS_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time);
                        X_est(TRACKING.REL_VEL_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_Y, track_number, index_time);
                        X_est(TRACKING.REL_ACC_Y, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.REL_ACC_Y, track_number, index_time);
                        X_est(TRACKING.REL_POS_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time);
                        X_est(TRACKING.REL_VEL_X, track_number, index_time) = Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_X, track_number, index_time);
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
    for i_X_est = 1:Traffic_Number
        if sum(Association_Map_Total(i_X_est, index_time)) ~= 0
            for track_number = 1:Traffic_Number
                if Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time)^2 ~= 0
                    % SBEV ROI
                    target_y_vertex = zeros(1,5);
                    target_x_vertex = zeros(1,5);

                    tmp_target_y_vertex = [-Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2,...
                        Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, ...
                        -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2];

                    tmp_target_x_vertex = [0, Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), 0, 0];

                    target_y_vertex_rot = tmp_target_x_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) + tmp_target_y_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                    target_x_vertex_rot = tmp_target_x_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - tmp_target_y_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                    target_y_vertex = target_y_vertex_rot + Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time);
                    target_x_vertex = target_x_vertex_rot + Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time);

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
    
    for track_number = 1:Traffic_Number

        collision_probability_max = 0;

        if Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time)^2 ~= 0
            % ROI
            target_y_vertex = zeros(1,5);
            target_x_vertex = zeros(1,5);

            tmp_target_y_vertex = [-Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2,...
                Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2, ...
                -Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2];

            tmp_target_x_vertex = [0, Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), 0, 0];

            target_y_vertex_rot = tmp_target_x_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) + tmp_target_y_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
            target_x_vertex_rot = tmp_target_x_vertex .* cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - tmp_target_y_vertex .* sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

            target_y_vertex = target_y_vertex_rot + Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time);
            target_x_vertex = target_x_vertex_rot + Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time);

            if ( (min(target_y_vertex) >= Y_MIN && min(target_y_vertex) <= Y_MAX) || (max(target_y_vertex) >= Y_MIN && max(target_y_vertex) <= Y_MAX) ) ...
                    && ( (min(target_x_vertex) >= X_MIN && min(target_x_vertex) <= X_MAX) || (max(target_x_vertex) >= X_MIN && max(target_x_vertex) <= X_MAX) ) ...
                    && sum(P_est(:, :, track_number, index_time), 'all') ~= 0

                Prediction_On(index_time, 1) = 1;

                for index_pred = 1:TARGET_PRED_WINDOW/SAMPLE_TIME
                    if index_pred == 60
                        a = 1;
                    end

                    if index_pred == 1
                        X_pred_window(1:6, index_time, index_pred, track_number) = A_CA * Fusion_Track([FUSION_TRACK.MEASURE.REL_POS_Y, FUSION_TRACK.MEASURE.REL_VEL_Y, FUSION_TRACK.MEASURE.REL_ACC_Y, ...
                            FUSION_TRACK.MEASURE.REL_POS_X, FUSION_TRACK.MEASURE.REL_VEL_X, FUSION_TRACK.MEASURE.REL_ACC_X], track_number, index_time);

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

                            tmp_y_f = EGO_VEHICLE.EGO_WIDTH/2 +...
                                Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) +....
                                Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                            tmp_y_i = -EGO_VEHICLE.EGO_WIDTH/2 -...
                                Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) * sign(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));

                            tmp_cdf_y_f = normcdf(tmp_y_f, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);
                            tmp_cdf_y_i = normcdf(tmp_y_i, X_pred_window(TRACKING.REL_POS_Y, index_time, index_pred, track_number) + X_pred_window(TRACKING.LENGTH, index_time, index_pred, track_number)/2*sin(X_pred_window(TRACKING.HEADING_ANGLE, index_time, index_pred, track_number)), tmp_sigma_y);

                            tmp_x_f = Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time)/2*cos(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time)) - ...
                                Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time)/2*sin(Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time));
                            tmp_x_i = -EGO_VEHICLE.EGO_LENGTH -...
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

                            X_pred_window_SBEV(:, index_time, index_pred/(TARGET_PRED_SAMPLE_RATE*10/SAMPLE_TIME*100)*1000, track_number) = X_pred_window(:, index_time, index_pred, track_number);
                        end
                    else
                        if index_pred == 1 %TARGET_PRED_WINDOW/SAMPLE_TIME
                            sample_time_total_for_collision_probability = zeros(TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE, 1);
                            for tmp_index = 1:TARGET_PRED_WINDOW/TARGET_PRED_SAMPLE_RATE
                                sample_time_total_for_collision_probability(tmp_index) = round(tmp_index*TARGET_PRED_SAMPLE_RATE*10/(SAMPLE_TIME *100) *10);
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
end



