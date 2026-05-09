function [DSM_out] = DSMOccupancy(DSM_in, State_trajectory, DSM_PARAM, TRAJ_PARAM)
% DSMOccupancy function generates DSM with occupancy channel
%
% [DSM_out] = DSMOccupancy(DSM_in, State_trajectory, DSM_PARAM, TRAJ_PARAM)
%
% DSM_out {double}            : generated DSM
%
% DSM_in {double}             : initialized DSM
% State_trajectory {double}   : trajectory
% DSM_PARAM {struct}          : parameters for generation of DSM
% TRAJ_PARAM {struct}         : parameters for state of trajectory

if DSM_PARAM.BACKGROUND_COLOR_BLACK == 1
    empty_DSM = zeros(DSM_PARAM.IMAGE_HEIGHT, DSM_PARAM.IMAGE_WIDTH, DSM_PARAM.IMAGE_CHANNEL);
elseif DSM_PARAM.BACKGROUND_COLOR_WHITE == 1
    empty_DSM = DSM_PARAM.RGB_MAX*ones(DSM_PARAM.IMAGE_HEIGHT, DSM_PARAM.IMAGE_WIDTH, DSM_PARAM.IMAGE_CHANNEL);
end

Target_Shape_Exist_Flag = 0;
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
% Trajectory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Target_Shape_Exist_Flag == 1
    for index_traj = 1:length(State_trajectory(1,:))
        if norm([State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj), State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj)],2) ~= 0 ...
                && State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj) >= DSM_PARAM.RANGE.X_MIN && State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj) <= DSM_PARAM.RANGE.X_MAX ...
                && State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj) >= DSM_PARAM.RANGE.Y_MIN && State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj) <= DSM_PARAM.RANGE.Y_MAX

            [~,Image_Position_X] = min(abs(State_trajectory(TRAJ_PARAM.REL_POS_X, index_traj) - DSM_PARAM.RANGE.X_RANGE));
            [~,Image_Position_Y] = min(abs(State_trajectory(TRAJ_PARAM.REL_POS_Y, index_traj) - DSM_PARAM.RANGE.Y_RANGE));

            if DSM_PARAM.BACKGROUND_COLOR_BLACK == 1
                for i_ch = 1:CH_LENGTH
                    if DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                        DSM_out(Image_Position_X,Image_Position_Y,DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = DSM_PARAM.RGB_MAX;
                    end
                end
            elseif DSM_PARAM.BACKGROUND_COLOR_WHITE == 1
                for i_ch = 1:CH_LENGTH
                    if DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_Y || DSM_PARAM.CHANNEL_INFO(i_ch).TRAJ_STATE(1) == TRAJ_PARAM.REL_POS_X % position
                        DSM_out(Image_Position_X,Image_Position_Y,DSM_PARAM.CHANNEL_INFO(i_ch).CHANNEL_NUMBER) = DSM_PARAM.RGB_MIN;
                    end
                end
            end
        end
    end
end


if Target_Exist_in_Input_SBEV == 0 && Target_Shape_Exist_Flag == 0
    DSM_out = empty_DSM; % delete SBEV(with only lane mark, without target)
end


end