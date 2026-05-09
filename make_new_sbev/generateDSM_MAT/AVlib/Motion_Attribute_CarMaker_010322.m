ROAD_CURVATURE_THRESHOLD     = ROAD.CURVATURE_THRESHOLD;
EGO_VEHICLE_MOTION_THRESHOLD = MOTION.EGO_VEHICLE_VELOCITY_THRESHOLD;
VELOCITY_THRESHOLD_FAST      = MOTION.VELOCITY_THRESHOLD_FAST_VEHICLE;
VELOCITY_THRESHOLD_SLOW      = MOTION.VELOCITY_THRESHOLD_SLOW_VEHICLE;

STATIONARY = MOTION.STATIONARY;
DYNAMIC = MOTION.DYNAMIC;

for index_time = 1:length(sim_time)

    Road_Curvature = Road(ROAD.MEASURE.CURVATURE,1,index_time);
    Vehicle_Speed = In_Vehicle_Sensor(IN_VEHICLE_SENSOR.PREPROCESSING.VEHICLE_SPEED,1,index_time);
    Yaw_Rate = In_Vehicle_Sensor(IN_VEHICLE_SENSOR.PREPROCESSING.VEHICLE_SPEED,1,index_time);

    if Vehicle_Speed > EGO_VEHICLE_MOTION_THRESHOLD
        VELOCITY_THRESHOLD = VELOCITY_THRESHOLD_FAST;
    else
        VELOCITY_THRESHOLD = VELOCITY_THRESHOLD_SLOW;
    end

    for track_number = 1:Traffic_Number
        Relative_POS_Y = 0;
        Relative_VEL_X = 0;
        
        if norm([Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time)],2) ~= 0
        
            Relative_POS_Y = Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time);
            Relative_VEL_X = Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_X, track_number, index_time);

            if Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time) == SHAPE.VEHICLE_CANDIDATE || Fusion_Track(FUSION_TRACK.SHAPE_ATTRIBUTE.SHAPE, track_number, index_time) == SHAPE.VEHICLE_CONFIRMED

                if abs(Road_Curvature) < ROAD_CURVATURE_THRESHOLD                           % straight road
                    Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) = abs(Relative_VEL_X + Vehicle_Speed); % absolut speed(m/s)
                    
                    if Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) < VELOCITY_THRESHOLD
                        Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time) = STATIONARY;           % Front Radar motion = stationary
                        
                    elseif Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) > VELOCITY_THRESHOLD
                        Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time) = DYNAMIC;              % Front Radar motion = dynamic
                    end
                else
                    if abs(Road_Curvature) >= ROAD_CURVATURE_THRESHOLD                      % Curved road
                        Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) = abs( (Relative_VEL_X + (Relative_POS_Y*Yaw_Rate)) + Vehicle_Speed );	% absolut speed(m/s)
                        if Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) < VELOCITY_THRESHOLD
                            Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time) = STATIONARY;           % Front Radar motion = stationary
                            
                        elseif Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) > VELOCITY_THRESHOLD
                            Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time) = DYNAMIC;              % Front Radar motion = dynamic
                        end
                    end
                end

            else
                if abs(Road_Curvature) < ROAD_CURVATURE_THRESHOLD                           % straight road
                    Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) = abs(Relative_VEL_X + Vehicle_Speed); % absolut speed(m/s)
                    
                    if Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) < MOTION.VELOCITY_THRESHOLD_VRU
                        Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time) = STATIONARY;           % Front Radar motion = stationary
                        
                    elseif Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) > MOTION.VELOCITY_THRESHOLD_VRU
                        Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time) = DYNAMIC;              % Front Radar motion = dynamic
                    end
                else
                    if abs(Road_Curvature) >= ROAD_CURVATURE_THRESHOLD                      % Curved road
                        Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) = abs( (Relative_VEL_X + (Relative_POS_Y*Yaw_Rate)) + Vehicle_Speed );	% absolut speed(m/s)
                        if Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) < MOTION.VELOCITY_THRESHOLD_VRU
                            Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time) = STATIONARY;           % Front Radar motion = stationary
                            
                        elseif Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.ABS_VEL, track_number, index_time) > MOTION.VELOCITY_THRESHOLD_VRU
                            Fusion_Track(FUSION_TRACK.MOTION_ATTRIBUTE.MOTION, track_number, index_time) = DYNAMIC;              % Front Radar motion = dynamic
                        end
                    end
                end
            end
        end
    end
end