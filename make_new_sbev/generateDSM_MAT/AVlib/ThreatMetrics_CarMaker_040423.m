%% Threat Metric

I_lat_out                                       = zeros(length(sim_time),1);
I_long_out                                      = zeros(length(sim_time),1);
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

for index_time = 1:length(sim_time)
    if index_time == 4151
        a = 1;
    end
    
    if index_time == 2506
        a = 1;
    end
    
    if index_time == 2491
        a = 1;
    end
    
    if index_time == 916
        a = 1;
    end

    if index_time == 951
        a = 1;
    end
    
    for track_number = 1:Traffic_Number
        
        if Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time)^2 + Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time)^2 ~= 0
            % Add Threat Metric to Training Data
            target_vel_x = Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_X, track_number, index_time) + In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time);
            target_vel_y = Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_Y, track_number, index_time) + In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LAT_VEL,1,index_time);
            
            %% TTC
            [tmp_TTC] = TTC(Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_X, track_number, index_time), TTC_PARAM);
            TTC_out(index_time, 1) = tmp_TTC;
            TTC_inverse_out(index_time,1) = 1/tmp_TTC;
            if TTC_inverse_out(index_time)>TTC_INVERSE_PARAM.TTC_INVERSE_MAX
                TTC_inverse_out(index_time)=TTC_INVERSE_PARAM.TTC_INVERSE_MAX;
            end
            
%             Fusion_Track(FUSION_TRACK.THREAT.TTC, track_number, index_time)                        = TTC_out(index_time, 1);
            
            %% TLC
            [tmp_TLC, tmp_DLC] = TLC(Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_Y, track_number, index_time), ...
                Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), ...
                Distance_to_Leftlane(index_time), Distance_to_Rightlane(index_time), TLC_PARAM);
            
            TLC_out(index_time, 1) = tmp_TLC;
            DLC_out(index_time) = tmp_DLC;
            TLC_inverse_out(index_time) = 1/tmp_TLC;
            if TLC_inverse_out(index_time)>TLC_INVERSE_PARAM.TLC_INVERSE_MAX
                TLC_inverse_out(index_time)=TLC_INVERSE_PARAM.TLC_INVERSE_MAX;
            end
            
            Fusion_Track(FUSION_TRACK.THREAT.TLC, track_number, index_time)                        = TLC_out(index_time, 1);
            
            %% Ilat (lateral collision index)
            % lateral: Ilat(combined and single),DLC and TLC
            % longitudinal : Ilong,dw,dbr,xp and TTC
            
            [tmp_I_lat, tmp_I_long, tmp_TTC, tmp_x_p, tmp_d_br, tmp_d_w, tmp_DLC, tmp_TLC] = I_lat(Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time),...
                Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_X, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_Y, track_number, index_time), ...
                In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time), target_vel_x, Fusion_Track(FUSION_TRACK.MEASURE.HEADING_ANGLE, track_number, index_time),...
                Fusion_Track(FUSION_TRACK.MEASURE.WIDTH, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.LENGTH, track_number, index_time), EGO_VEHICLE.EGO_LENGTH, ...
                Distance_to_Leftlane(index_time), Distance_to_Rightlane(index_time), I_LAT_PARAM);
            
            I_lat_out(index_time) = tmp_I_lat;
            I_long_out(index_time) = tmp_I_long;
            Fusion_Track(FUSION_TRACK.THREAT.TTC, track_number, index_time)                        = tmp_TTC;
            Fusion_Track(FUSION_TRACK.THREAT.WARNING_INDEX, track_number, index_time)                = tmp_x_p;
            Fusion_Track(FUSION_TRACK.THREAT.I_LAT, track_number, index_time)                        = I_lat_out(index_time, 1);
            Fusion_Track(FUSION_TRACK.THREAT.I_LONG, track_number, index_time)                       = I_long_out(index_time, 1);
            
            %% RSS (minimum safe distance x and y)
            run("Threat_Parameter")
            [tmp_rss_x,tmp_rss_y]= RSS_model(Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time), In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time),...
                target_vel_x, In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LAT_VEL,1,index_time), target_vel_y, RSS_Param);
            
            RSS_x(index_time) = tmp_rss_x;
            RSS_y(index_time) = tmp_rss_y;
            
            Fusion_Track(FUSION_TRACK.THREAT.RSS_X, track_number, index_time)                        = RSS_x(index_time, 1);
            Fusion_Track(FUSION_TRACK.THREAT.RSS_Y, track_number, index_time)                        = RSS_y(index_time, 1);
            
            %% Honda warning and avoidance algorithm (dw,dbr)
            
            [tmp_HONDA_w, tmp_HONDA_br] = HONDA(Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time), Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_Y, track_number, index_time),...
                Fusion_Track(FUSION_TRACK.MEASURE.REL_VEL_X, track_number, index_time), In_Vehicle_Sensor(IN_VEHICLE_SENSOR.MEASURE.LONG_VEL,1,index_time),target_vel_x, HONDA_PARAM);
            HONDA_w(index_time) = tmp_HONDA_w;
            HONDA_br(index_time) = tmp_HONDA_br;
            
            % THM(HONDA)
            THM(index_time)=(Fusion_Track(FUSION_TRACK.MEASURE.REL_POS_X, track_number, index_time) - HONDA_br(index_time))/(HONDA_w(index_time)-HONDA_br(index_time));
            
            Fusion_Track(FUSION_TRACK.THREAT.FCW_HONDA, track_number, index_time)                        = HONDA_w(index_time, 1);
        end
    end
end