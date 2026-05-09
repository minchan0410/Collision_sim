function [tmp_index_FB] = doubleThreshold(old_index_FB,tmp_predict_probability_index,Decision_index_count, Decision_index_probability,...
                                          NUM_FB, BUFFER_LENGTH, INDEX_FB_MODE,...
                                          THRESHOLD_FB_PROBABILITY)

tmp_index_FB = old_index_FB;
if NUM_FB == 7 % cannot decide CM52, 53
   
    INDEX_FB_MODE_NONE = INDEX_FB_MODE.INDEX_FB_MODE_NONE;
    INDEX_FB_MODE_ESL = INDEX_FB_MODE.INDEX_FB_MODE_ESL;
    INDEX_FB_MODE_ESR = INDEX_FB_MODE.INDEX_FB_MODE_ESR;
    INDEX_FB_MODE_ESS = INDEX_FB_MODE.INDEX_FB_MODE_ESS;
    INDEX_FB_MODE_ELCL = INDEX_FB_MODE.INDEX_FB_MODE_ELCL;
    INDEX_FB_MODE_ELCR = INDEX_FB_MODE.INDEX_FB_MODE_ELCR;
    INDEX_FB_MODE_CM = INDEX_FB_MODE.INDEX_FB_MODE_CM;
    INDEX_FB_MODE_NOT_CRASH = INDEX_FB_MODE.INDEX_FB_MODE_NOT_CRASH;
    
    for i_buff = 1:BUFFER_LENGTH
        if tmp_predict_probability_index(i_buff) == 12 || tmp_predict_probability_index(i_buff) == INDEX_FB_MODE_NONE...
           || tmp_predict_probability_index(i_buff) == 13
            Decision_index_count(INDEX_FB_MODE_NOT_CRASH) = Decision_index_count(INDEX_FB_MODE_NOT_CRASH) + 1;
            
        elseif tmp_predict_probability_index(i_buff) == 7 ||tmp_predict_probability_index(i_buff) == 8
            Decision_index_count(INDEX_FB_MODE_ESL) = Decision_index_count(INDEX_FB_MODE_ESL) + 1;
            
        elseif tmp_predict_probability_index(i_buff) == 9 ||tmp_predict_probability_index(i_buff) == 10 
            Decision_index_count(INDEX_FB_MODE_ESR) = Decision_index_count(INDEX_FB_MODE_ESR) + 1;
            
        elseif tmp_predict_probability_index(i_buff) == 11 
            Decision_index_count(INDEX_FB_MODE_ESS) = Decision_index_count(INDEX_FB_MODE_ESS) + 1;
            
        elseif tmp_predict_probability_index(i_buff) == 3 || tmp_predict_probability_index(i_buff) == 4
            Decision_index_count(INDEX_FB_MODE_ELCL) = Decision_index_count(INDEX_FB_MODE_ELCL) + 1;
            
        elseif tmp_predict_probability_index(i_buff) == 5 || tmp_predict_probability_index(i_buff) == 6
            Decision_index_count(INDEX_FB_MODE_ELCR) = Decision_index_count(INDEX_FB_MODE_ELCR) + 1;
            
        elseif tmp_predict_probability_index(i_buff) == 1 || tmp_predict_probability_index(i_buff) == 2 % CM
            Decision_index_count(INDEX_FB_MODE_CM) = Decision_index_count(INDEX_FB_MODE_CM) + 1;           

        end
    end
    
end

for i_Dec = 1:NUM_FB
    Decision_index_probability(i_Dec) = Decision_index_count(i_Dec)/BUFFER_LENGTH;
end

for i_Dec = 1:NUM_FB
    if Decision_index_probability(i_Dec) >= THRESHOLD_FB_PROBABILITY
        tmp_index_FB = i_Dec;
        break
    end
end

