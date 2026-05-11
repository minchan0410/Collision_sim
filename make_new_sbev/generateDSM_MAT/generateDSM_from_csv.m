function summary = generateDSM_from_csv(csvRoot, outRoot, varargin)
% Generate RG3-style DSM/SBEV PNGs from diffusion trajectory CSV files.
%
% The original RG3 MATLAB pipeline computes collision probability from
% sensor-fusion prediction covariance. CSV outputs do not contain that
% covariance, so this adapter keeps the original normcdf collision-overlap
% structure and replaces P_pred_window with empirical covariance estimated
% from diffusion prediction samples.

if nargin < 1 || isempty(csvRoot)
    csvRoot = fullfile('results_pkl', 'try_1', 'csv');
end
if nargin < 2 || isempty(outRoot)
    outRoot = fullfile('results_pkl', 'try_1', 'dsm_sbev');
end

opts = parseOptions(varargin{:});

thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'AVlib', 'dECISION', 'simplifiedBEV'));

csvFiles = resolveCsvFiles(csvRoot);
if isfinite(opts.MaxFiles)
    csvFiles = csvFiles(1:min(numel(csvFiles), opts.MaxFiles));
end
if isempty(csvFiles)
    error('generateDSM_from_csv:NoCsvFiles', 'No CSV files found: %s', char(csvRoot));
end

outRoot = char(outRoot);
if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end

[SBEV_PARAM, TRAJ, TRACKING, FRONT_VISION_LANE, EGO_VEHICLE, emptySbev] = buildRasterParams(opts);
summaryVariableNames = {'csv_path', 'sample', 'collision_mode', 'frame', 'collision_probability', 'nonwhite_pixels', 'png_name', 'png_path'};
summaryRows = cell(0, numel(summaryVariableNames));
savedCount = 0;

for iFile = 1:numel(csvFiles)
    csvPath = csvFiles{iFile};
    data = readtable(csvPath);
    validateCsv(data, csvPath);

    frames = selectFrames(data, opts.Frame, opts.MaxFrames);
    sampleIds = selectSamples(data, opts.Sample);

    if opts.UseCsvSubfolder
        csvOutDir = fullfile(outRoot, stripExtension(filenameOnly(csvPath)));
    else
        csvOutDir = outRoot;
    end
    if ~exist(csvOutDir, 'dir')
        mkdir(csvOutDir);
    end

    if opts.Verbose
        fprintf('[CSV] %s | samples=%d frames=%d\n', csvPath, numel(sampleIds), numel(frames));
    end

    for iSample = 1:numel(sampleIds)
        sampleId = sampleIds(iSample);
        collisionMode = collisionModeForSample(data, sampleId);
        sampleOutDir = fullfile(csvOutDir, sprintf('sample_%03d', sampleId));
        if ~exist(sampleOutDir, 'dir')
            mkdir(sampleOutDir);
        end

        for iFrame = 1:numel(frames)
            frame = frames(iFrame);
            [SBEV, probability, didDraw] = renderFrameFromCsv( ...
                data, sampleId, frame, emptySbev, SBEV_PARAM, TRAJ, TRACKING, ...
                FRONT_VISION_LANE, EGO_VEHICLE, opts);

            if didDraw || opts.SaveEmpty
                saveName = buildSbevPngName(csvPath, collisionMode, frame);
                savePath = fullfile(sampleOutDir, saveName);
                imwrite(uint8(SBEV), savePath);
                savedCount = savedCount + 1;
                nonWhitePixels = nnz(any(uint8(SBEV) ~= 255, 3));
                summaryRows(end + 1, :) = {csvPath, sampleId, collisionMode, frame, probability, nonWhitePixels, saveName, savePath}; %#ok<AGROW>
            end
        end
    end
end

summary = cell2table(summaryRows, 'VariableNames', summaryVariableNames);

summaryPath = fullfile(outRoot, 'summary.csv');
if ~isempty(summaryRows)
    writetable(summary, summaryPath);
end

if opts.Verbose
    fprintf('[Done] csv_files=%d images=%d out=%s\n', numel(csvFiles), savedCount, outRoot);
    if ~isempty(summaryRows)
        fprintf('[Summary] %s\n', summaryPath);
    end
end
end

function opts = parseOptions(varargin)
parser = inputParser;
parser.addParameter('Sample', 'all');
parser.addParameter('Frame', 'all');
parser.addParameter('DataDt', 0.02);
parser.addParameter('HistorySec', 0.9);
parser.addParameter('PredictionSec', 1.0);
parser.addParameter('PredictionSampleRate', 0.2);
parser.addParameter('CarWidth', 1.825);
parser.addParameter('CarLength', 4.650);
parser.addParameter('DrawLanes', false);
parser.addParameter('SigmaXMin', 0.35);
parser.addParameter('SigmaYMin', 0.20);
parser.addParameter('SigmaGrowthPerSec', 0.25);
parser.addParameter('KalmanMeasurementNoise', 0.35);
parser.addParameter('KalmanProcessNoise', 0.20);
parser.addParameter('MaxPredictionAccel', 6.0);
parser.addParameter('MaxFiles', inf);
parser.addParameter('MaxFrames', inf);
parser.addParameter('SaveEmpty', false);
parser.addParameter('Verbose', true);
parser.addParameter('UseCsvSubfolder', true);
parser.parse(varargin{:});
opts = parser.Results;
end

function csvFiles = resolveCsvFiles(csvRoot)
csvRoot = char(csvRoot);
if isfile(csvRoot)
    csvFiles = {csvRoot};
elseif isfolder(csvRoot)
    files = dir(fullfile(csvRoot, '*.csv'));
    [~, order] = sort({files.name});
    files = files(order);
    csvFiles = cell(numel(files), 1);
    for i = 1:numel(files)
        csvFiles{i} = fullfile(files(i).folder, files(i).name);
    end
else
    error('generateDSM_from_csv:MissingPath', 'CSV path does not exist: %s', csvRoot);
end
end

function validateCsv(data, csvPath)
required = {'agent', 'sample', 'frame', 'x', 'y'};
missing = setdiff(required, data.Properties.VariableNames);
if ~isempty(missing)
    error('generateDSM_from_csv:BadCsv', '%s is missing columns: %s', csvPath, strjoin(missing, ', '));
end
if ~any(data.agent == 0 & data.sample == 0)
    error('generateDSM_from_csv:MissingEgo', '%s has no agent=0,sample=0 ego trajectory.', csvPath);
end
if ~any(data.agent == 1)
    error('generateDSM_from_csv:MissingTarget', '%s has no agent=1 target trajectory.', csvPath);
end
end

function frames = selectFrames(data, requestedFrame, maxFrames)
egoFrames = unique(data.frame(data.agent == 0 & data.sample == 0));
targetFrames = unique(data.frame(data.agent == 1));
frames = intersect(egoFrames, targetFrames);
frames = frames(:)';
if ischar(requestedFrame) || isstring(requestedFrame)
    if ~strcmpi(char(requestedFrame), 'all')
        frames = str2double(char(requestedFrame));
    end
else
    frames = requestedFrame;
end
frames = sort(unique(double(frames(:)')));
if isfinite(maxFrames)
    frames = frames(1:min(numel(frames), maxFrames));
end
end

function sampleIds = selectSamples(data, requestedSample)
sampleIds = unique(data.sample(data.agent == 1));
sampleIds = sort(double(sampleIds(:)'));
if ischar(requestedSample) || isstring(requestedSample)
    if ~strcmpi(char(requestedSample), 'all')
        sampleIds = str2double(char(requestedSample));
    end
else
    sampleIds = requestedSample;
end
sampleIds = sort(unique(double(sampleIds(:)')));
end

function collisionMode = collisionModeForSample(data, sampleId)
validModes = [0, 11, 12, 13, 21, 23, 31, 33, 41, 43, 51, 52, 53];
collisionMode = 0;
if ~any(strcmp('cm', data.Properties.VariableNames))
    return;
end

mask = data.agent == 1 & data.sample == sampleId;
values = double(data.cm(mask));
values = values(isfinite(values));
if isempty(values)
    return;
end

values = round(values(:));
positiveValues = values(values > 0);
if ~isempty(positiveValues)
    collisionMode = mode(positiveValues);
else
    collisionMode = mode(values);
end

if ~ismember(collisionMode, validModes)
    collisionMode = 0;
end
end

function saveName = buildSbevPngName(csvPath, collisionMode, frame)
baseName = stripExtension(filenameOnly(csvPath));
baseName = regexprep(baseName, '_generation$', '');
replacement = sprintf('Image_%d_', collisionMode);
if ~isempty(regexp(baseName, '^Image_\d+_', 'once'))
    baseName = regexprep(baseName, '^Image_\d+_', replacement, 'once');
else
    baseName = [replacement baseName];
end
saveName = sprintf('%s_frame_%06d_gen.png', baseName, frame);
end

function [SBEV_PARAM, TRAJ, TRACKING, FRONT_VISION_LANE, EGO_VEHICLE, emptySbev] = buildRasterParams(opts)
SBEV_PARAM.IMAGE_HEIGHT = 201;
SBEV_PARAM.IMAGE_WIDTH = 101;
SBEV_PARAM.IMAGE_CHANNEL = 3;
SBEV_PARAM.RGB_MIN = 0;
SBEV_PARAM.RGB_MAX = 255;
SBEV_PARAM.RGB_IMAGE = 1;
SBEV_PARAM.GRAY_IMAGE = 0;
SBEV_PARAM.BACKGROUND_COLOR_BLACK = 0;
SBEV_PARAM.BACKGROUND_COLOR_WHITE = 1;

SBEV_PARAM.RANGE.X_MIN = -10;
SBEV_PARAM.RANGE.X_MAX = 30;
SBEV_PARAM.RANGE.Y_MIN = -10;
SBEV_PARAM.RANGE.Y_MAX = 10;
SBEV_PARAM.RANGE.X_RANGE = linspace(SBEV_PARAM.RANGE.X_MAX, SBEV_PARAM.RANGE.X_MIN, SBEV_PARAM.IMAGE_HEIGHT);
SBEV_PARAM.RANGE.Y_RANGE = linspace(SBEV_PARAM.RANGE.Y_MAX, SBEV_PARAM.RANGE.Y_MIN, SBEV_PARAM.IMAGE_WIDTH);
SBEV_PARAM.RANGE.I_LAT_MIN = -0.2;
SBEV_PARAM.RANGE.I_LAT_MAX = 1;
SBEV_PARAM.RANGE.I_LAT_RANGE = linspace(-0.2, 1, SBEV_PARAM.RGB_MAX);
SBEV_PARAM.RANGE.COLLISION_PROBABILITY_MIN = -0.2;
SBEV_PARAM.RANGE.COLLISION_PROBABILITY_MAX = 1;
SBEV_PARAM.RANGE.COLLISION_PROBABILITY_RANGE = linspace(-0.2, 1, SBEV_PARAM.RGB_MAX);

SBEV_PARAM.RGB_CYCLIST = 255;
SBEV_PARAM.RGB_CAR = 255;
SBEV_PARAM.RGB_PEDESTRIAN = 255;

SBEV_PARAM.TRAJECTORY.ON = 0;
SBEV_PARAM.TRAJECTORY.FADING.ON = 0;
SBEV_PARAM.TRAJECTORY_POSITION = 0;
SBEV_PARAM.TRAJECTORY_THREAT = 1;
SBEV_PARAM.TRAJECTORY_POSITION_WITH_THREAT = 0;

SBEV_PARAM.LANE_MARK.ON = double(opts.DrawLanes);
SBEV_PARAM.SHAPE.EGO = 1;
SBEV_PARAM.SHAPE.TARGET.POSITION = 0;
SBEV_PARAM.SHAPE.TARGET.THREAT = 1;

SBEV_PARAM.PREDICTION.ON = 1;
SBEV_PARAM.PREDICTION.TARGET = 1;
SBEV_PARAM.PREDICTION.TARGET_PRED_WINDOW = opts.PredictionSec;
SBEV_PARAM.PREDICTION.TARGET_PRED_SAMPLE_RATE = opts.PredictionSampleRate;
SBEV_PARAM.PREDICTION.TARGET_PRED_CHANNEL_NUMBER = 1;
SBEV_PARAM.PREDICTION.FADING.ON = 0;
SBEV_PARAM.PREDICTION.OVERLAP_FLAG = 0;
SBEV_PARAM.PREDICTION.ALL_SHAPE_FLAG = 1;
SBEV_PARAM.PREDICTION.TRAJECTORY_THREAT = 1;

SBEV_PARAM.COLLISION_PROBABILITY.ON = 1;
SBEV_PARAM.CLASS_RGB.ON = 0;
SBEV_PARAM.CLASS_BLACK.ON = 0;
SBEV_PARAM.REPEATED_CHANNEL_INFO_STACKED = 0;

TRAJ.NA = 0;
TRAJ.REL_POS_Y = 1;
TRAJ.REL_POS_X = 2;
TRAJ.REL_VEL_Y = 3;
TRAJ.REL_VEL_X = 4;
TRAJ.WIDTH = 5;
TRAJ.LENGTH = 6;
TRAJ.HEADING_ANGLE = 7;
TRAJ.I_LAT = 8;
TRAJ.I_LONG = 9;
TRAJ.RSS_X = 10;
TRAJ.RSS_Y = 11;
TRAJ.EGO_VEL_Y = 12;
TRAJ.EGO_VEL_X = 13;
TRAJ.ABS_VEL_Y = 14;
TRAJ.ABS_VEL_X = 15;
TRAJ.REL_ACC_Y = 16;
TRAJ.REL_ACC_X = 17;
TRAJ.COLLISION_PROBABILITY = 18;
TRAJ.RGB_CYCLIST = 19;
TRAJ.RGB_CAR = 20;
TRAJ.RGB_PEDESTRIAN = 21;

SBEV_PARAM.CHANNEL.COLLISION_PROBABILITY.CHANNEL_NUMBER = 1;
SBEV_PARAM.CHANNEL.LANE_MARK.CHANNEL_NUMBER = 2;
SBEV_PARAM.CHANNEL.LANE_MARK.TRAJ_STATE = TRAJ.NA;
SBEV_PARAM.CHANNEL_INFO(1).CHANNEL_NUMBER = 1;
SBEV_PARAM.CHANNEL_INFO(1).TRAJ_STATE = TRAJ.COLLISION_PROBABILITY;
SBEV_PARAM.CHANNEL_INFO(2).CHANNEL_NUMBER = 2;
SBEV_PARAM.CHANNEL_INFO(2).TRAJ_STATE = TRAJ.NA;
SBEV_PARAM.CHANNEL_INFO(3).CHANNEL_NUMBER = 3;
SBEV_PARAM.CHANNEL_INFO(3).TRAJ_STATE = TRAJ.NA;

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

FRONT_VISION_LANE.MEASURE.CONFIDENCE = 1;
FRONT_VISION_LANE.MEASURE.VIEWRANGE = 2;
FRONT_VISION_LANE.MEASURE.VIEWRANGE_START = 3;
FRONT_VISION_LANE.PREPROCESSING.DISTANCE = 4;
FRONT_VISION_LANE.PREPROCESSING.ROAD_SLOPE = 5;
FRONT_VISION_LANE.PREPROCESSING.CURVATURE = 6;
FRONT_VISION_LANE.PREPROCESSING.CURVATURE_RATE = 7;
FRONT_VISION_LANE.DESCRIPTION_CONFIDENCE.PREDICTION = 3;

EGO_VEHICLE.WIDTH = opts.CarWidth;
EGO_VEHICLE.LENGTH = opts.CarLength;

emptySbev = 255 * ones(SBEV_PARAM.IMAGE_HEIGHT, SBEV_PARAM.IMAGE_WIDTH, SBEV_PARAM.IMAGE_CHANNEL);
end

function [SBEV, probability, didDraw] = renderFrameFromCsv(data, sampleId, frame, emptySbev, SBEV_PARAM, TRAJ, TRACKING, FRONT_VISION_LANE, EGO_VEHICLE, opts)
SBEV = emptySbev;
laneInfoL = zeros(7, 1);
laneInfoR = zeros(7, 1);
laneFlag = 0;
egoFlag = 0;

targetPath = getPath(data, 1, sampleId);
if isempty(targetPath.frames)
    didDraw = false;
    probability = 0;
    return;
end

EGO_THIS = EGO_VEHICLE;
[egoWidth, egoLength] = vehicleSizeAtFrame(data, 0, 0, frame, opts);
EGO_THIS.WIDTH = egoWidth;
EGO_THIS.LENGTH = egoLength;

probability = computeDiffusionCollisionProbability(data, frame, EGO_THIS, opts);
stateTrajectory = buildStateTrajectory(data, targetPath, frame, probability, TRAJ, EGO_THIS, opts);

if all(stateTrajectory(TRAJ.REL_POS_X, end) == 0) && all(stateTrajectory(TRAJ.REL_POS_Y, end) == 0)
    didDraw = false;
    return;
end

targetPred = buildTargetPrediction(data, targetPath, frame, TRACKING, EGO_THIS, opts);

[SBEV, ~, ~] = DSMMaxCollisionProbabilityRG3_v5( ...
    SBEV, stateTrajectory, laneInfoL, laneInfoR, targetPred, laneFlag, egoFlag, ...
    SBEV_PARAM, TRAJ, FRONT_VISION_LANE, EGO_THIS, TRACKING, opts.DataDt);

didDraw = ~isequal(uint8(SBEV), uint8(emptySbev));
end

function stateTrajectory = buildStateTrajectory(data, targetPath, frame, probability, TRAJ, EGO_VEHICLE, opts)
stateCount = 21;
historySteps = max(1, round(opts.HistorySec / opts.DataDt) + 1);
stateTrajectory = zeros(stateCount, historySteps);
historyFrames = frame - historySteps + 1:frame;

for i = 1:numel(historyFrames)
    hFrame = historyFrames(i);
    [~, targetOk] = pointAtFrame(targetPath, hFrame, false);
    if ~targetOk
        continue;
    end

    [relRear, headingLocal, relVel, ~, okState] = relativeKinematicsInEgoFrame( ...
        data, targetPath, hFrame, EGO_VEHICLE, opts, hFrame);
    if ~okState
        continue;
    end
    stateTrajectory(TRAJ.REL_POS_Y, i) = relRear(2);
    stateTrajectory(TRAJ.REL_POS_X, i) = relRear(1);
    stateTrajectory(TRAJ.REL_VEL_Y, i) = relVel(2);
    stateTrajectory(TRAJ.REL_VEL_X, i) = relVel(1);
    [targetWidth, targetLength] = sizeAtFramePath(targetPath, hFrame, opts);
    stateTrajectory(TRAJ.WIDTH, i) = targetWidth;
    stateTrajectory(TRAJ.LENGTH, i) = targetLength;
    stateTrajectory(TRAJ.HEADING_ANGLE, i) = headingLocal;
    stateTrajectory(TRAJ.I_LAT, i) = probability;
    stateTrajectory(TRAJ.COLLISION_PROBABILITY, i) = probability;
    stateTrajectory(TRAJ.ABS_VEL_Y, i) = relVel(2);
    stateTrajectory(TRAJ.ABS_VEL_X, i) = relVel(1);
    stateTrajectory(TRAJ.RGB_CAR, i) = 4;
end
end

function targetPred = buildTargetPrediction(data, targetPath, frame, TRACKING, EGO_VEHICLE, opts)
stateNumber = 11;
predSteps = max(1, ceil(opts.PredictionSec / opts.DataDt));
targetPred = zeros(stateNumber, 1, predSteps);

[egoState, egoOk] = egoBasisAtFrame(data, frame, EGO_VEHICLE, opts);
if ~egoOk
    return;
end

[relRear, headingLocal, relVel, relAcc] = estimateRelativeCaState( ...
    data, targetPath, frame, EGO_VEHICLE, opts, frame);
relAcc = clipVector(relAcc, opts.MaxPredictionAccel);
state = [relRear(2); relVel(2); relAcc(2); relRear(1); relVel(1); relAcc(1)];
A = caTransition(opts.DataDt);
[targetWidth, targetLength] = sizeAtFramePath(targetPath, frame, opts);

for step = 1:predSteps
    state = A * state;

    targetPred(TRACKING.REL_POS_Y, 1, step) = state(1);
    targetPred(TRACKING.REL_VEL_Y, 1, step) = state(2);
    targetPred(TRACKING.REL_ACC_Y, 1, step) = state(3);
    targetPred(TRACKING.REL_POS_X, 1, step) = state(4);
    targetPred(TRACKING.REL_VEL_X, 1, step) = state(5);
    targetPred(TRACKING.REL_ACC_X, 1, step) = state(6);
    targetPred(TRACKING.WIDTH, 1, step) = targetWidth;
    targetPred(TRACKING.LENGTH, 1, step) = targetLength;
    targetPred(TRACKING.HEADING_ANGLE, 1, step) = headingLocal;
    targetPred(TRACKING.SHAPE, 1, step) = 4;
    targetPred(TRACKING.MOTION, 1, step) = 1;
end
end

function probability = computeDiffusionCollisionProbability(data, frame, EGO_VEHICLE, opts)
sampleIds = unique(data.sample(data.agent == 1 & data.sample > 0));
if isempty(sampleIds)
    sampleIds = unique(data.sample(data.agent == 1));
end
sampleIds = double(sampleIds(:)');
if isempty(sampleIds)
    probability = 0;
    return;
end

predHorizonCount = max(1, floor(opts.PredictionSec / opts.PredictionSampleRate));
probability = 0;

[egoState, egoOk] = egoBasisAtFrame(data, frame, EGO_VEHICLE, opts);
if ~egoOk
    return;
end

meanPath = getPath(data, 1, 0);
if isempty(meanPath.frames)
    meanPath = getPath(data, 1, sampleIds(1));
end
[relRear0, heading0, relVel0, relAcc0] = estimateRelativeCaState( ...
    data, meanPath, frame, EGO_VEHICLE, opts, frame);
relAcc0 = clipVector(relAcc0, opts.MaxPredictionAccel);
if all(relRear0 == 0) && all(relVel0 == 0)
    return;
end
state0 = [relRear0(2); relVel0(2); relAcc0(2); relRear0(1); relVel0(1); relAcc0(1)];

for k = 1:predHorizonCount
    horizonSec = k * opts.PredictionSampleRate;
    predFrame = frame + round(horizonSec / opts.DataDt);

    rearPositions = [];
    headings = [];
    for sampleId = sampleIds
        targetPath = getPath(data, 1, sampleId);
        if isempty(targetPath.frames)
            continue;
        end
        [relRearSample, headingLocal, ~, ~, okSample] = relativeKinematicsInEgoFrame( ...
            data, targetPath, predFrame, EGO_VEHICLE, opts, frame);
        if ~okSample
            continue;
        end
        rearPositions(end + 1, :) = relRearSample; %#ok<AGROW>
        headings(end + 1, 1) = headingLocal; %#ok<AGROW>
    end

    if isempty(rearPositions)
        continue;
    end

    if size(rearPositions, 1) >= 2
        c = cov(rearPositions);
        if any(~isfinite(c(:)))
            c = zeros(2);
        end
    else
        c = zeros(2);
    end

    horizonSteps = max(1, round(horizonSec / opts.DataDt));
    statePred = predictCaState(state0, horizonSteps, opts.DataDt);
    meanRear = [statePred(4), statePred(1)];
    heading = heading0;
    [targetWidth, targetLength] = sizeAtFramePath(meanPath, frame, opts);
    meanCenter = meanRear + 0.5 * targetLength * [cos(heading), sin(heading)];

    sigmaX = max(sqrt(max(c(1, 1), 0)), opts.SigmaXMin + opts.SigmaGrowthPerSec * horizonSec);
    sigmaY = max(sqrt(max(c(2, 2), 0)), opts.SigmaYMin + opts.SigmaGrowthPerSec * horizonSec);
    if ~isempty(headings) && ~isfinite(heading)
        heading = circularMean(headings);
    end

    p = collisionProbabilityFromNormal(meanCenter, heading, sigmaX, sigmaY, EGO_VEHICLE, opts, targetWidth, targetLength);
    probability = max(probability, p);
end

probability = min(max(probability, 0), 1);
end

function p = collisionProbabilityFromNormal(meanCenter, heading, sigmaX, sigmaY, EGO_VEHICLE, opts, targetWidth, targetLength)
tmpYf = EGO_VEHICLE.WIDTH / 2 + ...
    targetLength / 2 * sin(heading) * signOrOne(heading) + ...
    targetWidth / 2 * cos(heading);
tmpYi = -EGO_VEHICLE.WIDTH / 2 - ...
    targetLength / 2 * sin(heading) * signOrOne(heading) - ...
    targetWidth / 2 * cos(heading);

tmpXf = targetLength / 2 * cos(heading) - targetWidth / 2 * sin(heading);
tmpXi = -EGO_VEHICLE.LENGTH - ...
    targetLength / 2 * cos(heading) - ...
    targetWidth / 2 * sin(heading) * signOrOne(heading);

py = normalCdf(tmpYf, meanCenter(2), sigmaY) - normalCdf(tmpYi, meanCenter(2), sigmaY);
px = normalCdf(tmpXf, meanCenter(1), sigmaX) - normalCdf(tmpXi, meanCenter(1), sigmaX);
p = min(max(px * py, 0), 1);
end

function y = normalCdf(x, mu, sigma)
sigma = max(abs(sigma), eps);
y = 0.5 * (1 + erf((x - mu) ./ (sigma * sqrt(2))));
end

function s = signOrOne(x)
s = sign(x);
if s == 0
    s = 1;
end
end

function A = caTransition(dt)
A1 = [1, dt, 0.5 * dt^2; 0, 1, dt; 0, 0, 1];
A = blkdiag(A1, A1);
end

function stateOut = predictCaState(stateIn, steps, dt)
A = caTransition(dt);
stateOut = stateIn;
for i = 1:steps
    stateOut = A * stateOut;
end
end

function [relRear, headingLocal, relVel, relAcc, ok] = relativeKinematicsInEgoFrame(data, targetPath, frame, EGO_VEHICLE, opts, basisFrame)
[egoBasis, basisOk] = egoBasisAtFrame(data, basisFrame, EGO_VEHICLE, opts);
[egoFront, egoFrontVel, egoFrontAcc, egoOk] = egoFrontKinematicsAtFrame(data, frame, EGO_VEHICLE, opts);
[targetRear, targetRearVel, targetRearAcc, headingLocal, targetOk] = targetRearKinematicsAtFrame(targetPath, frame, egoBasis, opts);

ok = basisOk && egoOk && targetOk;
if ~ok
    relRear = [0, 0];
    headingLocal = 0;
    relVel = [0, 0];
    relAcc = [0, 0];
    return;
end

relRearWorld = targetRear - egoFront;
relVelWorld = targetRearVel - egoFrontVel;
relAccWorld = targetRearAcc - egoFrontAcc;

relRear = projectToEgoBasis(relRearWorld, egoBasis);
relVel = projectToEgoBasis(relVelWorld, egoBasis);
relAcc = projectToEgoBasis(relAccWorld, egoBasis);
end

function [relRear, headingLocal, relVel, relAcc] = estimateRelativeCaState(data, targetPath, frame, EGO_VEHICLE, opts, basisFrame)
historySteps = max(3, round(opts.HistorySec / opts.DataDt) + 1);
historyFrames = frame - historySteps + 1:frame;
measurements = [];
headings = [];

for hFrame = historyFrames
    [~, targetOkStrict] = pointAtFrame(targetPath, hFrame, false);
    if ~targetOkStrict
        continue;
    end

    [relRearSample, headingSample, ~, ~, okSample] = relativeKinematicsInEgoFrame( ...
        data, targetPath, hFrame, EGO_VEHICLE, opts, basisFrame);
    if ~okSample
        continue;
    end

    measurements(end + 1, :) = relRearSample; %#ok<AGROW>
    headings(end + 1, 1) = headingSample; %#ok<AGROW>
end

[relRearCurrent, headingCurrent, relVelCurrent, relAccCurrent, okCurrent] = relativeKinematicsInEgoFrame( ...
    data, targetPath, frame, EGO_VEHICLE, opts, basisFrame);

if size(measurements, 1) < 2
    if okCurrent
        relRear = relRearCurrent;
        headingLocal = headingCurrent;
        relVel = relVelCurrent;
        relAcc = relAccCurrent;
    else
        relRear = [0, 0];
        headingLocal = 0;
        relVel = [0, 0];
        relAcc = [0, 0];
    end
    return;
end

dt = opts.DataDt;
A = caTransition(dt);
H = [1, 0, 0, 0, 0, 0; 0, 0, 0, 1, 0, 0];
q = opts.KalmanProcessNoise^2;
Q1 = q * [dt^5 / 20, dt^4 / 8, dt^3 / 6; dt^4 / 8, dt^3 / 3, dt^2 / 2; dt^3 / 6, dt^2 / 2, dt];
Q = blkdiag(Q1, Q1);
R = opts.KalmanMeasurementNoise^2 * eye(2);

initialVel = (measurements(2, :) - measurements(1, :)) / dt;
x = [measurements(1, 2); initialVel(2); 0; measurements(1, 1); initialVel(1); 0];
P = diag([1, 10, 10, 1, 10, 10]);

for i = 1:size(measurements, 1)
    if i > 1
        x = A * x;
        P = A * P * A' + Q;
    end

    z = [measurements(i, 2); measurements(i, 1)];
    innovation = z - H * x;
    S = H * P * H' + R;
    K = P * H' / S;
    x = x + K * innovation;
    P = (eye(6) - K * H) * P;
end

relRear = [x(4), x(1)];
relVel = [x(5), x(2)];
relAcc = [x(6), x(3)];

if okCurrent
    headingLocal = headingCurrent;
elseif ~isempty(headings)
    headingLocal = circularMean(headings);
else
    headingLocal = 0;
end
end

function [front, frontVel, frontAcc, ok] = egoFrontKinematicsAtFrame(data, frame, EGO_VEHICLE, opts)
egoPath = getPath(data, 0, 0);
[front, ok] = egoFrontWorldAtFrame(egoPath, frame, EGO_VEHICLE, opts);
if ~ok
    frontVel = [0, 0];
    frontAcc = [0, 0];
    return;
end

[frontPrev, okPrev] = egoFrontWorldAtFrame(egoPath, frame - 1, EGO_VEHICLE, opts);
[frontNext, okNext] = egoFrontWorldAtFrame(egoPath, frame + 1, EGO_VEHICLE, opts);
[frontVel, frontAcc] = finiteDifferenceKinematics(frontPrev, front, frontNext, okPrev, okNext, opts.DataDt);
end

function [front, ok] = egoFrontWorldAtFrame(egoPath, frame, EGO_VEHICLE, opts)
[egoCenter, ok] = pointAtFrame(egoPath, frame, true);
if ~ok
    front = [0, 0];
    return;
end
forward = directionAtFrame(egoPath, frame, opts);
front = egoCenter + 0.5 * EGO_VEHICLE.LENGTH * forward;
end

function [targetRear, targetRearVel, targetRearAcc, headingLocal, ok] = targetRearKinematicsAtFrame(targetPath, frame, egoBasis, opts)
[targetRear, headingLocal, ok] = targetRearWorldAtFrame(targetPath, frame, egoBasis, opts);
if ~ok
    targetRearVel = [0, 0];
    targetRearAcc = [0, 0];
    return;
end

[targetRearPrev, ~, okPrev] = targetRearWorldAtFrame(targetPath, frame - 1, egoBasis, opts);
[targetRearNext, ~, okNext] = targetRearWorldAtFrame(targetPath, frame + 1, egoBasis, opts);
[targetRearVel, targetRearAcc] = finiteDifferenceKinematics(targetRearPrev, targetRear, targetRearNext, okPrev, okNext, opts.DataDt);
end

function [targetRear, headingLocal, ok] = targetRearWorldAtFrame(targetPath, frame, egoBasis, opts)
[targetRear, ok] = pointAtFrame(targetPath, frame, true);
if ~ok
    targetRear = [0, 0];
    headingLocal = 0;
    return;
end

targetDir = directionAtFrame(targetPath, frame, opts);
targetDirLocal = projectToEgoBasis(targetDir, egoBasis);
headingLocal = atan2(targetDirLocal(2), targetDirLocal(1));
end

function [vel, acc] = finiteDifferenceKinematics(prevPt, currentPt, nextPt, okPrev, okNext, dt)
if okPrev && okNext
    vel = (nextPt - prevPt) / (2 * dt);
    acc = (nextPt - 2 * currentPt + prevPt) / (dt^2);
elseif okPrev
    vel = (currentPt - prevPt) / dt;
    acc = [0, 0];
elseif okNext
    vel = (nextPt - currentPt) / dt;
    acc = [0, 0];
else
    vel = [0, 0];
    acc = [0, 0];
end
end

function localVec = projectToEgoBasis(worldVec, egoBasis)
localVec = [dot(worldVec, egoBasis.forward), dot(worldVec, egoBasis.left)];
end

function [relRear, headingLocal, relVel] = targetStateInEgoFrame(targetPath, frame, egoState, EGO_VEHICLE, opts)
[centerLocal, headingLocal] = targetCenterInEgoFrame(targetPath, frame, egoState, opts);
forwardLocal = [cos(headingLocal), sin(headingLocal)];
[~, targetLength] = sizeAtFramePath(targetPath, frame, opts);
relRear = centerLocal - 0.5 * targetLength * forwardLocal;

prevFrame = frame - 1;
nextFrame = frame + 1;
[centerPrev, ~, okPrev] = targetCenterInEgoFrame(targetPath, prevFrame, egoState, opts);
[centerNext, ~, okNext] = targetCenterInEgoFrame(targetPath, nextFrame, egoState, opts);
if okPrev && okNext
    relVel = (centerNext - centerPrev) / (2 * opts.DataDt);
else
    relVel = [0, 0];
end
end

function [relRear, headingLocal, relVel, relAcc] = targetKinematicsInEgoFrame(targetPath, frame, egoState, opts)
[centerLocal, headingLocal, okCenter] = targetCenterInEgoFrame(targetPath, frame, egoState, opts);
if ~okCenter
    relRear = [0, 0];
    headingLocal = 0;
    relVel = [0, 0];
    relAcc = [0, 0];
    return;
end

prevFrame = frame - 1;
nextFrame = frame + 1;
[centerPrev, ~, okPrev] = targetCenterInEgoFrame(targetPath, prevFrame, egoState, opts);
[centerNext, ~, okNext] = targetCenterInEgoFrame(targetPath, nextFrame, egoState, opts);

if okPrev && okNext
    relVel = (centerNext - centerPrev) / (2 * opts.DataDt);
    relAcc = (centerNext - 2 * centerLocal + centerPrev) / (opts.DataDt^2);
elseif okPrev
    relVel = (centerLocal - centerPrev) / opts.DataDt;
    relAcc = [0, 0];
elseif okNext
    relVel = (centerNext - centerLocal) / opts.DataDt;
    relAcc = [0, 0];
else
    relVel = [0, 0];
    relAcc = [0, 0];
end

if norm(relVel) > 1.0e-6
    headingLocal = atan2(relVel(2), relVel(1));
end

forwardLocal = [cos(headingLocal), sin(headingLocal)];
[~, targetLength] = sizeAtFramePath(targetPath, frame, opts);
relRear = centerLocal - 0.5 * targetLength * forwardLocal;
end

function [relRear, headingLocal, relVel, relAcc] = estimateTargetCaStateInEgoFrame(targetPath, frame, egoState, opts)
historySteps = max(3, round(opts.HistorySec / opts.DataDt) + 1);
historyFrames = frame - historySteps + 1:frame;
measurements = [];
headings = [];

for hFrame = historyFrames
    [centerLocal, headingLocal, ok] = targetCenterInEgoFrame(targetPath, hFrame, egoState, opts);
    if ~ok
        continue;
    end
    forwardLocal = [cos(headingLocal), sin(headingLocal)];
    [~, targetLength] = sizeAtFramePath(targetPath, hFrame, opts);
    relRear = centerLocal - 0.5 * targetLength * forwardLocal;
    measurements(end + 1, :) = relRear; %#ok<AGROW>
    headings(end + 1, 1) = headingLocal; %#ok<AGROW>
end

if size(measurements, 1) < 2
    [relRear, headingLocal, relVel, relAcc] = targetKinematicsInEgoFrame(targetPath, frame, egoState, opts);
    return;
end

dt = opts.DataDt;
A1 = [1, dt, 0.5 * dt^2; 0, 1, dt; 0, 0, 1];
A = blkdiag(A1, A1);
H = [1, 0, 0, 0, 0, 0; 0, 0, 0, 1, 0, 0];
q = opts.KalmanProcessNoise^2;
Q1 = q * [dt^5 / 20, dt^4 / 8, dt^3 / 6; dt^4 / 8, dt^3 / 3, dt^2 / 2; dt^3 / 6, dt^2 / 2, dt];
Q = blkdiag(Q1, Q1);
R = opts.KalmanMeasurementNoise^2 * eye(2);

initialVel = (measurements(2, :) - measurements(1, :)) / dt;
x = [measurements(1, 2); initialVel(2); 0; measurements(1, 1); initialVel(1); 0];
P = diag([1, 10, 10, 1, 10, 10]);

for i = 1:size(measurements, 1)
    if i > 1
        x = A * x;
        P = A * P * A' + Q;
    end

    z = [measurements(i, 2); measurements(i, 1)];
    innovation = z - H * x;
    S = H * P * H' + R;
    K = P * H' / S;
    x = x + K * innovation;
    P = (eye(6) - K * H) * P;
end

relRear = [x(4), x(1)];
relVel = [x(5), x(2)];
relAcc = [x(6), x(3)];

if norm(relVel) > 1.0e-6
    headingLocal = atan2(relVel(2), relVel(1));
elseif ~isempty(headings)
    headingLocal = circularMean(headings);
else
    headingLocal = 0;
end
end

function v = clipVector(v, maxNorm)
if ~isfinite(maxNorm) || maxNorm <= 0
    return;
end
n = norm(v);
if n > maxNorm
    v = v * (maxNorm / n);
end
end

function [centerLocal, headingLocal, ok] = targetCenterInEgoFrame(targetPath, frame, egoState, opts)
[targetRear, ok] = pointAtFrame(targetPath, frame, true);
if ~ok
    centerLocal = [0, 0];
    headingLocal = 0;
    return;
end
targetDir = directionAtFrame(targetPath, frame, opts);
[~, targetLength] = sizeAtFramePath(targetPath, frame, opts);
targetCenter = targetRear + 0.5 * targetLength * targetDir;
rel = targetCenter - egoState.front;
centerLocal = [dot(rel, egoState.forward), dot(rel, egoState.left)];
targetDirLocal = [dot(targetDir, egoState.forward), dot(targetDir, egoState.left)];
headingLocal = atan2(targetDirLocal(2), targetDirLocal(1));
end

function [egoState, ok] = egoBasisAtFrame(data, frame, EGO_VEHICLE, opts)
egoPath = getPath(data, 0, 0);
[egoCenter, ok] = pointAtFrame(egoPath, frame, true);
if ~ok
    egoState = struct('center', [0, 0], 'front', [0, 0], 'forward', [1, 0], 'left', [0, 1]);
    return;
end
forward = directionAtFrame(egoPath, frame, opts);
left = [-forward(2), forward(1)];
egoState.center = egoCenter;
egoState.forward = forward;
egoState.left = left;
egoState.front = egoCenter + 0.5 * EGO_VEHICLE.LENGTH * forward;
end

function path = getPath(data, agentId, sampleId)
mask = data.agent == agentId & data.sample == sampleId;
rows = data(mask, :);
if isempty(rows)
    path.frames = [];
    path.xy = [];
    path.yaw = [];
    path.width = [];
    path.length = [];
    return;
end
[frames, order] = sort(double(rows.frame));
xy = [double(rows.x(order)), double(rows.y(order))];
yaw = [];
if any(strcmp('yaw', data.Properties.VariableNames))
    yaw = double(rows.yaw(order));
end
width = [];
if any(strcmp('width', data.Properties.VariableNames))
    width = double(rows.width(order));
end
length = [];
if any(strcmp('length', data.Properties.VariableNames))
    length = double(rows.length(order));
end
[frames, uniqueIdx] = unique(frames, 'stable');
path.frames = frames(:);
path.xy = xy(uniqueIdx, :);
if ~isempty(yaw)
    path.yaw = yaw(uniqueIdx);
else
    path.yaw = [];
end
if ~isempty(width)
    path.width = width(uniqueIdx);
else
    path.width = [];
end
if ~isempty(length)
    path.length = length(uniqueIdx);
else
    path.length = [];
end
end

function [width, length] = vehicleSizeAtFrame(data, agentId, sampleId, frame, opts)
path = getPath(data, agentId, sampleId);
[width, length] = sizeAtFramePath(path, frame, opts);
end

function [width, length] = sizeAtFramePath(path, frame, opts)
width = opts.CarWidth;
length = opts.CarLength;
if isempty(path.frames)
    return;
end
if isfield(path, 'width') && ~isempty(path.width)
    value = interpFinite(path.frames, path.width, frame);
    if isfinite(value) && value > 0
        width = value;
    end
end
if isfield(path, 'length') && ~isempty(path.length)
    value = interpFinite(path.frames, path.length, frame);
    if isfinite(value) && value > 0
        length = value;
    end
end
end

function value = interpFinite(frames, values, frame)
value = NaN;
frames = double(frames(:));
values = double(values(:));
mask = isfinite(frames) & isfinite(values);
frames = frames(mask);
values = values(mask);
if isempty(frames)
    return;
end
if numel(frames) == 1
    value = values(1);
else
    value = interp1(frames, values, double(frame), 'nearest', 'extrap');
end
end

function [pt, ok] = pointAtFrame(path, frame, allowExtrapolation)
if isempty(path.frames)
    pt = [0, 0];
    ok = false;
    return;
end
frame = double(frame);
if numel(path.frames) == 1
    pt = path.xy(1, :);
    ok = frame == path.frames(1) || allowExtrapolation;
    return;
end
if allowExtrapolation
    pt = [interp1(path.frames, path.xy(:, 1), frame, 'linear', 'extrap'), ...
          interp1(path.frames, path.xy(:, 2), frame, 'linear', 'extrap')];
    ok = all(isfinite(pt));
elseif frame >= min(path.frames) && frame <= max(path.frames)
    pt = [interp1(path.frames, path.xy(:, 1), frame, 'linear'), ...
          interp1(path.frames, path.xy(:, 2), frame, 'linear')];
    ok = all(isfinite(pt));
else
    pt = [0, 0];
    ok = false;
end
end

function dirVec = directionAtFrame(path, frame, opts)
if isfield(path, 'yaw') && ~isempty(path.yaw)
    yaw = yawAtFrame(path, frame);
    if isfinite(yaw)
        dirVec = [cos(yaw), sin(yaw)];
        return;
    end
end

if isempty(path.frames) || size(path.xy, 1) < 2
    dirVec = [1, 0];
    return;
end
prevFrame = double(frame) - 1;
nextFrame = double(frame) + 1;
[prevPt, okPrev] = pointAtFrame(path, prevFrame, true);
[nextPt, okNext] = pointAtFrame(path, nextFrame, true);
if okPrev && okNext
    delta = nextPt - prevPt;
else
    [~, idx] = min(abs(path.frames - double(frame)));
    if idx == 1
        delta = path.xy(2, :) - path.xy(1, :);
    elseif idx == size(path.xy, 1)
        delta = path.xy(end, :) - path.xy(end - 1, :);
    else
        delta = path.xy(idx + 1, :) - path.xy(idx - 1, :);
    end
end
if norm(delta) < max(1e-9, opts.DataDt * 1e-9)
    dirVec = [1, 0];
else
    dirVec = delta / norm(delta);
end
end

function yaw = yawAtFrame(path, frame)
if isempty(path.frames) || isempty(path.yaw)
    yaw = NaN;
    return;
end

valid = isfinite(path.frames) & isfinite(path.yaw(:));
frames = path.frames(valid);
yaws = path.yaw(valid);
if isempty(frames)
    yaw = NaN;
    return;
end

if numel(frames) == 1
    yaw = yaws(1);
else
    yaws = unwrap(yaws);
    yaw = interp1(frames, yaws, double(frame), 'linear', 'extrap');
end

if isfinite(yaw)
    yaw = atan2(sin(yaw), cos(yaw));
end
end

function m = circularMean(angles)
if isempty(angles)
    m = 0;
else
    m = atan2(mean(sin(angles)), mean(cos(angles)));
end
end

function name = filenameOnly(pathValue)
[~, name, ext] = fileparts(char(pathValue));
name = [name ext];
end

function stem = stripExtension(name)
[~, stem] = fileparts(name);
end
