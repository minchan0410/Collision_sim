% Generate DSM/SBEV PNGs next to diffusion generation CSV files.
%
% Default behavior:
%   SBEV_Catalog_FNPrecrash/diffusion_generation_result/**/*
%       *_generation.csv
%       sbev/
%           sample_000/
%               Image_0_..._frame_000000_gen.png
%           sample_001/
%               Image_13_..._frame_000000_gen.png
%           summary.csv
%   SBEV_Catalog_FNPrecrash/diffusion_generation_result/Collision_mode_generation/
%       Collision Mode 11/
%       ...
%       Not Crash/
%
% Run from MATLAB:
%   run('SBEV_Catalog_FNPrecrash/generateDSM/run_generateDSM_from_generation_results.m')

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
catalogRoot = fileparts(scriptDir);
repoRoot = fileparts(catalogRoot);

if ~exist('GENERATION_ROOT', 'var') || isempty(GENERATION_ROOT)
    GENERATION_ROOT = fullfile(catalogRoot, 'diffusion_generation_result');
else
    GENERATION_ROOT = localAbsolutePath(GENERATION_ROOT, repoRoot);
end

if ~exist('CLEAR_OLD_SBEV', 'var') || isempty(CLEAR_OLD_SBEV)
    CLEAR_OLD_SBEV = true;
end

if ~exist('SAMPLE_SELECTION', 'var') || isempty(SAMPLE_SELECTION)
    SAMPLE_SELECTION = 'all';
end

if ~exist('FRAME_SELECTION', 'var') || isempty(FRAME_SELECTION)
    FRAME_SELECTION = 'all';
end

if ~exist('MAX_FILES', 'var') || isempty(MAX_FILES)
    MAX_FILES = inf;
end

if ~exist('MAX_FRAMES', 'var') || isempty(MAX_FRAMES)
    MAX_FRAMES = inf;
end

if ~exist('COLLISION_MODE_ROOT', 'var') || isempty(COLLISION_MODE_ROOT)
    COLLISION_MODE_ROOT = fullfile(GENERATION_ROOT, 'Collision_mode_generation');
else
    COLLISION_MODE_ROOT = localAbsolutePath(COLLISION_MODE_ROOT, repoRoot);
end

addpath(scriptDir);
addpath(fullfile(scriptDir, 'AVlib', 'dECISION', 'simplifiedBEV'));

csvFiles = localFindGenerationCsvs(GENERATION_ROOT);
if isfinite(MAX_FILES)
    csvFiles = csvFiles(1:min(numel(csvFiles), MAX_FILES));
end

if isempty(csvFiles)
    error('[DSM/SBEV] No *_generation.csv files found under: %s', GENERATION_ROOT);
end

fprintf('[DSM/SBEV] generation root: %s\n', GENERATION_ROOT);
fprintf('[DSM/SBEV] csv files: %d\n', numel(csvFiles));

totalImages = 0;
allSummary = table();
for iFile = 1:numel(csvFiles)
    csvPath = csvFiles{iFile};
    csvDir = fileparts(csvPath);
    sbevOutDir = fullfile(csvDir, 'sbev');

    if CLEAR_OLD_SBEV && exist(sbevOutDir, 'dir')
        rmdir(sbevOutDir, 's');
    end
    if ~exist(sbevOutDir, 'dir')
        mkdir(sbevOutDir);
    end

    fprintf('[DSM/SBEV] %d/%d %s -> %s\n', iFile, numel(csvFiles), csvPath, sbevOutDir);
    summary = generateDSM_from_csv(csvPath, sbevOutDir, ...
        'Sample', SAMPLE_SELECTION, ...
        'Frame', FRAME_SELECTION, ...
        'MaxFrames', MAX_FRAMES, ...
        'UseCsvSubfolder', false);

    if ~isempty(summary)
        totalImages = totalImages + height(summary);
        allSummary = [allSummary; summary]; %#ok<AGROW>
    end
end

classifiedImages = localBuildCollisionModeFolders(allSummary, COLLISION_MODE_ROOT, CLEAR_OLD_SBEV);

fprintf('[DSM/SBEV] done. images=%d classified=%d\n', totalImages, classifiedImages);

function csvFiles = localFindGenerationCsvs(rootDir)
csvFiles = {};
rootDir = char(rootDir);
if ~exist(rootDir, 'dir')
    return;
end

entries = dir(rootDir);
for i = 1:numel(entries)
    name = entries(i).name;
    if strcmp(name, '.') || strcmp(name, '..')
        continue;
    end

    fullPath = fullfile(entries(i).folder, name);
    if entries(i).isdir
        childFiles = localFindGenerationCsvs(fullPath);
        csvFiles = [csvFiles; childFiles(:)]; %#ok<AGROW>
    elseif ~isempty(regexp(name, '_generation\.csv$', 'once'))
        csvFiles{end + 1, 1} = fullPath; %#ok<AGROW>
    end
end

csvFiles = sort(csvFiles);
end

function copiedCount = localBuildCollisionModeFolders(summaryTable, outRoot, clearOld)
validModes = [11, 12, 13, 21, 23, 31, 33, 41, 43, 51, 52, 53];
outRoot = char(outRoot);
copiedCount = 0;

if clearOld && exist(outRoot, 'dir')
    rmdir(outRoot, 's');
end
if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end

for mode = validModes
    modeDir = fullfile(outRoot, sprintf('Collision Mode %d', mode));
    if ~exist(modeDir, 'dir')
        mkdir(modeDir);
    end
end
notCrashDir = fullfile(outRoot, 'Not Crash');
if ~exist(notCrashDir, 'dir')
    mkdir(notCrashDir);
end

if isempty(summaryTable) || height(summaryTable) == 0
    return;
end

for i = 1:height(summaryTable)
    if any(strcmp('png_name', summaryTable.Properties.VariableNames))
        mode = localCollisionModeFromImageName(localCellString(summaryTable.png_name, i));
    elseif any(strcmp('collision_mode', summaryTable.Properties.VariableNames))
        mode = round(double(summaryTable.collision_mode(i)));
    else
        mode = 0;
    end
    mode = localNormalizeCollisionMode(mode);
    modeDir = localCollisionModeDir(outRoot, mode);

    srcPath = localCellString(summaryTable.png_path, i);
    if isempty(srcPath) || ~exist(srcPath, 'file')
        continue;
    end

    [~, baseName, ext] = fileparts(srcPath);
    dstPath = fullfile(modeDir, [baseName ext]);
    if exist(dstPath, 'file')
        sampleId = 0;
        if any(strcmp('sample', summaryTable.Properties.VariableNames))
            sampleId = round(double(summaryTable.sample(i)));
        end
        frameId = 0;
        if any(strcmp('frame', summaryTable.Properties.VariableNames))
            frameId = round(double(summaryTable.frame(i)));
        end
        dstPath = localUniqueCopyPath(modeDir, baseName, ext, sampleId, frameId);
    end

    copyfile(srcPath, dstPath);
    copiedCount = copiedCount + 1;
end

fprintf('[DSM/SBEV] collision mode folders: %s images=%d\n', outRoot, copiedCount);
end

function mode = localNormalizeCollisionMode(mode)
validModes = [0, 11, 12, 13, 21, 23, 31, 33, 41, 43, 51, 52, 53];
if ~isfinite(mode) || ~ismember(mode, validModes)
    mode = 0;
end
end

function modeDir = localCollisionModeDir(outRoot, mode)
if mode == 0
    modeDir = fullfile(outRoot, 'Not Crash');
else
    modeDir = fullfile(outRoot, sprintf('Collision Mode %d', mode));
end
if ~exist(modeDir, 'dir')
    mkdir(modeDir);
end
end

function mode = localCollisionModeFromImageName(imageName)
mode = 0;
tokens = regexp(char(imageName), '^Image_(\d+)_', 'tokens', 'once');
if ~isempty(tokens)
    mode = str2double(tokens{1});
end
mode = localNormalizeCollisionMode(mode);
end

function value = localCellString(column, index)
if iscell(column)
    value = column{index};
elseif isstring(column)
    value = char(column(index));
else
    value = char(column(index, :));
end
end

function dstPath = localUniqueCopyPath(modeDir, baseName, ext, sampleId, frameId)
candidate = fullfile(modeDir, sprintf('%s_sample_%03d%s', baseName, sampleId, ext));
if ~exist(candidate, 'file')
    dstPath = candidate;
    return;
end

idx = 1;
while true
    candidate = fullfile(modeDir, sprintf('%s_sample_%03d_frame_%06d_dup_%03d%s', ...
        baseName, sampleId, frameId, idx, ext));
    if ~exist(candidate, 'file')
        dstPath = candidate;
        return;
    end
    idx = idx + 1;
end
end

function pathOut = localAbsolutePath(pathIn, baseDir)
pathOut = char(pathIn);
if isempty(pathOut)
    return;
end
isUnixAbsolute = startsWith(pathOut, filesep);
isWindowsAbsolute = ~isempty(regexp(pathOut, '^[A-Za-z]:[\\/]', 'once'));
if ~(isUnixAbsolute || isWindowsAbsolute)
    pathOut = fullfile(baseDir, pathOut);
end
end
