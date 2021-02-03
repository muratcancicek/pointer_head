from scipy.io import wavfile
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
import os

def readMeetingAudio(audioPath):
    meetingAudio_sr, meetingAudio = wavfile.read(audioPath)
    if not ((meetingAudio[:, 0] == meetingAudio[:, 1]).any() == False):
        meetingAudio = meetingAudio[:, 1]
    return meetingAudio_sr, meetingAudio

def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def getNoiseIntervals(audio, minNoiseLength, volThreshold = 2000):
    audio = abs(audio)
    loudIndices = np.where(audio > volThreshold)[0]
    loudStartIndex = loudIndices[0]
    lastLoudIndex = loudIndices[0]
    noiseIntervals = []
    for i in range(len(loudIndices[1:-1])):
        if lastLoudIndex + minNoiseLength >= loudIndices[i]:
            if loudIndices[i] + minNoiseLength < loudIndices[i+1]:
                noiseIntervals.append((loudStartIndex, loudIndices[i]))
        else:
            loudStartIndex = loudIndices[i]
        lastLoudIndex = loudIndices[i]
    if len(loudIndices) > 0:
        if len(noiseIntervals) == 0: 
            noiseIntervals.append((loudStartIndex, loudIndices[-1]))
        elif noiseIntervals[-1] != (loudStartIndex, loudIndices[i]):
            noiseIntervals.append((loudStartIndex, loudIndices[i]))
    return noiseIntervals

def findTrueCorrelation(audio, fltr, corrWaveLen = 50):
    corrs = []
    if len(audio) <= corrWaveLen or len(audio) <= len(fltr)/5:
        return 0
    elif len(audio) > len(fltr):
        windows = rolling_window(audio, len(fltr))
        windows = [windows[i] for i in 
                   range(0, len(windows), int(len(windows)/10))]
        corrs = [np.corrcoef(row, fltr)[0, 1] for row in windows]
    else:
        for i in range(corrWaveLen):
            corrs.append(np.corrcoef(audio[i:], fltr[:len(audio)-i])[0, 1])
    if not corrs: return 0
    return max(corrs)

def getBeepIntervals(meetingAudio, beepAudio, sampleRate, volThreshold=2000, 
                     beepConfidence=0.3, corrWaveLen=50, minNoiseLength=None):
    if not minNoiseLength: minNoiseLength = sampleRate/25
    noiseIntervals = getNoiseIntervals(meetingAudio, 
                                       minNoiseLength, volThreshold)
    beepIntervals = []
    for beginIndex, endIndex in noiseIntervals:
        noise = meetingAudio[beginIndex:endIndex]
        corr = findTrueCorrelation(noise, beepAudio, corrWaveLen)
        if corr > beepConfidence:
            beepIntervals.append((beginIndex, endIndex, corr))
    return beepIntervals

def secondsToTime(seconds):
    return '%02d:%02d' % (int(seconds/60), int(seconds % 60))

def waveIndexToSeconds(index, sampleRate):
    return index / sampleRate

def waveIndexToTime(index, sampleRate):
    return secondsToTime(index / sampleRate)

def detectBeeps(meetingAudio, beepAudio, sampleRate):
    beepIntervals = getBeepIntervals(meetingAudio, beepAudio, sampleRate)
    beeps = []
    for beginIndex, endIndex, corr in beepIntervals:
        index = int((endIndex + beginIndex) / 2)
        beep = (waveIndexToTime(index, sampleRate), 
                waveIndexToSeconds(index, sampleRate), 
                corr, index, beginIndex, endIndex)
        beeps.append(beep)
    print('%d Beep sounds has been detected.' % len(beeps))
    return beeps

def printBeeps(beeps):
    for b in beeps:
        print('\t%s - %.2f %.2f %d %d %d' % b)

def detectAllBeeps(meetingAudioFiles, beepFile):
    beepAudio_sr, beepAudio = wavfile.read(beepFile)
    meetingAudios = [readMeetingAudio(f) for f in meetingAudioFiles]
    names = [f.split(os.path.sep)[-1].split('.')[0] for f in meetingAudioFiles]
    meetingBeeps = {}
    for name, (sampleRate, mAudio) in zip(names, meetingAudios):
        print('For %s;' % name)
        meetingBeeps[name] = detectBeeps(mAudio, beepAudio, sampleRate)
    return meetingBeeps


def getTrailIntervals(beeps, trails, sampleRate):
    intervals = []
    for i in range(len(beeps)-1):
        beginIndex, endIndex = beeps[i][-1], beeps[i+1][-2]
        duration = (endIndex - beginIndex)/sampleRate
        frameCount = duration*trails['fps'][0]
        rang = 0.7
        for j in range(len(trails['duration'])):
            if duration > trails['duration'][j] - rang and \
                duration < trails['duration'][j] + rang and \
                duration > trails['duration'][j]:
                 intervals.append((trails.iloc[j]['name'], duration, 
                                   trails['duration'][j], frameCount,  
                                  trails['frameCount'][j], beginIndex, endIndex,
                                  beeps[i], beeps[i+1]))
                 break
    return intervals

def getTrailsInfo():
    trailsFile = 'C:\\cStorage\\Datasets\\WhiteBallExp\\' + \
      'trails\\25fps\\data\\summaries_2021-01-14 02-02-21.csv'
    return pd.read_csv(trailsFile)


def getTrailIntervalsDataFrame(beeps, trails, sampleRate, fps = 25):
    intervals = getTrailIntervals(beeps, trails, sampleRate)
    columns = ['Name', 'RecordingDuration', 'TrailDuration', 
               'RecordingFrameCount',  'TrailFrameCount', 'beginIndex',
               'endIndex', 'BeginBeep', 'endBeep']
    df = pd.DataFrame(intervals, columns = columns).set_index('Name')
    df['TimeGap'] = df['RecordingDuration'] - df['TrailDuration']
    df['FrameGap'] = df['RecordingFrameCount'] - df['TrailFrameCount']
    df['BeginFrame'] = fps*df['beginIndex']/sampleRate + df['FrameGap']
    return df

def printAllBeeps():
    meetingAudioFiles = ['z.wav', 'zoom2.wav', 'P1.wav', 'P2.wav']
    beepFile = 'beep.wav'
    meetingBeeps = detectAllBeeps(meetingAudioFiles, beepFile)
    for name, beeps in meetingBeeps.items():
        print('In %s;' % name)
        printBeeps(beeps)
        print()

def initializeRecorder( path, id, trailName, fps = 25, dims = (1280, 720)):
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    dir = path + ('%s%s' % (id, os.path.sep))
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok = True)
    now = str(datetime.now())[:-7].replace(':', '-').replace(' ', '_')
    recordName = trailName + '_%s_%s.avi' % (id, now)
    print(dir + recordName, 'will be written')
    return  cv2.VideoWriter(dir + recordName, fourcc, fps, dims)

def showFrame(frame, delay = 1):
    cv2.imshow('frame', frame)
    k = cv2.waitKey(delay)
    if k == 27 or k == ord('q'):
        return False
    else:
        return True 

def openVideo(path):
    cap = cv2.VideoCapture(path)
    frameCount = 0
    ret = True
    frames = []
    while(ret):
        ret, frame = cap.read()
        if not ret:
            if frameCount < 1:
                print('Something Wrong')
            break
        frameCount += 1
        #if frameCount > 7000:
        #    break
        frames.append(frame)
    cap.release()    
    return frames

def writeTrailRecording(video, id, trailName, beginIndex, length):
    beginIndex = int(beginIndex)
    recorder = initializeRecorder('', id, trailName)
    for f in video[beginIndex:beginIndex+length]:
        recorder.write(f.astype(np.uint8))
    recorder.release()

def main():
    beepAudio_sr, beepAudio = wavfile.read('beep.wav')
    _, meetingAudio = readMeetingAudio('P2.wav')
    beeps = detectBeeps(meetingAudio, beepAudio, beepAudio_sr)
    trails = getTrailsInfo()
    df = getTrailIntervalsDataFrame(beeps, trails, beepAudio_sr)
    #print(df['BeginFrame'])

    p = 'C:\\cStorage\\Datasets\\WhiteBallExp\\Subjects\\Actual Zoom Recordings\\P2.mp4'
    video = openVideo(p)
    for name in df.index:
        t = df.loc[name]
        print(int(t['BeginFrame']))
        #writeTrailRecording(video, '2', name, t['BeginFrame'], t['TrailFrameCount'])
    

if __name__ == '__main__':
    main()
