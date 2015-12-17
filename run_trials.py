#
# Copyright (c) 1996-2005, SR Research Ltd., All Rights Reserved
#
#


from pylink import *
import utils
import pygame
from pygame import display
import gc
import sys
import numpy as np


TARGET_SIZE = utils.TARGET_SIZE
FIXATION_SIZE = utils.FIXATION_SIZE
GREEN = utils.GREEN
RED = utils.RED
DISTRACTOR_SIZE = utils.DISTRACTOR_SIZE
#DISTRACTOR_SIZE_X, DISTRACTOR_SIZE_Y  = utils.DISTRACTOR_SIZE_X, utils.DISTRACTOR_SIZE_Y
BACKGROUND = utils.BACKGROUND

## Everything will be passed by the main module
FPS_CONTROL = 0
FRAME_INTERVALS = []
MyEyelink = None
MyMonitor = None
MySurface = None
MyTable = None
MyInfo = None
dummy = False
target = None
distractor = None
fixation = None
text = None
fps = None
clock = pygame.time.Clock()


def initStimuliX(MyEnv):
    global target, distractor, fixation, text, fps
    target = utils.CrossDiag(MyEnv,size=DISTRACTOR_SIZE, line_width = 2)
      ## diameter
    fixation = utils.Cross(MyEnv,size=FIXATION_SIZE) ## size of the container box
    text = utils.NormalText (MyEnv, "Hello", pos=(0,0))
    fps = utils.NormalText (MyEnv, "fps", pos=(0,20))
    fixation.setFillColor((255,255,255))
    target.setFillColor(RED)
    if MyInfo[0] == "same":
        distractor = utils.CrossDiag(MyEnv,size=DISTRACTOR_SIZE, line_width = 2)
        distractor.color1 = RED
    elif MyInfo[0] == "different":
        #distractor = utils.TriangleDown(MyEnv,size=DISTRACTOR_SIZE, line_width = 2) ## size of the container box
        distractor = utils.Circle(MyEnv,size=TARGET_SIZE, line_width = 2)
        distractor.color1 = RED
    else:
        print "Error, the ditractor have to be 'same' or 'different' from the target"
    distractor.color2 = (0,0,0, 0) ## display a black distractor (invisible on a black backgroung :P )

def initStimuliO(MyEnv):
    global target, distractor, fixation, text, fps
    target = utils.Circle(MyEnv,size=TARGET_SIZE, line_width = 2) ## diameter
    fixation = utils.Cross(MyEnv,size=FIXATION_SIZE) ## size of the container box
    text = utils.NormalText (MyEnv, "Hello", pos=(0,0))
    fps = utils.NormalText (MyEnv, "fps", pos=(0,20))
    fixation.setFillColor((255,255,255))
    target.setFillColor(RED)
    if MyInfo[0] == "same":
        distractor = utils.Circle(MyEnv,size=TARGET_SIZE, line_width = 2)
        distractor.color1 = RED ## ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    elif MyInfo[0] == "different":
        #distractor = utils.TriangleDown(MyEnv,size=DISTRACTOR_SIZE, line_width = 2) ## size of the container box
        distractor = utils.CrossDiag(MyEnv,size=DISTRACTOR_SIZE, line_width = 2) ## size of the container box
        distractor.color1 = RED
    else:
        print "Error, the ditractor have to be 'same' or 'different' from the target"
    distractor.color2 = (0,0,0, 0) ##

def end_trial():
    '''Ends recording: adds 100 msec of data to catch final events'''
    ## should clear the screen!!!!!!!!!!!
    pylink.endRealTimeMode();
    pumpDelay(100);
    MyEyelink.stopRecording();
    while MyEyelink.getkey() :
        pass;

def giveParametersToEyeTracker(par): ## not needed here but can be personnalized
    MyEyelink.sendMessage("!V TRIAL_VAR trial_num  %d" %par[0] )
    MyEyelink.sendMessage("!V TRIAL_VAR trial_type  %d" %par[1] )
    MyEyelink.sendMessage("!V TRIAL_VAR target_ecc  %d" %par[2] )
    MyEyelink.sendMessage("!V TRIAL_VAR target_dir  %d" %par[3] )
    MyEyelink.sendMessage("!V TRIAL_VAR distractor_dir  %d" %par[4] )
    MyEyelink.sendMessage("!V TRIAL_VAR distractor_dir  %d" %par[5] )
    MyEyelink.sendMessage("!V TRIAL_VAR distance  %d" %par[6] )

def updateStimuliFromParameters(par):
    ''' | ntrial | trial type | Target ecc. | Target dir. | Distractor ecc. | Distractor dir. | T-D Distance
    trial type: 0 for double, 1 for single, 2 for random-double, 3 for random-single. '''

    global distractor, target, text
    target.setPos(utils.polToCart(par[2], par[3]))
    if (par[1] == 0) or (par[1] == 2):
        #distractor.setFillColor(distractor.color1)
        distractor.drawn = True
        distractor.setPos(utils.polToCart(par[4], par[5]))
    else:
        distractor.drawn = False
        #distractor.setFillColor(distractor.color2) ## it is black and transparent
        distractor.setPos((0,0)) ## it is on the fixation but it is transparent
##        print "im here"

##    print "distractor degrees, cart:", utils.polToCart(par[4],par[5])
##    print "target degrees, cart:", utils.polToCart(par[2],par[3])
##    print "distractor pixel, cart:", distractor.pos - MyMonitor.size/2.0
##    print "target pixel, cart:", target.pos - MyMonitor.size/2.0
    #text.setText("target at %d , %d degrees, type = %d"%(par[2],par[3], par[1]))

def sign(n):
    ''' take 0 or 1 as input, give -1 for 0 and 1 for 1'''
    return n*2-1

def drawCondition(fixation_duration, stimuli_duration, remaining_duration):

    TOTAL_FRAMES = fixation_duration + stimuli_duration + remaining_duration
    STIM_DISAPPEARANCE = fixation_duration + stimuli_duration
    STIM_APPEARANCE = fixation_duration
    MyEyelink.flushKeybuttons(0)
    buttons =(0, 0);
    # Loop of realtime
    for frameN in xrange(TOTAL_FRAMES):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                    k = pygame.key.get_pressed()
                    m = pygame.key.get_mods()
                    if m & pygame.KMOD_CTRL and k[pygame.K_q]:
                        end_trial();
                        print "Crtl + Q pressed"
                        return ABORT_EXPT;
                    elif m & pygame.KMOD_CTRL and k[pygame.K_r]:
                        end_trial();
                        return REPEAT_TRIAL


        MySurface.fill(BACKGROUND)
        # check input (should be in a function)
        if dummy:
            MyEyelink.update()

        error = MyEyelink.isRecording()  # First check if recording is aborted
        if error!=0:
            end_trial();
            return error

##        if(MyEyelink.breakPressed()):   # Checks for program termination or ALT-F4 or CTRL-C keys
##            end_trial();
##            return ABORT_EXPT
##        elif(MyEyelink.escapePressed()): # Checks for local ESC key to abort trial (useful in debugging)
##            end_trial();
##            return SKIP_TRIAL

        # here you draw
        if frameN == 1:
            startTime = currentTime()
            fixation.draw()
            MyEyelink.sendMessage("Fixation ON SYNCTIME %d"%(currentTime()-startTime));
        elif frameN < STIM_APPEARANCE:
            fixation.draw()
        elif frameN == STIM_APPEARANCE:
            startTime = currentTime()
            target.draw()
            distractor.draw()
            MyEyelink.sendMessage("Fixation OFF and Stimuli ON SYNCTIME %d"%(currentTime()-startTime));
        elif frameN > STIM_APPEARANCE and frameN < STIM_DISAPPEARANCE:
            target.draw()
            distractor.draw()
        elif frameN == STIM_DISAPPEARANCE:
            startTime = currentTime()
            target.draw()
            MyEyelink.sendMessage("Distractor OFF SYNCTIME %d"%(currentTime()-startTime));
        elif frameN > STIM_DISAPPEARANCE:
            target.draw()
        if dummy:
            text.draw()
            utils.drawFPS(fps, clock)
        display.flip()
        FRAME_INTERVALS.append(clock.tick_busy_loop(FPS_CONTROL))

    end_trial();

    #The TRIAL_RESULT message defines the end of a trial for the EyeLink Data Viewer.
    #This is different than the end of recording message END that is logged when the trial recording ends.
    #Data viewer will not parse any messages, events, or samples that exist in the data file after this message.
    MyEyelink.sendMessage("TRIAL_RESULT %d"%(buttons[0]));
    return MyEyelink.getRecordingStatus()




def do_trial(par):
    '''Does the simple trial'''
    id_number = str(par) ## par contains the trial parameters
    ##This supplies the title at the bottom of the eyetracker display
#       message ="record_status_message 'Trial %s'"%id_number
#       MyEyelink.sendCommand(message);

    ##Always send a TRIALID message before starting to record.
    ##EyeLink Data Viewer defines the start of a trial by the TRIALID message.
    ##This message is different than the start of recording message START that is logged when the trial recording begins.
    ##The Data viewer will not parse any messages, events, or samples, that exist in the data file prior to this message.

    msg = "TRIALID %s"%id_number;
    MyEyelink.sendMessage(msg);

    ##This TRIAL_VAR command specifies a trial variable and value for the given trial.
    ##Send one message for each pair of trial condition variable and its corresponding value.
    ## You can put this in a function
    updateStimuliFromParameters(par)
    #giveParametersToEyeTracker(par)

    ## you can do a drifcorrection if you want
    ## fonction()

    error = MyEyelink.startRecording(1,1,1,1)
    if error:       return error;
    gc.disable(); # switch off the garbage collector
    #begin the realtime mode
    pylink.beginRealTimeMode(100)

    if not MyEyelink.waitForBlockStart(100, 1, 0):
        end_trial();
        print "ERROR: No link samples received!";
        return TRIAL_ERROR;



    ret_value = drawCondition(*par[-3:].astype(int));
    pylink.endRealTimeMode();
    gc.enable();
    return ret_value;





def run_trials(MyEnv, start, break_interval):
    ''' This function is used to run individual trials and handles the trial return values. '''

    ''' Returns a successful trial with 0, aborting experiment with ABORT_EXPT (3); It also handles
    the case of re-running a trial. '''
    global MySurface, MyMonitor, MyEyelink, MyTable, MyInfo, FPS_CONTROL
    MySurface, MyMonitor, MyEyelink, MyTable, MyInfo = MyEnv.getDetails()
    # Give the screen reference to Stimuli, and initialize them:
    FPS_CONTROL = MyMonitor.fps_control
    # Give the screen reference to Stimuli, and initialize them:
    if MyInfo[-1] == "x":
        initStimuliX(MyEnv)
    else:
        initStimuliO(MyEnv)
    if MyInfo[0] == "same":
        utils.displayInstruction(MyEnv, "instructions-same.txt")
    else:
        utils.displayInstruction(MyEnv, "instructions-different.txt")

    #Do the tracker setup at the beginning of the experiment.
    if MyEyelink.getTrackerVersion() == -1:
        global dummy
        dummy = True
        utils.displayTestScreen(MyEnv, 13., 16) # display test screen in dummy mode;
    nb_trials = len(MyTable[start:,0])

    for i, trial in enumerate(MyTable[start:,:]): ## read the parameter's table line-by-line from the start number

        if(MyEyelink.isConnected() ==0 or MyEyelink.breakPressed()): break;

        if i % break_interval == 0 and i>0:
            s = "Part %d on %d achieved !"%(i / break_interval, nb_trials/break_interval )
            event = utils.displayInstruction(MyEnv, "waiting_message.txt", additional_text = s)
            if event.key == pygame.K_r:
                event = utils.runCalibration(MyEnv)
            if event.key == pygame.K_ESCAPE:
                MyEyelink.sendMessage("EXPERIMENT ABORTED")
                return ABORT_EXPT, FRAME_INTERVALS;

        while 1:


            ret_value = do_trial(trial)
            endRealTimeMode()

            if (ret_value == TRIAL_OK):
                MyEyelink.sendMessage("TRIAL OK");
                break;
            elif (ret_value == SKIP_TRIAL):
                MyEyelink.sendMessage("TRIAL %s SKIPPED"%str(i));
                print "TRIAL %s SKIPPED"%str(i)
                break;
            elif (ret_value == ABORT_EXPT):
                MyEyelink.sendMessage("EXP. ABORTED AT TRIAL %s"%str(i));
                print "EXP. ABORTED AT TRIAL %s"%str(i)
                return ABORT_EXPT, FRAME_INTERVALS;
            elif (ret_value == REPEAT_TRIAL):
                utils.runCalibration(MyEnv)
                MyEyelink.sendMessage("TRIAL REPEATED after Calibration");
            else:
                MyEyelink.sendMessage("TRIAL ERROR")
                break;

    return 0, FRAME_INTERVALS;

