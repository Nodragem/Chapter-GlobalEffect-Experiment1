#
# Copyright (c) 1996-2005, SR Research Ltd., All Rights Reserved
#
#
# For use by SR Research licencees only. Redistribution and use in source
# and binary forms, with or without modification, are NOT permitted.
#
#
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the distribution.
#
# Neither name of SR Research Ltd nor the name of contributors may be used
# to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
# IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# $Date: 2007/08/27 15:25:31 $
#
#


from pylink import *
from pygame import *
import utils
import gc
import sys
import run_trials
import dummy

import os
os.environ['SDL_VIDEO_CENTERED']='1'
init()


#MyMonitor = utils.Monitor("Geoffrey-PC",1024,720, distance = 20, width_cm = 50)
MyMonitor = utils.Monitor("labo-Tom-PC",1280,1024, distance = 72, width_cm = 36.6)
#MyMonitor = utils.Monitor("Labo-Tom-Big",1024,768, distance = 130, width_cm = 206)
#dummy_mode = True
dummy_mode=False
full_screen = True
#full_screen = False
file_asking = 3
## there are 1600 trials
START = 0 ## from what line of the table we should start
BREAK_INTERVAL = 200 #200 ## display the take-a-break screen every XX trials
FRAME_RATE = 100.0 ## not used for now
calib_type = "HV13" ## try HV13 for 13 dots

## Ask questions on conditions, check if the file already exist:
if file_asking==3:
    info, table, path_to_table_file, path_to_eye_tracker_file = utils.initializeFromConditions()
elif file_asking == 2:
    import Tkinter, tkFileDialog
    root = Tkinter.Tk()
    root.withdraw()
    file_path = tkFileDialog.askopenfilename(title = "Open a Table file:")
    root.destroy()
    info, table, path_to_table_file, path_to_eye_tracker_file = utils.openTable(file_path)
    START = int(raw_input("From what trial would you like to continue?"))

else:
    info, table, path_to_table_file, path_to_eye_tracker_file = utils.openTable("")

print info

if not dummy_mode:
    MyEyelink = EyeLink()
    MyMonitor.setFPSControl(0)
else:
    ## this is my dummy, it works better than the official one
    MyEyelink = dummy.DummyEyeLink()
    MyMonitor.setFPSControl(FRAME_RATE) ## force 100 Hz during office-test ## switch to zero when running the experiment

## Initializes the graphics
display.init()
mouse.set_visible(False)
## do a new display surface
if full_screen:
    display.set_mode((MyMonitor.w, MyMonitor.h), FULLSCREEN |DOUBLEBUF | HWSURFACE,32)
else:
    display.set_mode((MyMonitor.w, MyMonitor.h), NOFRAME |DOUBLEBUF,32)

surf = display.get_surface()
MyEnv = utils.Environment(surf, MyMonitor,
    MyEyelink, table, info,calib_type = calib_type, units = 'deg')

event = utils.runCalibration(MyEnv)


## Gets the new display surface and sends a mesage to EDF file;
edfFileName = path_to_eye_tracker_file
## opens the EDF file.
MyEyelink.openDataFile(edfFileName)
pylink.flushGetkeyQueue();
MyEyelink.setOfflineMode();
utils.configEDFfile(MyEyelink)

MyEyelink.sendMessage("Info Experiment: %s"%(str(info)))
MyEyelink.sendCommand("screen_pixel_coords =  0 0 %d %d" %(surf.get_rect().w, surf.get_rect().h))
MyEyelink.sendMessage("DISPLAY_COORDS  0 0 %d %d" %(surf.get_rect().w, surf.get_rect().h))

error = 0
try:
    if(MyEyelink.isConnected() and not MyEyelink.breakPressed()):
        print "Let's run the trials"
        error, saveFrames = run_trials.run_trials(MyEnv, START, BREAK_INTERVAL)
except Exception, e:
    print "Caught:", e
finally: # whatever waht happened it will save the data
    print "Experiment ended"
    utils.displayInstruction(MyEnv, "Thank_you.txt", False)
    if error == 0:
        print "with success!"
        utils.saveEyeTrackerData(MyEyelink, edfFileName, edfFileName) ## normally, the src is on the Eyetracker HD, while the destination os on the Display HD
        display.quit()

    else:
        print "before end!"
        display.quit()
        name = raw_input("The experiment was aborted, rename the EDF file:")
        utils.saveEyeTrackerData(MyEyelink, edfFileName, ".\\results\\X%s.EDF"%name)

    MyEyelink.close()
    #print saveFrames
    import numpy as np
    name_fps = ".\\results\\FPS-" + "-".join(info)
    np.savetxt("%s.txt"%name_fps, saveFrames)
    print "Saved and closed!"



