
import cv2
import numpy as np
from matplotlib import pyplot as plt

from pulse import Pulse
import time
from threading import Lock, Thread
from plot_cont import DynamicPlot
from capture_frames import CaptureFrames
from process_mask import ProcessMasks

from utils import *
import multiprocessing as mp
import sys
from optparse import OptionParser

class RunPOS():
    # def test1(self):
    #     import cv2
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #
    #     from imutils.video import VideoStream
    #     from xlsxwriter import Workbook
    #
    #     fig = plt.figure()
    #
    #     plt.ion()  # Set interactive mode on
    #
    #     xs = []
    #     blue = []
    #     red = []
    #     green = []
    #
    #     b_frame = []
    #     g_frame = []
    #     r_frame = []
    #     s_frame = []
    #
    #     # We will be using Video-capture to get the fps value.
    #     capture = cv2.VideoCapture(0)
    #     fps = capture.get(cv2.CAP_PROP_FPS)
    #     capture.release()
    #
    #     # New module: VideoStream
    #     vs = VideoStream().start()
    #
    #     frame_count = 0
    #     second = 1
    #
    #     is_new_frame = False
    #
    #     while True:
    #         frame = vs.read()
    #
    #         if frame is None:
    #             break
    #
    #         if frame_count % int(fps) == 0:
    #             b, g, r = cv2.split(frame)
    #
    #             is_new_frame = True  # New frame has come
    #
    #             line = [line for line in zip(b, g, r) if len(line)]
    #
    #             s_frame.append(second)
    #             b_frame.append(np.mean(line[0]) * 0.02)
    #             g_frame.append(np.mean(line[1]) * 0.03)
    #             r_frame.append(np.mean(line[2]) * 0.04)
    #
    #             plt.plot(s_frame, b_frame, 'b', label='blue', lw=7)
    #             plt.plot(s_frame, g_frame, 'g', label='green', lw=4)
    #             plt.plot(s_frame, r_frame, 'r', label='red')
    #             plt.xlabel('seconds')
    #             plt.ylabel('mean')
    #             if frame_count == 0:
    #                 plt.legend()
    #             plt.show()
    #
    #             second += 1
    #
    #         elif second > 2:
    #
    #             if is_new_frame:
    #
    #                 if second == 3:
    #                     blue.extend(b_frame)
    #                     green.extend(g_frame)
    #                     red.extend(r_frame)
    #                     xs.extend(s_frame)
    #                 else:
    #                     blue.append(b_frame[len(b_frame) - 1])
    #                     green.append(g_frame[len(g_frame) - 1])
    #                     red.append(r_frame[len(r_frame) - 1])
    #                     xs.append(s_frame[len(s_frame) - 1])
    #
    #                 del b_frame[0]
    #                 del g_frame[0]
    #                 del r_frame[0]
    #                 del s_frame[0]
    #
    #                 is_new_frame = False  # we added the new frame to our list structure
    #
    #         cv2.imshow('Frame', frame)
    #
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #
    #         frame_count += 1
    #
    #     cv2.destroyAllWindows()
    #     capture.release()
    #     vs.stop()
    #
    #     book = Workbook('Channel.xlsx')
    #     sheet = book.add_worksheet()
    #
    #     row = 0
    #     col = 0
    #
    #     sheet.write(row, col, 'Seconds')
    #     sheet.write(row + 1, col, 'Blue mean')
    #     sheet.write(row + 2, col, 'Green mean')
    #     sheet.write(row + 3, col, 'Red mean')
    #
    #     col += 1
    #
    #     for s, b, g, r in zip(xs, blue, green, red):
    #         sheet.write(row, col, s)
    #         sheet.write(row + 1, col, b)
    #         sheet.write(row + 2, col, g)
    #         sheet.write(row + 3, col, r)
    #         col += 1
    #
    #     book.close()
    def __init__(self,  sz=270, fs=28, bs=30, plot=False):
        self.batch_size = bs
        self.frame_rate = fs
        self.signal_size = sz
        self.plot = plot


    def __call__(self, source):
        time1=time.time()
        
        mask_process_pipe, chil_process_pipe = mp.Pipe()
        self.plot_pipe = None
        if self.plot:
            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plotter = DynamicPlot(self.signal_size, self.batch_size)
            self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
            self.plot_process.start()
        
        process_mask = ProcessMasks(self.signal_size, self.frame_rate, self.batch_size)

        mask_processer = mp.Process(target=process_mask, args=(chil_process_pipe, self.plot_pipe, source, ), daemon=True)
        mask_processer.start()
        print("ok")
        capture = CaptureFrames(self.batch_size, source, show_mask=True)
        capture(mask_process_pipe, source)

        mask_processer.join()
        if self.plot:
            self.plot_process.join()

        time2=time.time()
        time2=time.time()
        print(f'time {time2-time1}')

def get_args():
    parser = OptionParser()
    parser.add_option('-s', '--source', dest='source', default=0,
                        help='Signal Source: 0 for webcam or file path')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=30,
                        type='int', help='batch size')
    parser.add_option('-f', '--frame-rate', dest='framerate', default=25,
                        help='Frame Rate')

    (options, _) = parser.parse_args()
    return options
        
if __name__=="__main__":
    args = get_args()
    source = args.source
    runPOS = RunPOS(270, args.framerate, args.batchsize, True)
    runPOS(source)


