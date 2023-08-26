import numpy as np
import cv2
import ffmpeg  #ffmpeg-python
in_file='rtp://192.168.42.1:55004/'#?overrun_nonfatal=1?buffer_size=10000000?fifo_size=100000'
# ffmpeg -stream_loop 5 -re -i OxfordStreet.avi -vcodec libx264 -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:23000/live.sdp

width = 320
height = 240
cv2.namedWindow("test")

process1 = (
    ffmpeg
    .input(in_file,rtsp_flags= 'listen')
    .output('pipe:', format='rawvideo', pix_fmt='bgr24')
    .run_async(pipe_stdout=True)
)
while True:
    in_bytes = process1.stdout.read(width * height * 3)
    if not in_bytes:
        break
    in_frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([height, width, 3])
    )
    cv2.imshow("test", in_frame)
    cv2.waitKey(10)

process1.wait()
