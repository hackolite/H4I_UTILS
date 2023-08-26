#!/usr/bin/env python
# coding=utf-8

"""
filename: rtcp.py
@desc:
@usage:
./rtcp.py stream1 stream2
stream ：l:port  c:host:port
l:port
c:host:port 
@author: watercloud, zd, knownsec team
@web: www.knownsec.com, blog.knownsec.com
@date: 2009-7
"""

import socket
import sys
import threading
import time


streams = [None, None]  
debug = 1


def _usage():
    print('Usage: ./rtcp.py stream1 stream2\nstream: l:port  or c:host:port')


def _get_another_stream(num):
    if num == 0:
        num = 1
    elif num == 1:
        num = 0
    else:
        raise 'ERROR'

    while True:
        if streams[num] == 'quit':
            print('can not connect to the target, quit now!')
            sys.exit(1)

        if streams[num] is not None:
            return streams[num]
        else:
            time.sleep(1)


def _xstream(num, s1, s2):
    try:
        while True:
            buff = s1.recv(1024)
            if debug > 0:
                print('%d recv' % num)
            if len(buff) == 0: 
                print('%d one closed' % num)
                break
            s2.sendall(buff)
            if debug > 0:
                print('%d sendall' % num)
    except:
        print('%d one connect closed.' % num)

    try:
        s1.shutdown(socket.SHUT_RDWR)
        s1.close()
    except:
        pass

    try:
        s2.shutdown(socket.SHUT_RDWR)
        s2.close()
    except:
        pass

    streams[0] = None
    streams[1] = None
    print('%d CLOSED' % num)


def connect(host, port, num):
    not_connet_time = 0
    wait_time = 36
    try_cnt = 199
    while True:
        if not_connet_time > try_cnt:
            streams[num] = 'quit'
            print('not connected')
            return None

        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            conn.connect((host, port))
        except Exception:
            print('can not connect %s:%s!' % (host, port))
            not_connet_time += 1
            time.sleep(wait_time)
            continue

        print('connected to %s:%i' % (host, port))
        streams[num] = conn  
        s2 = _get_another_stream(num)  
        _xstream(num, conn, s2)



if __name__ == '__main__':
    connect('192.168.42.1', '55004', 4)
