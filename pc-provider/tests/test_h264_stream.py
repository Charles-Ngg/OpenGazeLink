from fractions import Fraction
import socket
import threading
import time
import unittest
from unittest.mock import patch

import av
import numpy as np

from opengazelink_pc.camera import UdpYuvCamera, UdpYuvConfig
from opengazelink_pc.h264_stream import AVC_HEADER, AvcDecoder
from opengazelink_pc import runtime_clock


def encoded_frames(count=12):
    codec = av.CodecContext.create('libx264', 'w')
    codec.width, codec.height = 1280, 720
    codec.pix_fmt = 'yuv420p'
    codec.time_base = Fraction(1, 120)
    codec.framerate = Fraction(120, 1)
    codec.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'profile': 'baseline', 'crf': '22'}
    packets = []
    for i in range(count):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[:, :, 1] = 30 + i * 3 % 180
        image[100:400, 20+i*4:180+i*4, 2] = 240
        frame = av.VideoFrame.from_ndarray(image, format='bgr24')
        frame.pts = i
        packets.extend(codec.encode(frame))
    packets.extend(codec.encode(None))
    return packets


class H264StreamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packets = encoded_frames()

    def setUp(self):
        self.camera = UdpYuvCamera(UdpYuvConfig(bind='127.0.0.1', port=0, rotate=0, intrinsics_cache_path=''))
        self.port = self.camera._h264_receiver.listener.getsockname()[1]

    def tearDown(self):
        self.camera.release()

    def send(self, sock, seq, packet, fragmented=False):
        sensor = 1_000_000_000 + seq * 8_333_333
        data = bytes(packet)
        payload = AVC_HEADER.pack(b'AVC1', seq, 1 if packet.is_keyframe else 0, 1280, 720,
                                  sensor, sensor+40_000_000, sensor+41_000_000, len(data)) + data
        if fragmented:
            sock.sendall(payload[:7]); sock.sendall(payload[7:39]); sock.sendall(payload[39:])
        else:
            sock.sendall(payload)

    def wait_frames(self, count):
        deadline = time.perf_counter() + 3
        while self.camera._latest_seq < count and time.perf_counter() < deadline:
            time.sleep(.005)
        self.assertGreaterEqual(self.camera._latest_seq, count, self.camera.reported_mode())

    def wait_for(self, predicate):
        deadline = time.perf_counter() + 3
        while not predicate() and time.perf_counter() < deadline:
            time.sleep(.002)
        self.assertTrue(predicate(), self.camera.reported_mode())

    def test_reference_frames_fragmented_transport_and_latest_only_read(self):
        with socket.create_connection(('127.0.0.1', self.port)) as sock:
            for i, packet in enumerate(self.packets):
                self.send(sock, i, packet, fragmented=True)
            self.wait_for(lambda: self.camera.latest_frame_timing().get('phone_frame_sequence') == len(self.packets)-1)
            ok, image, timestamp, seq = self.camera.read_latest(timeout_s=.1)
            self.assertTrue(ok)
            self.assertEqual(image.shape, (720, 1280, 3))
            self.assertLessEqual(seq, len(self.packets))
            self.assertEqual(self.camera._h264_reference_frames, len(self.packets))
            self.assertEqual(self.camera._decode_drops, 0)
            self.assertAlmostEqual(timestamp, (1_000_000_000 + 11*8_333_333)/1e6)
            self.assertEqual(self.camera.inference_max_fps, 0)
            self.assertEqual(self.camera.reported_mode()['fourcc'], 'H264')
            timing = self.camera.latest_frame_timing(seq)
            self.assertEqual(timing['phone_frame_sequence'], 11)
            self.assertLessEqual(timing['pc_last_packet_monotonic_ns'], timing['pc_decode_done_monotonic_ns'])
            self.assertFalse(self.camera.read_latest(seq, timeout_s=.02)[0])

    def test_blocked_image_conversion_does_not_drop_reference_frames(self):
        packets = encoded_frames(40)
        entered, resume = threading.Event(), threading.Event()
        preprocess = self.camera._preprocess
        def blocked(frame):
            entered.set()
            if not resume.wait(3):
                raise RuntimeError('test conversion gate timed out')
            return preprocess(frame)
        self.camera._preprocess = blocked
        try:
            with socket.create_connection(('127.0.0.1', self.port)) as sock:
                self.send(sock, 0, packets[0])
                self.assertTrue(entered.wait(3))
                for seq, packet in enumerate(packets[1:], 1):
                    self.send(sock, seq, packet)
                    # Deterministic conversion-only overload, not a wire burst.
                    self.wait_for(lambda: self.camera._h264_reference_frames == seq+1)
                self.assertEqual(self.camera._decode_drops, 0)
                self.assertEqual(self.camera._h264_skipped_images, 38)
                self.assertEqual(self.camera._latest_seq, 0)
                resume.set()
                self.wait_for(lambda: self.camera.latest_frame_timing().get('phone_frame_sequence') == 39)
                self.assertEqual(self.camera._latest_seq, 2)
                timing = self.camera.latest_frame_timing()
                self.assertEqual(timing['phone_sensor_time_ns'], 1_000_000_000 + 39*8_333_333)
                self.assertAlmostEqual(timing['decode_ms'], sum(timing[key] for key in (
                    'h264_reference_decode_ms', 'h264_image_queue_ms', 'h264_image_conversion_ms')), places=5)
                expected = AvcDecoder()
                for seq, packet in enumerate(packets):
                    last = expected.decode(bytes(packet), int(packet.is_keyframe), 1_000_000_000 + seq*8_333_333)[0]
                np.testing.assert_array_equal(self.camera.read_latest()[1], last)
        finally:
            resume.set()

    def test_conversion_failure_closes_connection_and_reconnects(self):
        preprocess = self.camera._preprocess
        def broken(frame):
            raise ValueError('test conversion failure')
        self.camera._preprocess = broken
        with socket.create_connection(('127.0.0.1', self.port)) as sock:
            sock.settimeout(3)
            self.send(sock, 0, self.packets[0])
            self.assertEqual(sock.recv(1), b'')
        self.wait_for(lambda: 'test conversion failure' in self.camera._last_error)
        self.camera._preprocess = preprocess
        with socket.create_connection(('127.0.0.1', self.port)) as sock:
            self.send(sock, 0, self.packets[0])
            self.wait_frames(1)
            self.assertEqual(self.camera._last_error, '')

    def test_reference_decode_overload_still_recovers_at_keyframe(self):
        entered, resume = threading.Event(), threading.Event()
        decode = AvcDecoder.decode_native
        def blocked(decoder, data, flags, sensor):
            if not entered.is_set():
                entered.set()
                if not resume.wait(3):
                    raise RuntimeError('test decode gate timed out')
            return decode(decoder, data, flags, sensor)
        try:
            with patch.object(AvcDecoder, 'decode_native', blocked):
                with socket.create_connection(('127.0.0.1', self.port)) as sock:
                    self.send(sock, 0, self.packets[0])
                    self.assertTrue(entered.wait(3))
                    for seq in range(1, 16):
                        self.send(sock, seq, self.packets[1])
                    self.wait_for(lambda: self.camera._decode_drops >= 15)
                    self.send(sock, 16, self.packets[0])
                    self.send(sock, 17, self.packets[1])
                    resume.set()
                    self.wait_for(lambda: self.camera.latest_frame_timing().get('phone_frame_sequence') == 17)
                    self.assertEqual(self.camera._decode_drops, 15)
                    self.assertEqual(self.camera._h264_reference_frames, 3)
        finally:
            resume.set()

    def test_eof_drains_latest_pending_image_before_reconnect(self):
        with socket.create_connection(('127.0.0.1', self.port)) as sock:
            for seq, packet in enumerate(self.packets):
                self.send(sock, seq, packet)
            sock.shutdown(socket.SHUT_WR)
            self.wait_for(lambda: self.camera.latest_frame_timing().get('phone_frame_sequence') == 11)
        prior = self.camera._latest_seq
        with socket.create_connection(('127.0.0.1', self.port)) as sock:
            self.send(sock, 0, self.packets[0])
            self.wait_frames(prior+1)
            self.assertEqual(self.camera.latest_frame_timing()['phone_frame_sequence'], 0)

    def test_release_stops_all_connection_workers(self):
        with socket.create_connection(('127.0.0.1', self.port)) as sock:
            self.send(sock, 0, self.packets[0])
            self.wait_frames(1)
            workers = [thread for thread in threading.enumerate() if thread.name in (
                'h264-tcp-decode', 'h264-tcp-receive', 'h264-image-publish')]
            self.assertEqual(len(workers), 3)
            self.camera.release()
            self.assertTrue(all(not thread.is_alive() for thread in workers))

    def test_reconnect_recreates_decoder_and_accepts_sequence_zero(self):
        for expected in (1, 2):
            with socket.create_connection(('127.0.0.1', self.port)) as sock:
                self.send(sock, 0, self.packets[0])
                self.wait_frames(expected)

    def test_bad_header_disconnects_and_next_connection_recovers(self):
        with socket.create_connection(('127.0.0.1', self.port)) as sock:
            sock.settimeout(2)
            sock.sendall(AVC_HEADER.pack(b'AVC1', 0, 0, 1280, 720, 1, 2, 3, 9_000_000))
            self.assertEqual(sock.recv(1), b'')
        with socket.create_connection(('127.0.0.1', self.port)) as sock:
            self.send(sock, 0, self.packets[0]); self.wait_frames(1)

    def test_codec_config_is_not_published_as_frame(self):
        decoder = AvcDecoder()
        self.assertEqual(decoder.decode(b'', 2, 0), [])
        frames = decoder.decode(bytes(self.packets[0]), 1, 1_000_000_000)
        self.assertEqual(len(frames), 1)

    def test_clock_probes_continue_when_only_metadata_arrives(self):
        from opengazelink_pc.camera import INTRINSICS_HEADER, INTRINSICS_MAGIC
        # A non-frame datagram establishes the clock return address even without YUV traffic.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(2)
            sock.sendto(INTRINSICS_HEADER.pack(INTRINSICS_MAGIC, 0, 12, 0), ('127.0.0.1', self.port))
            first, _ = sock.recvfrom(64)
            second, _ = sock.recvfrom(64)
            self.assertEqual(len(first), 32)
            self.assertEqual(len(second), 32)
            self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
