"""Length-framed AVC over TCP. Decode every reference; publish only the latest image."""
from __future__ import annotations

from fractions import Fraction
from collections import deque
import socket
import struct
import threading
import logging

from . import runtime_clock

AVC_HEADER = struct.Struct('<4sIIHHQQQI')
MAX_PACKET_BYTES = 4 * 1024 * 1024
LOGGER = logging.getLogger("eyetracing.h264")


def receive_exact(sock, length, stop):
    result = bytearray()
    started = runtime_clock.monotonic()
    while len(result) < length:
        if stop.is_set():
            raise EOFError('receiver stopped')
        try:
            data = sock.recv(length - len(result))
        except socket.timeout:
            if runtime_clock.monotonic() - started > 2:
                raise TimeoutError('H.264 receive stalled')
            continue
        if not data:
            raise EOFError('H.264 phone disconnected')
        result.extend(data)
    return bytes(result)


class AvcDecoder:
    def __init__(self):
        import av
        self.av = av
        self.codec = av.CodecContext.create('h264', 'r')
        self.codec.thread_count = 1
        self.codec.thread_type = 'SLICE'
        self.config = b''

    def decode(self, data, flags, sensor_ns):
        # Offline replay consumes every image. Live reception converts images
        # on a separate latest-only worker, never blocking reference decoding.
        return [frame.to_ndarray(format='bgr24')
                for frame in self.decode_native(data, flags, sensor_ns)]

    def decode_native(self, data, flags, sensor_ns):
        if flags & 2:  # Android BUFFER_FLAG_CODEC_CONFIG: SPS/PPS, not a frame.
            self.config = data
            return []
        if flags & 8:
            raise ValueError('Partial H.264 access units are unsupported')
        packet = self.av.Packet((self.config if flags & 1 else b'') + data)
        packet.pts = packet.dts = sensor_ns
        packet.time_base = Fraction(1, 1_000_000_000)
        frames = self.codec.decode(packet)
        # No B frames and no frame threading: do not silently misattribute timing.
        if len(frames) != 1 or frames[0].pts != sensor_ns:
            raise ValueError('H.264 decoder delayed/reordered a frame; timing unavailable')
        return frames


class _LatestImageWorker:
    """One in-flight image and one replaceable pending native AVFrame.

    AVFrames retain their own FFmpeg buffer references after decoder advances.
    Only decoded images may be replaced; compressed reference packets may not.
    """
    def __init__(self, publish):
        self.publish = publish
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.error = None
        self.thread = threading.Thread(target=self._run, name='h264-image-publish', daemon=True)
        self.thread.start()

    def submit(self, item):
        with self.condition:
            if self.error is not None:
                raise self.error
            replaced = self.pending is not None
            self.pending = item
            self.condition.notify()
            return replaced

    def _run(self):
        from .runtime_scheduling import set_realtime_thread_priority
        set_realtime_thread_priority()
        try:
            while True:
                with self.condition:
                    while self.pending is None and not self.closed:
                        self.condition.wait()
                    if self.pending is None:
                        return
                    item, self.pending = self.pending, None
                self.publish(*item)
        except Exception as error:
            with self.condition:
                self.error = error
                self.pending = None

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        # Finish before accepting another connection: an old frame must never
        # overwrite the new stream. No camera/queue lock is held while joining.
        self.thread.join()


class H264TcpReceiver:
    DECODE_QUEUE_FRAMES = 12
    def __init__(self, camera):
        self.camera = camera
        self.stop = threading.Event()
        self.connection = None
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = camera._socket.getsockname()[1]
            try:
                self.listener.bind((camera.config.bind, port))
            except OSError:
                # Unit/in-process callers commonly allocate an ephemeral UDP
                # port while another test process owns the same TCP number.
                # Production uses a fixed configured port and must surface the
                # bind failure instead of silently changing the phone target.
                if int(getattr(camera.config, "port", port) or 0) != 0:
                    raise
                self.listener.bind((camera.config.bind, 0))
            self.listener.listen(1)
            self.listener.settimeout(.2)
        except Exception:
            self.listener.close()
            raise
        self.thread = threading.Thread(target=self.run, name='h264-tcp-decode', daemon=True)
        self.thread.start()

    def run(self):
        from .runtime_scheduling import set_realtime_thread_priority
        set_realtime_thread_priority()
        while not self.stop.is_set():
            try:
                connection, address = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self.connection = connection
            with connection:
                if self.camera._allowed_source_ip and address[0] != self.camera._allowed_source_ip:
                    LOGGER.warning("H.264 source rejected: peer=%s allowed=%s; waiting for paired discovery",
                                   address[0], self.camera._allowed_source_ip)
                    continue
                try:
                    connection.settimeout(.1)
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
                    with self.camera._lock:
                        self.camera._sensor_samples.clear()
                        self.camera._times.clear()
                        self.camera._last_source_ip = address[0]
                    self._serve(connection)
                except Exception as error:
                    if not self.stop.is_set():
                        if isinstance(error, EOFError):
                            LOGGER.info("H.264 connection ended: %s", error)
                        else:
                            LOGGER.warning("H.264 connection ended: %s", error, exc_info=True)
                        with self.camera._frame_condition:
                            self.camera._last_error = str(error)
                            self.camera._frame_condition.notify_all()
            self.connection = None

    def _serve(self, connection):
        """Read the wire independently from decode/model CPU load.

        Decode references in order; convert/rotate only the latest image on a
        separate worker. If reference decoding itself falls behind, discard
        queued dependent frames and resume at the next keyframe as a last resort.
        """
        from .camera import FrameAssembly, FORMAT_H264
        items = deque()
        condition = threading.Condition()
        reader_done = False
        reader_error = None
        waiting_for_keyframe = False
        codec_config = b''

        def read_wire():
            nonlocal reader_done, reader_error, waiting_for_keyframe, codec_config
            from .runtime_scheduling import set_realtime_thread_priority
            set_realtime_thread_priority()
            last_seq = -1
            try:
                while not self.stop.is_set():
                    raw = receive_exact(connection, AVC_HEADER.size, self.stop)
                    first = runtime_clock.monotonic()
                    magic, seq, flags, width, height, sensor, encoded, sent, size = AVC_HEADER.unpack(raw)
                    if magic != b'AVC1' or not (0 < size <= MAX_PACKET_BYTES) or (width, height) != (1280, 720):
                        raise ValueError('Invalid H.264 packet header')
                    data = receive_exact(connection, size, self.stop)
                    completed = runtime_clock.monotonic()
                    metadata = {
                        "pc_receive_ms": completed * 1000.0, "sequence": seq,
                        "flags": flags, "width": width, "height": height,
                        "sensor_time_ns": sensor, "encoded_time_ns": encoded,
                        "phone_send_time_ns": sent,
                    }
                    if flags & 2:
                        self.camera._h264_codec_config_archive = (
                            "h264_access_unit", metadata, raw + data,
                        )
                    archive = getattr(self.camera, "video_archive", None)
                    if archive is not None:
                        archive.submit("h264_access_unit", metadata, raw + data)
                    if seq <= last_seq:
                        LOGGER.warning("stale H.264 sequence: previous=%s got=%s", last_seq, seq)
                        continue
                    if last_seq >= 0 and seq != last_seq + 1:
                        LOGGER.warning("H.264 sequence gap: expected=%s got=%s", last_seq + 1, seq)
                    last_seq = seq
                    if flags & 2:
                        codec_config = data
                    item = (seq, flags, width, height, sensor, encoded, sent,
                            first, completed, data, codec_config)
                    with condition:
                        reset_decoder = False
                        if len(items) >= self.DECODE_QUEUE_FRAMES:
                            dropped = len(items)
                            items.clear()
                            waiting_for_keyframe = True
                            self.camera._decode_drops += dropped
                            LOGGER.warning("H.264 decode queue overloaded; dropped=%s, awaiting keyframe", dropped)
                        if waiting_for_keyframe and not (flags & 1 or flags & 2):
                            self.camera._decode_drops += 1
                            continue
                        if flags & 1:
                            reset_decoder = waiting_for_keyframe
                            waiting_for_keyframe = False
                        items.append((reset_decoder, item))
                        condition.notify()
            except Exception as error:
                reader_error = error
            finally:
                with condition:
                    reader_done = True
                    condition.notify_all()

        reader = threading.Thread(target=read_wire, name='h264-tcp-receive', daemon=True)
        reader.start()
        def publish_image(native_frame, item, decode_started_at, decoded_at):
            if self.stop.is_set():
                return
            seq, flags, width, height, sensor, encoded, sent, first, completed, data, config = item
            conversion_started_at = runtime_clock.monotonic()
            if (native_frame.width, native_frame.height) != (width, height):
                raise ValueError('H.264 decoded dimensions do not match header')
            frame = self.camera._preprocess(native_frame.to_ndarray(format='bgr24'))
            now = runtime_clock.monotonic()
            timing = self.camera._transport_clock.latest(int(first * 1e9))
            if 'phone_to_pc_offset_ns' in timing:
                minimum_age_ms = (first*1e9-sent-timing['phone_to_pc_offset_ns'])/1e6 - timing['clock_probe_uncertainty_ms']
                if minimum_age_ms > 100:
                    LOGGER.warning("H.264 transport backlog estimate %.1fms", minimum_age_ms)
            timing['phone_encoded_time_ns'] = encoded
            timing['h264_reference_decode_ms'] = (decoded_at - decode_started_at) * 1000
            timing['h264_image_queue_ms'] = (conversion_started_at - decoded_at) * 1000
            timing['h264_image_conversion_ms'] = (now - conversion_started_at) * 1000
            assembly = FrameAssembly(seq, width, height, FORMAT_H264, 1, sensor, sent,
                                     first, {}, completed, timing)
            with self.camera._frame_condition:
                self.camera._phone_pipeline_ms = (sent - sensor) / 1e6
                self.camera._publish_decoded(
                    assembly, frame, decode_started_at,
                    (now - decode_started_at) * 1000, now,
                )

        images = _LatestImageWorker(publish_image)
        try:
            decoder = AvcDecoder()
            while not self.stop.is_set():
                with condition:
                    while not items and not reader_done and not self.stop.is_set() and images.error is None:
                        condition.wait(.1)
                    if images.error is not None:
                        raise images.error
                    if not items:
                        break
                    reset_decoder, item = items.popleft()
                seq, flags, width, height, sensor, encoded, sent, first, completed, data, config = item
                if reset_decoder:
                    decoder = AvcDecoder()
                    decoder.config = config
                decode_started_at = runtime_clock.monotonic()
                try:
                    frames = decoder.decode_native(data, flags, sensor)
                except Exception as error:
                    LOGGER.warning("H.264 decode lost synchronization: %s; awaiting keyframe", error)
                    with condition:
                        dropped = len(items)
                        items.clear()
                        waiting_for_keyframe = True
                        self.camera._decode_drops += dropped + 1
                    continue
                if not frames:
                    continue
                decoded_at = runtime_clock.monotonic()
                self.camera._h264_reference_frames += 1
                if images.submit((frames[0], item, decode_started_at, decoded_at)):
                    self.camera._h264_skipped_images += 1
        finally:
            # Also release the reader on decoder/conversion failures, not just
            # on EOF. Otherwise a previous reader can outlive the connection.
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            reader.join(timeout=1)
            images.close()
        if images.error is not None:
            raise images.error
        if reader_error is not None:
            raise reader_error

    def close(self):
        self.stop.set()
        self.listener.close()
        if self.connection:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
        self.thread.join(timeout=2)
