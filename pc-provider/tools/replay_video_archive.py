"""Recover every complete raw camera frame, including those skipped by inference.

Writes lossless PNGs and timing/reassembly diagnostics to a NEW directory. The
original data.bin/index.jsonl are never changed. UDP payloads retain their source
JPEG/NV21 representation in the archive; decoded PNGs are optional derivatives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from opengazelink_pc.camera import HEADER, MAGIC, INTRINSICS_MAGIC, FrameAssembly, decode_udp_frame


def recover(archive, output):
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True,exist_ok=False)
    assemblies, recovered = {}, 0
    with (archive/"data.bin").open("rb") as data, (output/"frames.jsonl").open("w",encoding="utf-8") as frames, (output/"recovery.jsonl").open("w",encoding="utf-8") as log:
        for line in (archive/"index.jsonl").read_text(encoding="utf-8").splitlines():
            record=json.loads(line)
            data.seek(record["offset"])
            payload=data.read(record["length"])
            try:
                if len(payload)!=record["length"]:
                    raise ValueError("truncated raw payload")
                if record["kind"] in ("windows_bgr","processed_bgr"):
                    image=np.frombuffer(payload,dtype=record["dtype"]).reshape(record["shape"])
                    metadata=record
                elif record["kind"]=="udp_packet":
                    if len(payload)<HEADER.size or int.from_bytes(payload[:4],"little")!=MAGIC:
                        log.write(json.dumps(dict(record,reason="non_frame_packet_retained"))+"\n")
                        continue
                    magic,version,header_size,seq,chunk,count,width,height,fmt,flags,sensor,send,size=HEADER.unpack_from(payload)
                    if version!=1 or header_size!=HEADER.size or not 0<=chunk<count or len(payload)<header_size+size:
                        raise ValueError("invalid frame packet")
                    # Sensor timestamp distinguishes phone sequence restarts.
                    key=(seq,sensor,send)
                    if key not in assemblies:
                        assemblies[key]=FrameAssembly(seq,width,height,fmt,count,sensor,send,record["pc_receive_ms"]/1000,{})
                    item=assemblies[key]
                    if (width,height,fmt,count)!=(item.width,item.height,item.fmt,item.chunk_count):
                        raise ValueError("inconsistent frame packet")
                    item.chunks[chunk]=payload[header_size:header_size+size]
                    if not item.complete:
                        continue
                    image=decode_udp_frame(item.payload(),width,height,fmt)
                    metadata={"phone_frame_sequence":seq,"phone_sensor_time_ns":sensor,"phone_send_time_ns":send,
                              "pc_first_packet_ms":item.started_at*1000,"format":fmt,"rotation_applied":False}
                    del assemblies[key]
                else:
                    raise ValueError("unknown raw record kind")
                name=f"{recovered:07d}.png"
                ok,encoded=cv2.imencode(".png",image,[cv2.IMWRITE_PNG_COMPRESSION,1])
                if not ok:
                    raise ValueError("PNG encoding failed")
                (output/name).write_bytes(encoded.tobytes())
                frames.write(json.dumps(dict(metadata,image=name))+"\n")
                recovered+=1
            except Exception as error:
                log.write(json.dumps(dict(record,reason=str(error)))+"\n")
        for key,item in assemblies.items():
            log.write(json.dumps({"key":key,"reason":"incomplete_packet_assembly","received_chunks":sorted(item.chunks),"expected_chunks":item.chunk_count})+"\n")
    return {"recovered_frames":recovered,"incomplete_frames":len(assemblies),"output":str(output.resolve())}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    print(json.dumps(recover(args.archive,args.output),ensure_ascii=False))
