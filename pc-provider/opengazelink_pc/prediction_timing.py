"""Causal frame-age estimates with explicit clock uncertainty."""
import math


class PredictionClock:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_source=None
        self.offset=None
        self.epoch=0

    def observe(self,source_ms,timing,now_ms,display_delay_ms=16.):
        if self.last_source is not None and source_ms<=self.last_source:
            self.offset=None
            self.epoch+=1
        self.last_source=source_ms
        sensor=timing.get("phone_sensor_time_ns",0)/1e6
        send=timing.get("phone_send_time_ns",0)/1e6
        receive=timing.get("pc_first_packet_monotonic_ns",0)/1e6
        capture=timing.get("source_capture_monotonic_ns",0)/1e6
        source_pc=None
        phone_pipeline=None
        transport_excess=None
        transport_to_first=None
        clock_details={}
        basis="unavailable"
        if capture>0:
            source_pc=capture
            basis="camera_read_completion_proxy"
        elif sensor>0 and send>=sensor and receive>0:
            offset=receive-send
            self.offset=offset if self.offset is None else min(self.offset,offset)
            source_pc=sensor+self.offset
            phone_pipeline=send-sensor
            transport_excess=max(0.,receive-(send+self.offset))
            transport_to_first=transport_excess
            basis="phone_minimum_transit_proxy"
            if timing.get("phone_to_pc_offset_ns") is not None:
                offset=timing["phone_to_pc_offset_ns"]/1e6
                # A midpoint assumes symmetric paths. Preserve the signed
                # estimate and uncertainty even when the display clamps at zero.
                signed_transit=receive-send-offset
                transport_to_first=max(0.,signed_transit)
                source_pc=receive-(send-sensor)-transport_to_first
                basis="phone_roundtrip_alignment"
                clock_details={key:timing.get(key) for key in (
                    "clock_probe_rtt_ms","clock_probe_uncertainty_ms","clock_probe_age_ms",
                    "clock_probe_samples","phone_to_pc_offset_ns")}
                clock_details["transport_signed_estimate_ms"]=signed_transit
        age=now_ms-source_pc if source_pc is not None else None
        valid=age is not None and math.isfinite(age) and 0<=age<=2000
        return {"schema":"opengazelink-frame-age-v1","clock_epoch":self.epoch,"clock_basis":basis,
            "source_pc_ms_proxy":source_pc,"pc_predict_ms":now_ms,"frame_age_ms_proxy":age if valid else None,
            "display_delay_ms_assumed":display_delay_ms,"horizon_ms_proxy":age+display_delay_ms if valid else None,
            "phone_capture_to_send_ms":phone_pipeline,"transport_excess_ms_proxy":transport_excess,
            "transport_to_first_packet_ms":transport_to_first,**clock_details,
            "unknown_transport_floor":basis=="phone_minimum_transit_proxy",
            "is_sensor_to_photon_measurement":False}
