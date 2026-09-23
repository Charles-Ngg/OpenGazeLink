#!/usr/bin/env python3
"""Cross-check the Swift wire format against the PC provider's own definitions.

The iOS app cannot be compiled without macOS, so the highest-risk failure mode
is silent drift between the Swift byte layout and the `struct.Struct` formats the
PC unpacks with. This script reads those formats out of the PC source, derives
the golden byte sequences, and asserts the Swift constants and unit-test vectors
still match them.

Runs anywhere with a stock Python 3; no third-party dependencies, so CI can run
it on Linux alongside the macOS build.

Usage:  python3 ios-app/Tools/check_wire_contract.py [repo-root]
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path


class CheckFailure(Exception):
    pass


def read(path: Path) -> str:
    if not path.exists():
        raise CheckFailure(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def find_struct(source: str, name: str) -> str:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*struct\.Struct\(\s*[\"']([^\"']+)[\"']\s*\)", source, re.M)
    if not match:
        raise CheckFailure(f"could not find struct definition {name}")
    return match.group(1)


def find_int(source: str, name: str) -> int:
    """Reads a Python integer constant, allowing a trailing comment and simple
    arithmetic such as `4 * 1024 * 1024`."""
    match = re.search(
        rf"^\s*{re.escape(name)}\s*(?::\s*[\w.\[\]]+)?\s*=\s*([^#\n]+?)\s*(?:#.*)?$", source, re.M
    )
    if not match:
        raise CheckFailure(f"could not find integer constant {name}")
    expression = match.group(1).strip().replace("_", "")
    try:
        return int(evaluate_constant_expression(expression))
    except (ValueError, SyntaxError) as error:
        raise CheckFailure(f"could not evaluate {name} = {expression!r}: {error}") from error


def evaluate_constant_expression(expression: str) -> int:
    """Evaluates a literal integer expression without using eval()."""
    import ast

    def walk(node: "ast.AST") -> int:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp):
            left = walk(node.left)
            right = walk(node.right)
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.LShift):
                return left << right
        raise ValueError(f"unsupported expression: {ast.dump(node)}")

    return walk(ast.parse(expression, mode="eval"))


def find_tuple(source: str, name: str) -> tuple[int, ...]:
    match = re.search(rf"{re.escape(name)}\s*!=\s*\(([^)]*)\)", source)
    if not match:
        raise CheckFailure(f"could not find the {name} comparison")
    return tuple(int(part.strip()) for part in match.group(1).split(",") if part.strip())


def swift_hex_arrays(source: str) -> list[tuple[int, ...]]:
    arrays = []
    for body in re.findall(r"let expected: \[UInt8\] = \[(.*?)\]", source, re.S):
        values = tuple(int(token, 16) for token in re.findall(r"0x([0-9A-Fa-f]{2})", body))
        if values:
            arrays.append(values)
    return arrays


def swift_constants(source: str) -> dict[str, int]:
    """Reads `static let name[: Type] = <integer expression>` out of Swift.

    Expressions that are not integer arithmetic (arrays, strings, function
    calls) are skipped rather than treated as an error.
    """
    constants: dict[str, int] = {}
    for name, raw in re.findall(
        r"static let (\w+)\s*(?::[^=\n]+)?=\s*([^\n]+)", source
    ):
        expression = raw.split("//")[0].strip().rstrip(";")
        if not expression or not re.fullmatch(r"[\dxXa-fA-F_*+\s()<>|]+-?\d*", expression):
            continue
        try:
            constants[name] = int(evaluate_constant_expression(expression.replace("_", "")))
        except (ValueError, SyntaxError, TypeError):
            continue
    return constants


def hex_bytes(values: tuple[int, ...]) -> str:
    return " ".join(f"{value:02X}" for value in values)


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    pc = root / "pc-provider" / "opengazelink_pc"
    swift_dir = root / "ios-app" / "OpenGazeLink"
    tests_dir = root / "ios-app" / "OpenGazeLinkTests"

    camera = read(pc / "camera.py")
    h264 = read(pc / "h264_stream.py")
    clock = read(pc / "transport_clock.py")
    wire = read(swift_dir / "Core" / "WireFormat.swift")
    wire_tests = read(tests_dir / "WireFormatTests.swift")
    option_tests = read(tests_dir / "CaptureOptionTests.swift")

    failures: list[str] = []
    checks = 0

    def expect(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)

    # ---- struct sizes -------------------------------------------------------
    frame_format = find_struct(camera, "HEADER")
    intrinsics_format = find_struct(camera, "INTRINSICS_HEADER")
    avc_format = find_struct(h264, "AVC_HEADER")
    clock_format = find_struct(clock, "CLOCK_PACKET")

    frame_size = struct.calcsize(frame_format)
    intrinsics_size = struct.calcsize(intrinsics_format)
    avc_size = struct.calcsize(avc_format)
    clock_size = struct.calcsize(clock_format)

    constants = swift_constants(wire)
    expect(constants.get("frameHeaderSize") == frame_size,
           f"frameHeaderSize is {constants.get('frameHeaderSize')}, PC {frame_format!r} needs {frame_size}")
    expect(constants.get("intrinsicsHeaderSize") == intrinsics_size,
           f"intrinsicsHeaderSize is {constants.get('intrinsicsHeaderSize')}, PC needs {intrinsics_size}")
    expect(constants.get("avcHeaderSize") == avc_size,
           f"avcHeaderSize is {constants.get('avcHeaderSize')}, PC {avc_format!r} needs {avc_size}")
    expect(constants.get("clockPacketSize") == clock_size,
           f"clockPacketSize is {constants.get('clockPacketSize')}, PC needs {clock_size}")

    # ---- magic numbers ------------------------------------------------------
    expect(constants.get("frameMagic") == find_int(camera, "MAGIC"),
           "frameMagic does not match camera.py MAGIC")
    expect(constants.get("intrinsicsMagic") == find_int(camera, "INTRINSICS_MAGIC"),
           "intrinsicsMagic does not match camera.py INTRINSICS_MAGIC")
    expect(constants.get("clockMagic") == find_int(clock, "CLOCK_MAGIC"),
           "clockMagic does not match transport_clock.py CLOCK_MAGIC")
    expect(constants.get("formatNV21") == find_int(camera, "FORMAT_NV21"),
           "formatNV21 does not match camera.py FORMAT_NV21")
    expect(constants.get("formatJPEG") == find_int(camera, "FORMAT_JPEG"),
           "formatJPEG does not match camera.py FORMAT_JPEG")

    avc_magic_match = re.search(r"avcMagic\s*(?::[^=\n]+)?=\s*Array\(\"(\w+)\"\.utf8\)", wire)
    expect(avc_magic_match is not None, "could not find WireFormat.avcMagic")
    if avc_magic_match:
        expect(avc_magic_match.group(1).encode() == b"AVC1",
               f"avcMagic is {avc_magic_match.group(1)!r}, PC expects b'AVC1'")

    expect(constants.get("maxAvcPacketBytes") == find_int(h264, "MAX_PACKET_BYTES"),
           "maxAvcPacketBytes does not match h264_stream.py MAX_PACKET_BYTES")

    # ---- Android sender parity ---------------------------------------------
    udp_sender = read(root / "phone-app" / "app" / "src" / "main" / "java" / "com" / "eyetracing" / "android" / "UdpYuvSender.kt")
    expect(constants.get("frameChunkPayloadBytes") == 1400,
           "frameChunkPayloadBytes must stay at the Android 1400 byte chunk payload")
    expect("private const val HEADER_SIZE = 42" in udp_sender,
           "Android UdpYuvSender HEADER_SIZE moved away from 42")
    expect("private const val VERSION = 1" in udp_sender,
           "Android UdpYuvSender VERSION moved away from 1")

    # ---- golden byte vectors ------------------------------------------------
    frame_golden = struct.pack(
        frame_format, find_int(camera, "MAGIC"), 1, frame_size, 1, 0, 1, 1280, 720,
        find_int(camera, "FORMAT_JPEG"), 0,
        0x0102030405060708, 0x1112131415161718, 3,
    )
    intrinsics_golden = struct.pack(intrinsics_format, find_int(camera, "INTRINSICS_MAGIC"), 1, intrinsics_size, 9)
    avc_golden = struct.pack(
        avc_format, b"AVC1", 7, 3, 1280, 720,
        0x0102030405060708, 0x1112131415161718, 0x2122232425262728, 5,
    )
    clock_golden = struct.pack(
        clock_format, find_int(clock, "CLOCK_MAGIC"), 1, 2,
        0x0102030405060708, 0x1112131415161718, 0x2122232425262728,
    )

    expected_vectors = {
        tuple(frame_golden),
        tuple(intrinsics_golden),
        tuple(avc_golden),
        tuple(clock_golden),
    }
    found_vectors = set(swift_hex_arrays(wire_tests))
    missing = expected_vectors - found_vectors
    expect(
        not missing,
        "WireFormatTests.swift is missing golden vectors for: "
        + ", ".join(hex_bytes(vector) for vector in sorted(missing)),
    )
    for vector in sorted(found_vectors & expected_vectors):
        print(f"  ok  {len(vector):3d} bytes  {hex_bytes(vector)}")

    # ---- H.264 geometry gate ------------------------------------------------
    pc_dimensions = find_tuple(h264, "(width, height)")
    expect(
        (constants.get("acceptedH264Width"), constants.get("acceptedH264Height")) == pc_dimensions,
        f"WireFormat accepts {constants.get('acceptedH264Width')}x{constants.get('acceptedH264Height')} "
        f"but h264_stream.py compares against {pc_dimensions}",
    )
    expect(
        f"XCTAssertTrue(WireFormat.acceptsH264(width: {pc_dimensions[0]}, height: {pc_dimensions[1]}))" in wire_tests,
        "WireFormatTests.swift does not assert the accepted H.264 geometry",
    )

    # ---- UDP formats --------------------------------------------------------
    supported = re.search(r"SUPPORTED_FRAME_FORMATS\s*=\s*\{([^}]*)\}", camera)
    expect(supported is not None, "could not find SUPPORTED_FRAME_FORMATS")
    if supported:
        # The set is written with symbolic names, e.g. {FORMAT_NV21, FORMAT_JPEG}.
        pc_formats = {
            find_int(camera, part.strip())
            for part in supported.group(1).split(",")
            if part.strip()
        }
        # The iOS sender is JPEG-only on purpose: iOS delivers biplanar NV12
        # (UV order) which is not byte-compatible with the PC's NV21 reader.
        expect(find_int(camera, "FORMAT_JPEG") in pc_formats,
               "the PC no longer accepts JPEG on the UDP path")
        expect(find_int(camera, "FORMAT_NV21") in pc_formats,
               "the PC no longer accepts NV21; the JPEG-only decision should be revisited")
        expect("supportedUDPFormats: Set<UInt8> = [formatJPEG]" in wire,
               "WireFormat.supportedUDPFormats must stay JPEG-only on iOS")

    # ---- discovery ----------------------------------------------------------
    expect("DISCOVERY_MAGIC = \"EYETRACING_DISCOVERY_V1\"" in read(pc / "pairing.py"),
           "pairing.py discovery magic changed")
    expect("static let discoveryMagic = \"EYETRACING_DISCOVERY_V1\"" in wire,
           "WireFormat.discoveryMagic changed")
    expect(constants.get("discoveryVersion") == 1, "discovery version must stay 1")
    expect(constants.get("discoveryPort") == find_int(read(pc / "config.py"), "discovery_port"),
           "WireFormat.discoveryPort does not match config.py discovery_port")
    expect(constants.get("defaultDataPort") == find_int(read(pc / "config.py"), "udp_port"),
           "WireFormat.defaultDataPort does not match config.py udp_port")

    # ---- capture options contract ------------------------------------------
    expect("HIGH_SPEED" in read(swift_dir / "Model" / "CaptureOption.swift"),
           "Swift SessionMode raw values must keep the Android spelling")
    expect(constants.get("acceptedH264Width") == 1280,
           "WireFormat.acceptedH264Width must stay 1280")
    expect("func acceptsH264" in wire, "WireFormat.acceptsH264 is missing")
    expect("isPCH264Compatible" in option_tests,
           "CaptureOptionTests must keep asserting the PC geometry gate")

    print()
    if failures:
        print(f"FAILED ({len(failures)} of {checks} checks)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"OK: {checks} wire-contract checks passed against pc-provider")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailure as error:
        print(f"FAILED: {error}")
        sys.exit(1)
