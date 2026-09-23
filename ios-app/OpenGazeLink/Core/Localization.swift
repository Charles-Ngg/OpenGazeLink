import Foundation

/// In-app language switch, mirroring the Android app's bundled en / zh-rCN pair.
///
/// The table lives in Swift rather than in `.lproj/Localizable.strings` on
/// purpose: this project is authored on Linux with no Xcode available, so a
/// silently unwired variant group would fail closed (keys shown instead of
/// text) with no way to notice. A dictionary cannot be misconfigured. See
/// ios-app/README.md for the migration path to standard localization.
enum AppLanguage: String, CaseIterable, Identifiable {
    case english = "en"
    case simplifiedChinese = "zh-Hans"
    case traditionalChinese = "zh-Hant"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .english: return "English"
        case .simplifiedChinese: return "简体中文"
        case .traditionalChinese: return "繁體中文"
        }
    }
}

/// Localized text lookup.
struct L10n {
    enum Key: String {
        // Connection
        case connectionTitle
        case hostLabel
        case hostHint
        case portLabel
        case findPC
        case connectionIdle
        case discoveryActive
        case discoveryFailed
        case pcFound
        case paired
        case pairingChanged
        case changePairedPC
        case manualAddressHint
        case discoveryEntitlementHint
        case sweepProgress
        // Capture
        case captureTitle
        case modeHighSpeed
        case modeCamera
        case highSpeedHelp
        case cameraHelp
        case cameraLabel
        case frontCamera
        case backCamera
        case resolutionLabel
        case frameRateLabel
        case formatsHelp
        case noHighSpeed
        case noCameraOptions
        case noCombinations
        case noCameras
        case readingCameras
        case refreshFormats
        case h264PCUnsupported
        case h264PCSupportedNote
        // Orientation and crop
        case orientationTitle
        case rotationLabel
        case rotationAuto
        case cropLabel
        case cropLeft
        case cropRight
        case cropTop
        case cropBottom
        case cropHelp
        case cropInvalid
        case cropOutput
        case jpegQualityLabel
        case jpegQualityHelp
        // Intrinsics
        case intrinsicsTitle
        case intrinsicsSource
        case intrinsicsFovReference
        case intrinsicsFovHorizontal
        case intrinsicsFovDiagonal
        case intrinsicsLimitation
        // Streaming
        case startStream
        case stopStream
        case ready
        case opening
        case streaming
        case stopped
        case hostRequired
        case portInvalid
        case permissionDenied
        case cameraReadFailed
        case combinationFailed
        case connectionFailed
        case cameraFailed
        case intrinsicsFailed
        case streamHint
        // Performance
        case performanceTitle
        case performanceDetails
        case performanceSummary
        case captureFPS
        case sentFPS
        case videoRate
        case encodeAge
        case exposure
        case performanceNote
        case notAvailable
        // General
        case languageLabel
        case dismiss
    }

    let language: AppLanguage

    func text(_ key: Key) -> String {
        if let value = L10n.table[key], let localized = value[language] {
            return localized
        }
        return L10n.table[key]?[.english] ?? key.rawValue
    }

    /// Formats with positional arguments, matching the Android `%1$s` style.
    func text(_ key: Key, _ arguments: CVarArg...) -> String {
        // `String(format:arguments:)` rather than the variadic initialiser, so
        // forwarding the already-collected array is unambiguous.
        String(format: text(key), arguments: arguments)
    }

    private static let table: [Key: [AppLanguage: String]] = [
        .connectionTitle: [
            .english: "1 · Connect to PC",
            .simplifiedChinese: "1 · 连接电脑",
            .traditionalChinese: "1 · 連接電腦",
        ],
        .hostLabel: [
            .english: "PC address",
            .simplifiedChinese: "电脑地址",
            .traditionalChinese: "電腦地址",
        ],
        .hostHint: [
            .english: "IP address or host name",
            .simplifiedChinese: "IP 地址或主机名",
            .traditionalChinese: "IP 位址或主機名",
        ],
        .portLabel: [
            .english: "PC port (UDP / TCP)",
            .simplifiedChinese: "电脑端口（UDP / TCP）",
            .traditionalChinese: "電腦埠（UDP / TCP）",
        ],
        .findPC: [
            .english: "Find PC",
            .simplifiedChinese: "查找电脑",
            .traditionalChinese: "尋找電腦",
        ],
        .connectionIdle: [
            .english: "Use the same network or USB tethering. Confirm pairing on the PC.",
            .simplifiedChinese: "使用同一网络或 USB 共享网络。请在电脑端确认配对。",
            .traditionalChinese: "使用同一網絡或 USB 共享網絡。請在電腦端確認配對。",
        ],
        .discoveryActive: [
            .english: "Looking for a PC. You can also enter its address manually.",
            .simplifiedChinese: "正在查找电脑。也可以手动输入地址。",
            .traditionalChinese: "正在尋找電腦。也可以手動輸入位址。",
        ],
        .discoveryFailed: [
            .english: "Could not find a PC. Check the network or enter its address.",
            .simplifiedChinese: "未能找到电脑。请检查网络或手动输入地址。",
            .traditionalChinese: "未能找到電腦。請檢查網絡或手動輸入位址。",
        ],
        .pcFound: [
            .english: "Found %1$@ (%2$@). Confirm this phone on the PC.",
            .simplifiedChinese: "已找到 %1$@（%2$@）。请在电脑端确认此手机。",
            .traditionalChinese: "已找到 %1$@（%2$@）。請在電腦端確認此手機。",
        ],
        .paired: [
            .english: "Paired with %1$@ · %2$@:%3$d",
            .simplifiedChinese: "已与 %1$@ 配对 · %2$@:%3$d",
            .traditionalChinese: "已與 %1$@ 配對 · %2$@:%3$d",
        ],
        .pairingChanged: [
            .english: "PC address changed. Stop the current stream to reconnect.",
            .simplifiedChinese: "电脑地址已变化。请停止串流后重新连接。",
            .traditionalChinese: "電腦位址已變更。請停止串流後重新連接。",
        ],
        .changePairedPC: [
            .english: "Change paired PC",
            .simplifiedChinese: "更换配对电脑",
            .traditionalChinese: "更換配對電腦",
        ],
        .manualAddressHint: [
            .english: "If discovery finds nothing, type the PC's LAN address above and start streaming.",
            .simplifiedChinese: "若自动查找无结果，请在上方填入电脑的局域网地址后开始串流。",
            .traditionalChinese: "若自動尋找無結果，請在上方填入電腦的區域網位址後開始串流。",
        ],
        .discoveryEntitlementHint: [
            .english: "iOS blocks UDP broadcast without Apple's Multicast Networking entitlement, so this build sweeps the local subnet with unicast probes instead.",
            .simplifiedChinese: "iOS 在没有 Apple 多播网络授权时会拦截 UDP 广播，因此本版本改用单播扫描局域网地址。",
            .traditionalChinese: "iOS 在沒有 Apple 多播網絡授權時會攔截 UDP 廣播，因此本版本改用單播掃描區域網位址。",
        ],
        .sweepProgress: [
            .english: "Sweeping %1$d local addresses…",
            .simplifiedChinese: "正在扫描 %1$d 个局域网地址…",
            .traditionalChinese: "正在掃描 %1$d 個區域網位址…",
        ],
        .captureTitle: [
            .english: "2 · Choose capture",
            .simplifiedChinese: "2 · 选择采集",
            .traditionalChinese: "2 · 選擇採集",
        ],
        .modeHighSpeed: [
            .english: "High-speed session",
            .simplifiedChinese: "高速会话",
            .traditionalChinese: "高速會話",
        ],
        .modeCamera: [
            .english: "Camera",
            .simplifiedChinese: "摄像头",
            .traditionalChinese: "攝像頭",
        ],
        .highSpeedHelp: [
            .english: "Hardware H.264 over TCP, full frame, no crop. The PC accepts 1280 × 720 only.",
            .simplifiedChinese: "硬件 H.264 over TCP，完整画面，不裁剪。电脑端仅接受 1280 × 720。",
            .traditionalChinese: "硬件 H.264 over TCP，完整畫面，不裁剪。電腦端僅接受 1280 × 720。",
        ],
        .cameraHelp: [
            .english: "Software JPEG over UDP. Crop before compression and adjust quality to reduce bandwidth and latency.",
            .simplifiedChinese: "软件 JPEG over UDP。可在压缩前裁剪，并调整质量以降低带宽与时延。",
            .traditionalChinese: "軟件 JPEG over UDP。可在壓縮前裁剪，並調整質量以降低頻寬與時延。",
        ],
        .cameraLabel: [
            .english: "Camera",
            .simplifiedChinese: "摄像头",
            .traditionalChinese: "攝像頭",
        ],
        .frontCamera: [
            .english: "Front camera · %1$@",
            .simplifiedChinese: "前置摄像头 · %1$@",
            .traditionalChinese: "前置攝像頭 · %1$@",
        ],
        .backCamera: [
            .english: "Rear camera · %1$@",
            .simplifiedChinese: "后置摄像头 · %1$@",
            .traditionalChinese: "後置攝像頭 · %1$@",
        ],
        .resolutionLabel: [
            .english: "Resolution",
            .simplifiedChinese: "分辨率",
            .traditionalChinese: "解析度",
        ],
        .frameRateLabel: [
            .english: "Target frame rate",
            .simplifiedChinese: "目标帧率",
            .traditionalChinese: "目標幀率",
        ],
        .formatsHelp: [
            .english: "%1$d resolutions · %2$d frame rates. Resolution is the capture size before optional Camera-mode cropping.",
            .simplifiedChinese: "%1$d 种分辨率 · %2$d 种帧率。分辨率为裁剪前的采集尺寸。",
            .traditionalChinese: "%1$d 種解析度 · %2$d 種幀率。解析度為裁剪前的採集尺寸。",
        ],
        .noHighSpeed: [
            .english: "No high-speed / PC-compatible combination for this camera. Choose another camera or Camera mode.",
            .simplifiedChinese: "该摄像头没有可用且与电脑兼容的高速组合。请更换摄像头或改用摄像头模式。",
            .traditionalChinese: "該攝像頭沒有可用且與電腦兼容的高速組合。請更換攝像頭或改用攝像頭模式。",
        ],
        .noCameraOptions: [
            .english: "No usable combination for this camera. Choose another camera or refresh.",
            .simplifiedChinese: "该摄像头没有可用组合。请更换摄像头或刷新。",
            .traditionalChinese: "該攝像頭沒有可用組合。請更換攝像頭或重新整理。",
        ],
        .noCombinations: [
            .english: "No available combinations",
            .simplifiedChinese: "没有可用组合",
            .traditionalChinese: "沒有可用組合",
        ],
        .noCameras: [
            .english: "No cameras found. Grant camera permission and refresh.",
            .simplifiedChinese: "未找到摄像头。请授予相机权限后刷新。",
            .traditionalChinese: "未找到攝像頭。請授予相機權限後重新整理。",
        ],
        .readingCameras: [
            .english: "Checking supported camera modes…",
            .simplifiedChinese: "正在检查支持的摄像头模式…",
            .traditionalChinese: "正在檢查支持的攝像頭模式…",
        ],
        .refreshFormats: [
            .english: "Refresh available combinations",
            .simplifiedChinese: "刷新可用组合",
            .traditionalChinese: "重新整理可用組合",
        ],
        .h264PCUnsupported: [
            .english: "%1$@ is not accepted by the PC receiver (1280 × 720 only)",
            .simplifiedChinese: "电脑端不接受 %1$@（仅支持 1280 × 720）",
            .traditionalChinese: "電腦端不接受 %1$@（僅支持 1280 × 720）",
        ],
        .h264PCSupportedNote: [
            .english: "High-speed mode mirrors the Android app: 1280 × 720, hardware H.264 over TCP, no crop. Other sizes are listed but disabled because the PC receiver rejects them before decode.",
            .simplifiedChinese: "高速模式与 Android 端一致：1280 × 720、硬件 H.264 over TCP、不裁剪。其他尺寸会被电脑端在解码前拒绝，因此仅列出而不可选。",
            .traditionalChinese: "高速模式與 Android 端一致：1280 × 720、硬件 H.264 over TCP、不裁剪。其他尺寸會被電腦端在解碼前拒絕，因此僅列出而不可選。",
        ],
        .orientationTitle: [
            .english: "Orientation and crop",
            .simplifiedChinese: "方向与裁剪",
            .traditionalChinese: "方向與裁剪",
        ],
        .rotationLabel: [
            .english: "Frame rotation",
            .simplifiedChinese: "画面旋转",
            .traditionalChinese: "畫面旋轉",
        ],
        .rotationAuto: [
            .english: "Auto (recommended)",
            .simplifiedChinese: "自动（推荐）",
            .traditionalChinese: "自動（推薦）",
        ],
        .cropLabel: [
            .english: "Crop edges after rotation (%)",
            .simplifiedChinese: "旋转后裁剪边缘（%）",
            .traditionalChinese: "旋轉後裁剪邊緣（%）",
        ],
        .cropLeft: [.english: "Left (%)", .simplifiedChinese: "左（%）", .traditionalChinese: "左（%）"],
        .cropRight: [.english: "Right (%)", .simplifiedChinese: "右（%）", .traditionalChinese: "右（%）"],
        .cropTop: [.english: "Top (%)", .simplifiedChinese: "上（%）", .traditionalChinese: "上（%）"],
        .cropBottom: [.english: "Bottom (%)", .simplifiedChinese: "下（%）", .traditionalChinese: "下（%）"],
        .cropHelp: [
            .english: "Remove 0–45% per edge, relative to the rotated view. Only the retained area is copied and compressed; there is no resizing. Values are rounded to even pixels. Keep PC rotation on Auto.",
            .simplifiedChinese: "每边可去除 0–45%（相对旋转后的画面）。仅保留区域会被复制与压缩，不做缩放。数值取整到偶数像素。电脑端旋转请保持自动。",
            .traditionalChinese: "每邊可去除 0–45%（相對旋轉後的畫面）。僅保留區域會被複製與壓縮，不做縮放。數值取整到偶數像素。電腦端旋轉請保持自動。",
        ],
        .cropInvalid: [
            .english: "Enter a crop percentage from 0 to 45 for each edge.",
            .simplifiedChinese: "请输入 0 到 45 之间的裁剪百分比。",
            .traditionalChinese: "請輸入 0 到 45 之間的裁剪百分比。",
        ],
        .cropOutput: [
            .english: "Received view: %1$d × %2$d · clockwise rotation %3$d°",
            .simplifiedChinese: "接收画面：%1$d × %2$d · 顺时针旋转 %3$d°",
            .traditionalChinese: "接收畫面：%1$d × %2$d · 順時針旋轉 %3$d°",
        ],
        .jpegQualityLabel: [
            .english: "Software JPEG quality",
            .simplifiedChinese: "软件 JPEG 质量",
            .traditionalChinese: "軟件 JPEG 質量",
        ],
        .jpegQualityHelp: [
            .english: "Lower quality reduces data size; higher preserves detail. Default Q80. Encoding is done on the GPU, not in hardware JPEG.",
            .simplifiedChinese: "质量越低数据量越小，越高细节越完整。默认 Q80。使用 GPU 编码，并非硬件 JPEG。",
            .traditionalChinese: "質量越低數據量越小，越高細節越完整。預設 Q80。使用 GPU 編碼，並非硬件 JPEG。",
        ],
        .intrinsicsTitle: [
            .english: "Camera intrinsics",
            .simplifiedChinese: "相机内参",
            .traditionalChinese: "相機內參",
        ],
        .intrinsicsSource: [
            .english: "Source: %1$@",
            .simplifiedChinese: "来源：%1$@",
            .traditionalChinese: "來源：%1$@",
        ],
        .intrinsicsFovReference: [
            .english: "Field-of-view reference axis",
            .simplifiedChinese: "视场角参考轴",
            .traditionalChinese: "視場角參考軸",
        ],
        .intrinsicsFovHorizontal: [
            .english: "Horizontal",
            .simplifiedChinese: "水平",
            .traditionalChinese: "水平",
        ],
        .intrinsicsFovDiagonal: [
            .english: "Diagonal",
            .simplifiedChinese: "对角",
            .traditionalChinese: "對角",
        ],
        .intrinsicsLimitation: [
            .english: "AVFoundation exposes no per-format factory calibration for the video path, so fx/fy are derived from the active format's field of view and the principal point is the frame centre. AVCameraCalibrationData is the only exact source and requires a still capture. Verify the reference axis on the device.",
            .simplifiedChinese: "AVFoundation 的视频路径不提供出厂标定数据，因此 fx/fy 由当前格式的视场角推导，主点为画面中心。AVCameraCalibrationData 是唯一精确来源，但需要拍照。请在真机验证参考轴。",
            .traditionalChinese: "AVFoundation 的視頻路徑不提供出廠標定數據，因此 fx/fy 由當前格式的視場角推導，主點為畫面中心。AVCameraCalibrationData 是唯一精確來源，但需要拍照。請在真機驗證參考軸。",
        ],
        .startStream: [
            .english: "Start streaming",
            .simplifiedChinese: "开始串流",
            .traditionalChinese: "開始串流",
        ],
        .stopStream: [
            .english: "Stop",
            .simplifiedChinese: "停止",
            .traditionalChinese: "停止",
        ],
        .ready: [
            .english: "Ready. Start when the PC receiver is open.",
            .simplifiedChinese: "就绪。请在电脑端接收器打开后开始。",
            .traditionalChinese: "就緒。請在電腦端接收器打開後開始。",
        ],
        .opening: [
            .english: "Opening %1$@…",
            .simplifiedChinese: "正在打开 %1$@…",
            .traditionalChinese: "正在打開 %1$@…",
        ],
        .streaming: [
            .english: "Streaming · %1$@",
            .simplifiedChinese: "串流中 · %1$@",
            .traditionalChinese: "串流中 · %1$@",
        ],
        .stopped: [
            .english: "Stopped",
            .simplifiedChinese: "已停止",
            .traditionalChinese: "已停止",
        ],
        .hostRequired: [
            .english: "Enter a PC address or use Find PC.",
            .simplifiedChinese: "请输入电脑地址或使用查找电脑。",
            .traditionalChinese: "請輸入電腦位址或使用尋找電腦。",
        ],
        .portInvalid: [
            .english: "Use a port from 1 to 65535.",
            .simplifiedChinese: "请使用 1 到 65535 之间的端口。",
            .traditionalChinese: "請使用 1 到 65535 之間的埠。",
        ],
        .permissionDenied: [
            .english: "Camera permission is required. Grant it in Settings, then retry.",
            .simplifiedChinese: "需要相机权限。请在设置中授予后重试。",
            .traditionalChinese: "需要相機權限。請在設定中授予後重試。",
        ],
        .cameraReadFailed: [
            .english: "Could not read camera capabilities. Close other camera apps and refresh.",
            .simplifiedChinese: "无法读取摄像头能力。请关闭其他相机应用后刷新。",
            .traditionalChinese: "無法讀取攝像頭能力。請關閉其他相機應用後重新整理。",
        ],
        .combinationFailed: [
            .english: "This combination could not start and has been excluded for now. Select another one or refresh to retry.",
            .simplifiedChinese: "该组合无法启动，已暂时排除。请选择其他组合或刷新重试。",
            .traditionalChinese: "該組合無法啟動，已暫時排除。請選擇其他組合或重新整理重試。",
        ],
        .connectionFailed: [
            .english: "Connection stopped. Check the PC receiver, address and network, then start again.",
            .simplifiedChinese: "连接已中断。请检查电脑端接收器、地址与网络后重新开始。",
            .traditionalChinese: "連接已中斷。請檢查電腦端接收器、位址與網絡後重新開始。",
        ],
        .cameraFailed: [
            .english: "Camera stopped. Close other camera apps and try again.",
            .simplifiedChinese: "摄像头已停止。请关闭其他相机应用后重试。",
            .traditionalChinese: "攝像頭已停止。請關閉其他相機應用後重試。",
        ],
        .intrinsicsFailed: [
            .english: "Camera geometry could not be sent. Try another camera or reconnect to the PC.",
            .simplifiedChinese: "无法发送相机几何参数。请更换摄像头或重新连接电脑。",
            .traditionalChinese: "無法發送相機幾何參數。請更換攝像頭或重新連接電腦。",
        ],
        .streamHint: [
            .english: "Keep this app in the foreground while streaming. The screen stays awake.",
            .simplifiedChinese: "串流时请保持本应用在前台。屏幕会保持常亮。",
            .traditionalChinese: "串流時請保持本應用在前台。屏幕會保持常亮。",
        ],
        .performanceTitle: [
            .english: "Performance monitor",
            .simplifiedChinese: "性能监视",
            .traditionalChinese: "性能監視",
        ],
        .performanceDetails: [
            .english: "Show / hide details",
            .simplifiedChinese: "显示 / 隐藏详情",
            .traditionalChinese: "顯示 / 隱藏詳情",
        ],
        .performanceSummary: [
            .english: "Output %1$.1f FPS · Sent %2$.1f FPS",
            .simplifiedChinese: "采集 %1$.1f FPS · 发送 %2$.1f FPS",
            .traditionalChinese: "採集 %1$.1f FPS · 發送 %2$.1f FPS",
        ],
        .captureFPS: [
            .english: "Available camera frames",
            .simplifiedChinese: "可用相机帧",
            .traditionalChinese: "可用相機幀",
        ],
        .sentFPS: [
            .english: "Sent frames",
            .simplifiedChinese: "已发送帧",
            .traditionalChinese: "已發送幀",
        ],
        .videoRate: [
            .english: "Video data rate",
            .simplifiedChinese: "视频数据率",
            .traditionalChinese: "視頻數據率",
        ],
        .encodeAge: [
            .english: "Capture → encoded frame",
            .simplifiedChinese: "采集 → 编码完成",
            .traditionalChinese: "採集 → 編碼完成",
        ],
        .exposure: [
            .english: "Camera exposure",
            .simplifiedChinese: "相机曝光",
            .traditionalChinese: "相機曝光",
        ],
        .performanceNote: [
            .english: "Phone-side metrics only. High-speed output counts encoded packets; Camera mode counts frames delivered to the app. Capture-to-encode timing uses the host clock and excludes network, PC processing and display delay.",
            .simplifiedChinese: "仅为手机端指标。高速模式统计已编码包数；摄像头模式统计交付给应用的帧数。采集到编码的时延使用主机时钟，不含网络、电脑处理与显示延迟。",
            .traditionalChinese: "僅為手機端指標。高速模式統計已編碼包數；攝像頭模式統計交付給應用的幀數。採集到編碼的時延使用主機時鐘，不含網絡、電腦處理與顯示延遲。",
        ],
        .notAvailable: [
            .english: "Not available",
            .simplifiedChinese: "不可用",
            .traditionalChinese: "不可用",
        ],
        .languageLabel: [
            .english: "Language",
            .simplifiedChinese: "语言",
            .traditionalChinese: "語言",
        ],
        .dismiss: [
            .english: "Dismiss",
            .simplifiedChinese: "关闭",
            .traditionalChinese: "關閉",
        ],
    ]
}
