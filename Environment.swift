// Environment.swift — promptless macOS senses for the fly:
// window terrain + window-appearance looms (CGWindowList), circadian clock,
// user-idle detection (CGEventSource), thermal-state "temperature", and the
// market sense: a small JSON file the Trader Fly organism (trader/) writes.
// None of these trigger a TCC permission dialog.

import Cocoa

// A walkable window top edge, in scene coordinates (origin at screen center).
struct Ledge {
    let y: CGFloat
    let x0: CGFloat
    let x1: CGFloat
    let id: Int
}

final class WindowSense {
    struct Snapshot {
        let ledges: [Ledge]
        let newWindows: [(center: CGPoint, size: CGFloat)]
    }

    private var knownIDs = Set<Int>()
    private var first = true
    private let myPID = NSRunningApplication.current.processIdentifier

    func poll(screen: NSRect) -> Snapshot {
        guard let info = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements],
                                                    kCGNullWindowID) as? [[String: Any]] else {
            return Snapshot(ledges: [], newWindows: [])
        }
        // CG global coords are top-left-of-primary; Cocoa are bottom-left-of-primary.
        let primaryH = NSScreen.screens.first(where: { $0.frame.origin == .zero })?.frame.height
            ?? screen.height
        let W = screen.width, H = screen.height
        var ledges: [Ledge] = []
        var newWins: [(CGPoint, CGFloat)] = []
        var ids = Set<Int>()
        for w in info {
            guard (w["kCGWindowLayer"] as? Int) == 0,                     // normal windows only
                  (w["kCGWindowOwnerPID"] as? Int32) != myPID,
                  ((w["kCGWindowAlpha"] as? Double) ?? 1) > 0.05,
                  let b = w["kCGWindowBounds"] as? [String: Any],
                  let rect = CGRect(dictionaryRepresentation: b as CFDictionary),
                  rect.width >= 160, rect.height >= 60,
                  let num = w["kCGWindowNumber"] as? Int else { continue }
            ids.insert(num)
            // only windows on the fly's current display
            let cocoaRect = NSRect(x: rect.minX, y: primaryH - rect.maxY,
                                   width: rect.width, height: rect.height)
            guard cocoaRect.intersects(screen) else { continue }
            // scene coords: centered on this display, y up
            let topY = (primaryH - rect.minY) - screen.midY
            let x0 = max(rect.minX - screen.midX, -W / 2 + 15)
            let x1 = min(rect.maxX - screen.midX, W / 2 - 15)
            if topY < H / 2 - 8, topY > -H / 2 + 8, x1 - x0 > 100, ledges.count < 12 {
                ledges.append(Ledge(y: topY, x0: x0, x1: x1, id: num))
            }
            if !first && !knownIDs.contains(num) {
                let center = CGPoint(x: rect.midX - screen.midX,
                                     y: (primaryH - rect.midY) - screen.midY)
                newWins.append((center, max(rect.width, rect.height)))
            }
        }
        knownIDs = ids
        first = false
        return Snapshot(ledges: ledges, newWindows: newWins)
    }
}

// Drosophila circadian activity: morning and evening peaks, midday siesta,
// night quiescence. Returns a multiplier for the sim's baseline drive.
func circadianActivity(hour: Double) -> Float {
    let pts: [(Double, Float)] = [(0, 0.25), (5, 0.25), (8, 1.0), (10, 1.0), (13, 0.55),
                                  (15, 0.55), (17, 1.0), (20, 1.0), (23, 0.3), (24, 0.25)]
    for i in 0..<(pts.count - 1) where hour >= pts[i].0 && hour <= pts[i + 1].0 {
        let t = Float((hour - pts[i].0) / max(0.001, pts[i + 1].0 - pts[i].0))
        return pts[i].1 + (pts[i + 1].1 - pts[i].1) * t
    }
    return 0.25
}

// Seconds since the user last touched mouse or keyboard. CGEventSource idle
// queries are permission-free (they reveal when, never what).
func userIdleSeconds() -> CGFloat {
    let types: [CGEventType] = [.mouseMoved, .leftMouseDown, .keyDown, .scrollWheel]
    let idles = types.map { CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: $0) }
    return CGFloat(idles.min() ?? 0)
}

// Flies are ectotherms: a hot Mac is a fast fly.
func thermalTempo() -> CGFloat {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return 1.0
    case .fair: return 1.15
    case .serious: return 1.35
    case .critical: return 1.5
    @unknown default: return 1.0
    }
}

// MARK: - Market sense (Trader Fly bridge)

// The trader/ organism writes ~/.desktopfly/market_state.json after every decision
// (see trader/ui/state.py). The fly reads only three things from it: a behavior
// word, an arousal level and the write time. It is a *stimulus*, not a command:
// ESCAPE becomes one abrupt looming step into the LC4/LPLC2 pathway and the
// connectome decides whether the giant fiber fires; ALERT holds a sub-escape
// looming floor (nervous, wings up) and keeps the fly awake; CALM changes nothing.
struct MarketSignals {
    enum Behavior: String {
        case sleep, explore, alert, escape, hunt_long, hunt_short, rest, reward, aversive, unknown
    }
    var behavior: Behavior
    var arousal: CGFloat        // 0..1, the organism's own arousal
    var decision: String        // CALM / ALERT / ESCAPE or LONG / SHORT / NO_TRADE (display only)
    var updated: TimeInterval   // unix seconds the file was written
    var age: TimeInterval       // seconds since `updated` at poll time
    var fresh: Bool { age < MARKET_STALE_SECONDS }
    var threatening: Bool { behavior == .escape || behavior == .aversive }
    var keepsAwake: Bool { threatening || behavior == .alert }
}

// a state older than this is ignored: a finished replay must not freeze the fly in ESCAPE
let MARKET_STALE_SECONDS: TimeInterval = 300

final class MarketSense {
    static func defaultPath() -> String {
        if let p = ProcessInfo.processInfo.environment["DESKTOPFLY_MARKET_STATE"], !p.isEmpty { return p }
        return NSHomeDirectory() + "/.desktopfly/market_state.json"
    }

    let path: String
    init(path: String = MarketSense.defaultPath()) { self.path = path }

    func poll(now: Date = Date()) -> MarketSignals? {
        guard let data = FileManager.default.contents(atPath: path) else { return nil }
        return MarketSense.parse(data, now: now)
    }

    static func parse(_ data: Data, now: Date = Date()) -> MarketSignals? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let behavior = MarketSignals.Behavior(rawValue: (obj["behavior"] as? String) ?? "") ?? .unknown
        let arousal = CGFloat((obj["arousal"] as? Double) ?? 0)
        let updated = (obj["updated"] as? Double) ?? 0
        let decision = (obj["decision"] as? String) ?? ""
        let age: TimeInterval = updated > 0 ? now.timeIntervalSince1970 - updated : .infinity
        return MarketSignals(behavior: behavior, arousal: clampf(arousal, 0, 1), decision: decision,
                             updated: updated, age: age)
    }
}
