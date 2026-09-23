import SwiftUI

/// Phone-side metrics, mirroring Android's performance monitor.
///
/// Every number here is measured on the phone. Network transit, PC decode and
/// display latency are deliberately absent because they cannot be observed from
/// this side; the PC control centre reports those.
struct PerformanceSectionView: View {
    @EnvironmentObject private var model: AppModel
    @State private var showsDetails = false

    var body: some View {
        Section(model.text(.performanceTitle)) {
            Text(model.text(
                .performanceSummary,
                model.rates.captureFPS,
                model.rates.sentFPS
            ))
            .font(.subheadline)

            Toggle(model.text(.performanceDetails), isOn: $showsDetails)

            if showsDetails {
                metric(model.text(.captureFPS), value: model.rates.captureFPS, unit: "FPS")
                metric(model.text(.sentFPS), value: model.rates.sentFPS, unit: "FPS")
                metric(model.text(.videoRate), value: model.rates.megabitsPerSecond, unit: "Mbps")
                metric(model.text(.encodeAge), value: model.rates.encodingAgeMs, unit: "ms")
                metric(model.text(.exposure), value: model.rates.exposureMs, unit: "ms")

                Text(model.text(.performanceNote))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func metric(_ title: String, value: Double?, unit: String) -> some View {
        HStack {
            Text(title).font(.footnote)
            Spacer()
            Text(formatted(value: value, unit: unit)).font(.footnote).monospacedDigit()
        }
    }

    private func formatted(value: Double?, unit: String) -> String {
        guard let value, value.isFinite else { return model.text(.notAvailable) }
        return String(format: "%.1f %@", value, unit)
    }
}
