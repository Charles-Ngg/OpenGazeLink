import SwiftUI

@main
struct OpenGazeLinkApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .onAppear { model.onAppear() }
        }
    }
}
