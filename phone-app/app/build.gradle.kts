plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.eyetracing.android"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.eyetracing.android"
        minSdk = 26
        targetSdk = 35
        versionCode = 33
        versionName = "0.16.3-startup-fix"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // Both bundled languages must remain available to the in-app switch.
    bundle {
        language { enableSplit = false }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-ktx:1.10.1")
    testImplementation("junit:junit:4.13.2")
}
