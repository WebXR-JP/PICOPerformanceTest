# PICOPerformanceTest

PICO4（OpenXR + Vulkan）頂点負荷テストアプリ。

## 概要

- **目的**: GPU頂点処理性能の計測（gprobe GPUトレース用の負荷生成）
- **描画**: グリッドメッシュ（GRID_N×GRID_N）を両目にステレオ描画
- **計測**: フレームタイム（ms）・FPS・ポリゴン数を logcat に毎秒出力

## ビルド環境

- **Android SDK**: `C:/Users/halby/AppData/Local/Android/Sdk`
- **NDK**: 26.1.10909125（Gradleが自動選択）
- **AGP**: 8.4.0
- **glslc**: NDK 27.2.12479018 の `shader-tools/windows-x86_64/glslc.exe`（CMakeが自動検出）
- **OpenXR SDK**: PICO OpenXR SDK 3.0.0 (`%20`を含むパスに注意)

## パス設定

`CMakeLists.txt` は以下の優先順位でOpenXR SDKを探す：
1. `P:` ドライブ（`subst P:` でマウント済みの場合）
2. 直接絶対パス（`%20`を含む `PICO_Developer_Center_Downloads` 以下）

## ビルド手順

```bash
# buildStagingDirectory は C:/PICO_bld/performancetest
mkdir -p /c/PICO_bld/performancetest

cd /e/Repos/cpp/CG/PICOPerformanceTest/project/android
./gradlew :assembleDebug
```

### インストールと起動

```bash
adb install -r project/android/build/outputs/apk/debug/PICOPerformanceTest-debug.apk
adb shell am start -n com.example.picoperftest/android.app.NativeActivity
```

### logcatで計測結果を確認

```bash
adb logcat -s PICOPerfTest
```

出力例:
```
[PERF] FPS=72.0  FrameTime=13.89ms  Polygons=1000898  Vertices=502500
```

## 負荷設定（CMakeLists.txt を編集）

`GRID_N` の値でポリゴン数を変更できる：

| GRID_N | ポリゴン数（約） |
|--------|----------------|
| 708    | 100万          |
| 1584   | 500万          |
| 2237   | 1000万         |

`CMakeLists.txt` の `target_compile_definitions` に追加：
```cmake
-DGRID_N=708   # デフォルト（100万ポリゴン）
```

または `build.gradle` の cmake arguments に追加：
```groovy
arguments "-DANDROID_STL=c++_shared", "-DGRID_N=708"
```

## プロジェクト構成

```
PICOPerformanceTest/
├── cmake/embed_spirv.cmake   # SPIR-V → C++ヘッダー変換スクリプト
├── shaders/
│   ├── vertex.vert           # 頂点シェーダー（MVP変換のみ）
│   └── fragment.frag         # フラグメントシェーダー（単色）
├── cpp/main.cpp              # 全実装（OpenXR + Vulkan + 計測）
├── CMakeLists.txt
└── project/android/          # Androidプロジェクト
    ├── AndroidManifest.xml
    ├── build.gradle
    ├── settings.gradle
    └── local.properties      # sdk.dir を設定
```
